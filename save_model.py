import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes, convert_to_mins_maxes
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('filter_boxes', False, 'filter boxes that do not meet confidence threshold')
flags.DEFINE_boolean('nms', False, 'perform combined non max supression on model output')
flags.DEFINE_boolean('export_mins_maxes', False, 'return boxes as (y1,x1,y2,x2) instead of (x_center,y_center,w,h)')

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    if FLAGS.model ==  'yolov4-tiny-3l':
      for i, fm in enumerate(feature_maps):
        # import pdb; pdb.set_trace()
        if i == 0:
          output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        elif i == 1:
          output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
          output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])    
    else:
      for i, fm in enumerate(feature_maps):
        if i == 0:
          output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
          output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])  
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    if FLAGS.filter_boxes:
      pred_bbox, pred_prob = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres)
    
    if FLAGS.nms or FLAGS.export_mins_maxes:
      pred_bbox = convert_to_mins_maxes(pred_bbox)
      if FLAGS.nms:
        pred_bbox, pred_prob, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=pred_bbox,
            scores=pred_prob,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.7,
            score_threshold=0.8
        )

    pred = (pred_bbox, pred_prob)
  
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
  model.summary()
  model.save(FLAGS.output)

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
