import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')


class TensorflowYoloModel():
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.saved_model_loaded = None
        self.infer_func = None

    def load_model(self):
        self.saved_model_loaded = tf.saved_model.load(self.model_path, tags=[tag_constants.SERVING])
        self.infer_func = self.saved_model_loaded.signatures['serving_default']

    def infer(self, input):
        batch_data = tf.constant(input)
        pred_bbox = self.infer_func(batch_data)
        return pred_bbox


def preprocess_img_obj(img):
    img = cv2.resize(img, (FLAGS.size, FLAGS.size))
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_data = img / 255
    return image_data


def convert_to_mins_maxes(box_xywh, input_shape=np.array([416, 416])):
    """
    Converts boxes of yolo output that comes as (x_center, y_center, w, h) to (y_min, x_min, y_max, x_max).
    """
    box_xy, box_wh = np.array_split(box_xywh, 2, axis=-1)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    return boxes


def non_max_suppression(bounding_boxes, class_scores, iou_threshold=0.7, scores_threshold=0.2):
    """
    performs non max suppression. based on Malisiewicz et al. Code from https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    alternative: https://nms.readthedocs.io/en/latest/_modules/nms/malisiewicz.html
    :param bounding_boxes: bounding boxes in shape (num_boxes,4) as (y1,x1,y2,x2)
    :param class_scores: class scores in shape (num_boxes, num_classes)
    :param iou_threshold:
    :param scores_threshold:
    :return:
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    ## filter all boxes that do not meet confidence threshold
    # get max indices (class ids)
    max_scores_indices = np.argmax(class_scores, axis=1)
    # flatten to highest scores
    max_scores_values = class_scores[np.arange(len(class_scores)), max_scores_indices]
    # filter for threshold
    filtered_scores_indices = np.nonzero(max_scores_values >= scores_threshold)[0]

    filtered_classes = np.take(max_scores_indices, filtered_scores_indices)
    filtered_scores = np.take(max_scores_values, filtered_scores_indices)
    filtered_boxes = np.take(bounding_boxes, filtered_scores_indices, axis=0)

    # coordinates of bounding boxes
    start_x = filtered_boxes[:, 0]
    start_y = filtered_boxes[:, 1]
    end_x = filtered_boxes[:, 2]
    end_y = filtered_boxes[:, 3]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_classes = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(filtered_scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # highest_class_confidence_idx = np.argmax(class_scores[index])
        # if class_scores[index][highest_class_confidence_idx] < scores_threshold:
        #     break
        # Pick the bounding box with largest confidence score
        picked_boxes.append(filtered_boxes[index])
        picked_score.append(filtered_scores[index])
        picked_classes.append(filtered_classes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < iou_threshold)
        order = order[left]
    return picked_boxes, picked_score, picked_classes


def main(_argv):
    print(FLAGS.weights)
    print(FLAGS.size)
    print(FLAGS.image)

    model = TensorflowYoloModel(FLAGS.weights)

    image = cv2.imread(FLAGS.image)
    image = preprocess_img_obj(image)

    pred_bbox = model.infer(image)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    print('\n\nTESTE')
    print(boxes, pred_conf)
    print('\n\nFIM TESTE')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

