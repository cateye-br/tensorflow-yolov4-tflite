import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import colorsys

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
    model.load_model()

    original_image = cv2.imread(FLAGS.image)
    image = preprocess_img_obj(original_image)

    images_data = [image]
    images_data = np.asarray(images_data).astype(np.float32)

    pred_bbox = model.infer(images_data)

    boxes = []
    pred_conf = []
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]


    print(boxes.shape)
    print(pred_conf.shape)
    

    boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4))
    scores = tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1]))

    #boxes x,y,w,h para boxes x1,y1,x2,y2
    bboxes = convert_to_mins_maxes(boxes)

    print("MIN MAX BOXES", bboxes[0])

    picked_boxes, picked_score, picked_classes = non_max_suppression(bboxes, scores)

    print("FINAL BOXES", picked_boxes, picked_score, picked_classes)

    num_classes = 4
    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i in range(picked_boxes):
        fontScale = 0.5
        score = picked_score[0][i]
        class_ind = int(picked_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (picked_boxes[1], picked_boxes[0]), (picked_boxes[3], picked_boxes[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        bbox_mess = '%s: %.2f' % (picked_classes[class_ind], score)
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

        cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

