from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from functools import partial
import numpy as np
import tensorflow as tf


flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-trt-fp16-416', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float16', 'quantize mode (int8, float16)')
flags.DEFINE_string('dataset', "/media/user/Source/Data/coco_dataset/coco/5k.txt", 'path to dataset')
flags.DEFINE_integer('loop', 8, 'loop')


def convert_to_tensorflow_trt(tf_model_path, output_model_path, optimize_offline=False,
                              precision_mode=trt.TrtPrecisionMode.FP16):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode,
        # set trt workspace size to 4GB
        max_workspace_size_bytes=1 << 32,
        max_batch_size=1)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=tf_model_path, conversion_params=conversion_params)
    converter._enable_ = True

    if precision_mode == trt.TrtPrecisionMode.INT8:
        converted_model = converter.convert(calibration_input_fn=partial(input_fn, 5, "Calibration"))
    else:
        converted_model = converter.convert()

    if optimize_offline:
        converter.build(input_fn=partial(input_fn, 1, "Building"))

    converter.save(output_saved_model_dir=output_model_path)
    print('Done Converting to TF-TRT')
    return converted_model


# taken partially from https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/convert_trt.py
def get_dataset(batch_size,
                input_size,
                dtype=tf.float32):
    features = np.random.normal(
        loc=112, scale=70,
        size=(batch_size, input_size, input_size, 3)).astype(np.float32)
    features = np.clip(features, 0.0, 255.0).astype(dtype.as_numpy_dtype)
    features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
        "features", initializer=tf.constant(features)))
    dataset = tf.data.Dataset.from_tensor_slices([features])
    dataset = dataset.repeat()
    return dataset


def input_fn(num_iterations, model_phase):
    dataset = get_dataset(batch_size=1, input_size=416)
    for i, batch_images in enumerate(dataset):
        if i >= num_iterations:
            break
        yield batch_images,
        print("* [%s] - step %02d/%02d" % (model_phase, i + 1, num_iterations))
        i += 1


def main(_argv):
  convert_to_tensorflow_trt(FLAGS.weights, FLAGS.output)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


