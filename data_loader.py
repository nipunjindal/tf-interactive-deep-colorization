import tensorflow as tf
import os
import glob

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [256, 256])
    return image_resized

def input_fn(dir_sketch, dir_color, batch_size):
    
    current_path = os.getcwd()
    os.chdir(dir_sketch)

    filenames = glob.glob('*.png')

    feature_filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    feature_image_dataset = feature_filename_dataset.map(_parse_function)

    os.chdir(dir_color)
    label_filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    label_image_dataset = label_filename_dataset.map(_parse_function)

    os.chdir(current_path)
    dataset = tf.data.Dataset.from_tensor_slices((feature_image_dataset, label_image_dataset))
    dataset = dataset.shuffle(1000).repeat().batch_size(batch_size)

    return dataset


