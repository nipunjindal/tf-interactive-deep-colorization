import tensorflow as tf
import os
import glob

def _parse_function(sketch_filename, color_filename):
    sketch_image_string = tf.read_file(sketch_filename)
    sketch_image_decoded = tf.image.decode_png(sketch_image_string)
    sketch_image_resized = tf.image.resize_images(sketch_image_decoded, [256, 256])
    sketch_image = tf.image.rgb_to_grayscale(sketch_image_resized)

    color_image_string = tf.read_file(color_filename)
    color_image_decoded = tf.image.decode_png(color_image_string)
    color_image_resized = tf.image.resize_images(color_image_decoded, [256, 256])
    
    return sketch_image, color_image_resized

def _parse_color_function(filename):
    
    return image_resized

def input_fn(dir_sketch, dir_color, batch_size):
    
    current_path = os.getcwd()
    os.chdir(dir_sketch)
    filenames = glob.glob('*.png')
    os.chdir(current_path)

    feature_filename_dataset = [dir_sketch + '/' + filename for filename in filenames]
    label_filename_dataset = [dir_color + '/' + filename for filename in filenames]
    
    dataset = tf.data.Dataset.from_tensor_slices((feature_filename_dataset, label_filename_dataset))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


