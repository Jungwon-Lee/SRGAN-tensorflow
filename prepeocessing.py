import numpy
import tensorflow as tf

def scaling(input_image):
    input_image = input_image / 127.5 - 1.
    return input_image


def random_flip(input_image):
#     print(input_image)
    if np.random.random() < 0.5:
        input_image = tf.image.flip_left_right(input_image)
    return input_image

# resize input
def process_input(input, input_size):
    return tf.image.resize(input, [input_size, input_size], method="area")