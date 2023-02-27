import tensorflow as tf
import tensorflow_probability as tfp


DIST = tfp.distributions.Categorical([0.5,0.5])

def flip_left_right(image_1, image_2):
    do_flip = DIST.sample()
    out = tf.cond(
        tf.equal(do_flip, 1),
        lambda : _do_flip(image_1, image_2),
        lambda : (image_1 , image_2)
        )
    return out[0], out[1]


def flip_up_down(image_1, image_2):
    do_flip = DIST.sample()
    out = tf.cond(
        tf.equal(do_flip, 1),
        lambda : _do_flip_ud(image_1, image_2),
        lambda : (image_1 , image_2)
        )
    return out[0], out[1]




def _do_flip_ud(image_1, image_2):
    flipped_1 = tf.image.flip_up_down(image_1)
    flipped_2 = tf.image.flip_up_down(image_2)
    return flipped_1, flipped_2


def _do_flip(image_1, image_2):
    flipped_1 = tf.image.flip_left_right(image_1)
    flipped_2 = tf.image.flip_left_right(image_2)
    return flipped_1, flipped_2

def normalize_patches(patch_1, patch_2):
    patch_1 = tf.cast(patch_1, tf.float32)
    patch_2 = tf.cast(patch_2, tf.float32)
    patch_1 = patch_1/255.0
    patch_2 = patch_2/255.0
    patch_1*=2.0
    patch_2*=2.0
    patch_1 = patch_1 - 1
    patch_2 = patch_2 - 1
    return patch_1, patch_2

def adjust_brightness(image_1, image_2):
    do_adjust = DIST.sample()
    out = tf.cond(
        tf.equal(do_adjust, 1),
        lambda : _brightness(image_1, image_2),
        lambda : (image_1, image_2)
        )
    
    return out[0], out[1]

def _brightness(image_1, image_2):
    delta = tf.random.uniform(shape=(), dtype=tf.float32)
    image_1 = tf.image.adjust_brightness(image_1, delta)
    image_2 = tf.image.adjust_brightness(image_2, delta)
    return image_1, image_2
