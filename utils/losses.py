import numpy as np
import tensorflow as tf

import config as cfg

def gram_matrix(feature_set):
    """
    Given a set of vectors, in the form of a tensor, from a layer,
    compute the Gram matrix (https://en.wikipedia.org/wiki/Gramian_matrix).

    Args:
        feature_set: Tensor of vectors
            ([1, filter_height, filter_width, num_feature_maps])
    Returns:
        gram_matrix: Computed Gram matrix ([num_feature_maps, num_feature_maps])
    """
    feature_set_shape = tf.shape(feature_set)
    feature_set = tf.reshape(
        feature_set,
        [
            feature_set_shape[0], 
            feature_set_shape[1] * feature_set_shape[2], 
            feature_set_shape[3]
        ],
        name='vectorize_map')
    gram_matrix = tf.matmul(
        feature_set, feature_set, transpose_a=True, name='gram_map')

    return gram_matrix

def style_layer_loss(gram_matrix_desired, gram_matrix_predicted, filter_size):
    """
    Compute the loss between the gram matrix of the styling image and the
    gram matrix of the image undergoing optimization.

    Args:
        gram_matrix_desired  : Gram matrix of the styling image
        gram_matrix_predicted: Gram matrix of the image undergoing optimization.
        filter_size          : The size of an individual filter map 
            (filter_height * filter_width)
    Returns:
        loss_contribution: The loss contribution from this layer
    """

    _, num_filters, _ = gram_matrix_desired.get_shape().as_list()
    num_filters = float(num_filters)
    summed_squared_difference = tf.reduce_sum(
        tf.square(
            gram_matrix_predicted - gram_matrix_desired), 
        name='summed_squared_diff')
    loss_contribution = (1 / (4 * np.power(num_filters, 2) 
        * np.power(filter_size, 2))) \
        * summed_squared_difference

    return loss_contribution

def total_variation_loss(images):
    """
    Calculate and return the total variation loss for a batch of images. 

    Args: 
        images: Batch of images ([batch_size, height, width, channels])
    Returns: 
        tv_loss: Total variation loss for the batch of images.
    """
    tv_loss = tf.reduce_sum(
        tf.image.total_variation(images, name='total_variation'))
    return tv_loss