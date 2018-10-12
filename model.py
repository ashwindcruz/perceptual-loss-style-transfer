"""
This script implements different network blocks described in the following work: 

Johnson, J., Alahi, A. and Fei-Fei, L., 2016, October. 
Perceptual losses for real-time style transfer and super-resolution. 
In European Conference on Computer Vision (pp. 694-711). Springer, Cham.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config as cfg
from utils.imaging import display_image, format_image, save_image
from utils.losses import gram_matrix, style_layer_loss
import vgg


def conv_with_batch_norm(
	inputs, filters, kernel_size, strides, is_training, name):
	"""
	Convolution followed by batch norm and ReLU activation. 

	Args:
		inputs: Tensor input. 
		filters: Integer, the dimensionality of the output space 
			(i.e. the number of filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the 
			height and width of the 2D convolution window. 
			Can be a single integer to specify the same value for all 
			spatial dimensions.
		strides: An integer or tuple/list of 2 integers, specifying the strides 
			of the convolution along the height and width. 
			Can be a single integer to specify the same value for all s
			patial dimensions. 
		is_training: Whether or not the model is being trained.
		name: A string, used for creating names for the operations used here.
	Returns:
		net: Tensor output. 
	"""
	with tf.variable_scope('conv_with_batch_norm_' + name):
		net = tf.layers.conv2d(
			inputs, filters, kernel_size, strides, padding='same', 
			name='conv')
		net = tf.layers.batch_normalization(
			net, training=is_training, name='bn')
		net = tf.nn.relu(net, name='relu')

	return net

def conv_transposed_with_batch_norm(
	inputs, filters, kernel_size, strides, is_training, name):
	"""
	Transposed convolution followed by batch norm and ReLU activation. 

	Args:
		inputs: Tensor input. 
		filters: Integer, the dimensionality of the output space 
			(i.e. the number of filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the 
			height and width of the 2D convolution window. 
			Can be a single integer to specify the same value for all 
			spatial dimensions.
		strides: An integer or tuple/list of 2 integers, specifying the strides 
			of the convolution along the height and width. 
			Can be a single integer to specify the same value for all s
			patial dimensions. 
		is_training: Whether or not the model is being trained.
		name: A string, used for creating names for the operations used here.
	Returns:
		net: Tensor output. 
	"""
	with tf.variable_scope('conv_transpose_with_batch_norm_' + name):
		net = tf.layers.conv2d_transpose(
			inputs, filters, kernel_size, strides, padding='same', 
			name='conv')
		net = tf.layers.batch_normalization(
			net, training=is_training, name='bn')
		net = tf.nn.relu(net, name='relu')

	return net

def residual_block(inputs, is_training, name):
	"""
	Edited residual block as shown in Figure 1 of this link: 
	https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf 

	Args:
		inputs: Tensor input. 
		is_training: Whether or not the model is being trained.
		name: A string, used for creating names for the operations used here.
	Returns:
		net: Tensor output. 
	"""
	with tf.variable_scope('residual_block_'+name): 
		net = tf.layers.conv2d(
			inputs, filters=128, kernel_size=3, strides=1, name='conv_1')
		net = tf.layers.batch_normalization(
			net, training=is_training, name='bn_1')
		net = tf.nn.relu(net, name='relu')
		net = tf.layers.conv2d(
			net, filters=128, kernel_size=3, strides=1, name='conv_2')
		net = tf.layers.batch_normalization(
			net, training=is_training, name='bn_2')

		# The conv operations don't have padding so the volume shrinks in 
		# width and height. 
		# So to add the residual connection, the input must be cropped.
		cropped_inputs = inputs[:, 2:-2, 2:-2, :]

		net = net + cropped_inputs

	return net

def image_transform_net(inputs, is_training):
	"""
	Implements the image transform net as noted in Figure 2 of the original 
	paper. 
	Details of the architecture were obtained from the following link: 
	https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf

	Args:
		inputs: Input image(s) to style. Tensor must have dimensions: 
			[batch_size, 256, 256, 3]
		is_training: Whether or not the model is being trained.
	"""
	with tf.variable_scope('image_transform_net'):
		# Initial reflection padding to ensure the output is the same size
		# as the input
		padding = tf.constant(
			[[0, 0], [40, 40,], [40, 40], [0, 0]], name='reflection_padding')
		net = tf.pad(inputs, padding, "REFLECT")

		# First round of convolutions
		net = conv_with_batch_norm(
			net, 32, 9, 1, is_training, '1')
		net = conv_with_batch_norm(
			net, 64, 3, 2, is_training, '2')
		net = conv_with_batch_norm(
			net, 128, 3, 2, is_training, '3')

		# Residual connections
		net = residual_block(net, is_training, '1')
		net = residual_block(net, is_training, '2')
		net = residual_block(net, is_training, '3')
		net = residual_block(net, is_training, '4')
		net = residual_block(net, is_training, '5')
		
		# Transposed convolutions
		net = conv_transposed_with_batch_norm(
			net, filters=64, kernel_size=3, strides=2, 
			is_training=is_training, name='1')
		net = conv_transposed_with_batch_norm(
			net, filters=32, kernel_size=3, strides=2, 
			is_training=is_training, name='2')
				
		# Convolution
		net = conv_with_batch_norm(
			net, 3, 9, 1, is_training, '4')

		return net
