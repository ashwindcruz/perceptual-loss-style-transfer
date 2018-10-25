
"""
Training script for model described in:
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
import model
import read_images
from utils.imaging import display_image, format_image, save_image
from utils.losses import gram_matrix, style_layer_loss
import vgg

# Set TF debugging to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: Move this setup into utilities, maybe
# Directories setup

if cfg.RESET_SAVES is True:
    # Ensure tensorboard is not running when you try to delete
    # this directory
    if os.path.exists(cfg.TENSORBOARD_DIR):
        shutil.rmtree(cfg.TENSORBOARD_DIR)

# Create the debug directory if it doesn't exist
# Tensorboard directory is made automatically if it doesn't exist
if os.path.exists(cfg.DEBUG_DIR):
    shutil.rmtree(cfg.DEBUG_DIR)
os.makedirs(cfg.DEBUG_DIR)

# Set the seeds to provide consistency between runs
# Can also comment out for variability between runs
np.random.seed(cfg.NP_SEED)
tf.set_random_seed(cfg.TF_SEED)


# The input node to the image transform network
itn_inputs = tf.placeholder(
    tf.float32, shape=(None, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS),
    name='itn_inputs')

# Obtain output from image transform network
itn_outputs = model.image_transform_net(itn_inputs, is_training=True)
image_summary = tf.summary.image('val_images', itn_outputs, max_outputs=None)

# The input node to the vgg 16 network
vgg_16_inputs = tf.placeholder(
    tf.float32, shape=(cfg.BATCH_SIZE, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS),
    name='vgg_16_inputs')

# Obtain endpoints from the vgg 16 network
with slim.arg_scope(vgg.vgg_arg_scope()):
    _, end_points_inference = vgg.vgg_16(
        vgg_16_inputs, num_classes=None, is_training=False)

 # This is connected directly to the output of the itn
with slim.arg_scope(vgg.vgg_arg_scope()):
    _, end_points_training = vgg.vgg_16(
        itn_outputs, num_classes=None, is_training=False)

vgg_16_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg_16')
vgg_16_saver = tf.train.Saver(var_list=vgg_16_vars)

############################
### Style Representation ###
############################


# Style target is the same for the entire training regime
# So collect the response and place into constants

# Construct the style image tensor
# And the graph operation which assigns it to input_var
style_image = plt.imread(cfg.STYLE_IMAGE_PATH)
style_image = cv2.resize(style_image, (cfg.HEIGHT, cfg.WIDTH))
style_image_batch = np.tile(style_image, cfg.BATCH_SIZE)
style_image_batch = np.reshape(
    style_image_batch, (cfg.BATCH_SIZE, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
style_image_batch = np.asarray(style_image_batch, dtype=np.float32)


# Set up gram matrix nodes
grams = []
filter_sizes = []
for i in range(cfg.CHOSEN_DEPTH):
    chosen_layer = end_points_inference[cfg.STYLE_LIST[i]]
    gram_features = gram_matrix(chosen_layer)
    grams.append(gram_features)

    # Determine the size of the filters used at each layer
    # This is needed to calculate the loss from that layer
    _, filter_height, filter_width, _ = chosen_layer.get_shape().as_list()
    filter_size = float(filter_height * filter_width)
    filter_sizes.append(filter_size)

init_op = tf.group(
    tf.global_variables_initializer(), tf.local_variables_initializer(),
    name='initialize_all')

config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    # Initialize new variables and then restore vgg_19 variables
    sess.run(init_op)
    vgg_16_saver.restore(sess, cfg.CHECKPOINT_PATH)

    real_image_grams = sess.run(
        grams,  feed_dict={vgg_16_inputs: style_image_batch})

# Create constants with the real image gram matrices
gram_constants = []
for i in range(cfg.CHOSEN_DEPTH):
    node_name = 'gram_constant_{}'.format(i)
    gram_constant = tf.constant(real_image_grams[i], name=node_name)
    gram_constants.append(gram_constant)


# Obtain gram matrix information for output of itn
grams = []
filter_sizes = []
for i in range(cfg.CHOSEN_DEPTH):
    chosen_layer = end_points_training[cfg.STYLE_LIST[i]]
    gram_features = gram_matrix(chosen_layer)
    grams.append(gram_features)

    # Determine the size of the filters used at each layer
    # This is needed to calculate the loss from that layer
    _, filter_height, filter_width, _ = chosen_layer.get_shape().as_list()
    filter_size = float(filter_height * filter_width)
    filter_sizes.append(filter_size)

# Calculate the style loss
layer_losses = []
for i in range(cfg.CHOSEN_DEPTH):
    layer_loss = style_layer_loss(gram_constants[i], grams[i], filter_sizes[i])
    # Equal weighting on each loss, summing to 1
    layer_loss *= (1.0 / cfg.CHOSEN_DEPTH)
    layer_losses.append(layer_loss)
style_loss = tf.add_n(layer_losses, name='sum_layer_losses')


##############################
### Content Representation ###
##############################

content_image_batch = read_images.sample_batch()

content_rep_real = end_points_inference[cfg.CONTENT_LAYER]
content_rep_itn = end_points_training[cfg.CONTENT_LAYER]

content_loss = tf.losses.mean_squared_error(
    labels=content_rep_real, predictions=content_rep_itn)


# Set up the final loss, optimizer, and summaries
loss = (cfg.CONTENT_WEIGHT * content_loss) + (cfg.STYLE_WEIGHT * style_loss)
optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)
train_op = optimizer.minimize(
    loss,
    var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, 'image_transform_net'))

init_op = tf.group(
    tf.global_variables_initializer(), tf.local_variables_initializer(),
    name='initialize_all')

# Tensorboard summaries
loss_summary = tf.summary.scalar('loss', loss)

# Training
config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    # Initialize all variables and then
    # restore weights for feature extractor
    sess.run(init_op)
    vgg_16_saver.restore(sess, cfg.CHECKPOINT_PATH)

    # Set up summary writer for tensorboard, saving graph as well
    train_writer = tf.summary.FileWriter(cfg.TENSORBOARD_DIR, sess.graph)

    # Begin training
    for i in range(cfg.TRAINING_STEPS):

        # TODO: Should the 2 input placeholders be collapsed into 1?
        content_image_batch = read_images.sample_batch()
        loss_summary_, _ = sess.run([loss_summary, train_op],
                feed_dict={vgg_16_inputs:content_image_batch,
                    itn_inputs:content_image_batch})

        train_writer.add_summary(loss_summary_, i)

        if i % cfg.VALIDATION_STEPS == 0:
            val_image_batch = read_images.val_images()
            loss_, image_summary_ = sess.run([loss, image_summary],
                feed_dict={vgg_16_inputs:val_image_batch,
                    itn_inputs:val_image_batch})
            train_writer.add_summary(image_summary_, i)


            print(loss_)


