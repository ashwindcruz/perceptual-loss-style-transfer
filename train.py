
"""
Training script for model described in:
Johnson, J., Alahi, A. and Fei-Fei, L., 2016, October.
Perceptual losses for real-time style transfer and super-resolution.
In European Conference on Computer Vision (pp. 694-711). 
Springer, Cham.
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
from utils.losses import gram_matrix, style_layer_loss, total_variation_loss
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


# Input image placeholder
# Same placeholder is used in two locations: 
# 1. Feeding into the image transform network
# 2. Feeding into the vgg 16 network
image_inputs = tf.placeholder(
    tf.float32, shape=(None, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS),
    name='itn_inputs')

# Obtain output from image transform network
itn_outputs = model.image_transform_net(
    image_inputs, is_training=True)
image_summary = tf.summary.image(
    'val_images', itn_outputs, max_outputs=cfg.BATCH_SIZE)

# Obtain endpoints from the vgg 16 network on the style and content 
# images we are trying to match 
with slim.arg_scope(vgg.vgg_arg_scope()):
    _, end_points_fixed = vgg.vgg_16(
        image_inputs, num_classes=None, is_training=False)

 # Obtain endpoints from the vgg 16 network on the output of the 
 # image transform network 
with slim.arg_scope(vgg.vgg_arg_scope()):
    _, end_points_itn = vgg.vgg_16(
        itn_outputs, num_classes=None, is_training=False)

vgg_16_vars = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, 'vgg_16')
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
    style_image_batch, 
    (cfg.BATCH_SIZE, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
style_image_batch = np.asarray(style_image_batch, dtype=np.float32)


# Set up gram matrix nodes
grams = []
filter_sizes = []
for i in range(cfg.CHOSEN_DEPTH):
    chosen_layer = end_points_fixed[cfg.STYLE_LIST[i]]
    gram_features = gram_matrix(chosen_layer)
    grams.append(gram_features)

    # Determine the size of the filters used at each layer
    # This is needed to calculate the loss from that layer
    _, filter_height, filter_width, _ = \
        chosen_layer.get_shape().as_list()
    filter_size = float(filter_height * filter_width)
    filter_sizes.append(filter_size)

init_op = tf.group(
    tf.global_variables_initializer(), 
    tf.local_variables_initializer(),
    name='initialize_all')

with tf.Session() as sess:
    # Initialize new variables and then restore vgg_19 variables
    sess.run(init_op)
    vgg_16_saver.restore(sess, cfg.CHECKPOINT_PATH)

    real_image_grams = sess.run(
        grams,  feed_dict={image_inputs: style_image_batch})

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
    chosen_layer = end_points_itn[cfg.STYLE_LIST[i]]
    gram_features = gram_matrix(chosen_layer)
    grams.append(gram_features)

    # Determine the size of the filters used at each layer
    # This is needed to calculate the loss from that layer
    _, filter_height, filter_width, _ = \
        chosen_layer.get_shape().as_list()
    filter_size = float(filter_height * filter_width)
    filter_sizes.append(filter_size)

# Calculate the style loss
layer_losses = []
for i in range(cfg.CHOSEN_DEPTH):
    layer_loss = style_layer_loss(
        gram_constants[i], grams[i], filter_sizes[i])
    # Equal weighting on each loss, summing to 1
    layer_loss *= (1.0 / cfg.CHOSEN_DEPTH)
    layer_losses.append(layer_loss)
style_loss = tf.add_n(layer_losses, name='sum_layer_losses')


##############################
### Content Representation ###
##############################

content_rep_real = end_points_fixed[cfg.CONTENT_LAYER]
content_rep_itn = end_points_itn[cfg.CONTENT_LAYER]

content_loss = tf.losses.mean_squared_error(
    labels=content_rep_real, predictions=content_rep_itn)

# Calculate total variation loss
tv_loss = total_variation_loss(itn_outputs)

# Set up the final loss, optimizer, and summaries
loss = (cfg.CONTENT_WEIGHT * content_loss) \
    + (cfg.STYLE_WEIGHT * style_loss) \
    + (cfg.TV_WEIGHT * tv_loss)
optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)
train_op = optimizer.minimize(
    loss,
    var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, 'image_transform_net'))

init_op = tf.group(
    tf.global_variables_initializer(), 
    tf.local_variables_initializer(),
    name='initialize_all')

# Tensorboard summaries
train_loss_summary = tf.summary.scalar('train_loss', loss)

val_loss_ph = tf.placeholder(
    dtype=tf.float64, shape=(), name='val_loss_placeholder')
val_loss = tf.identity(val_loss_ph, name='val_loss_tensor')
val_loss_summary = tf.summary.scalar('val_loss', val_loss)

# Training
with tf.Session() as sess:
    # Initialize all variables and then
    # restore weights for feature extractor
    sess.run(init_op)
    vgg_16_saver.restore(sess, cfg.CHECKPOINT_PATH)

    # Set up summary writer for tensorboard, saving graph as well
    train_writer = tf.summary.FileWriter(
        cfg.TENSORBOARD_DIR, sess.graph)

    # Begin training
    num_train_batches = read_images.NUM_TRAIN_IMAGES // cfg.BATCH_SIZE
    num_val_batches = read_images.NUM_VAL_IMAGES // cfg.BATCH_SIZE

    training_step = 0
    for i in range(cfg.TRAINING_EPOCHS):
        # Go through one pass of the training data
        for j in range(num_train_batches):

            start_index = j * cfg.BATCH_SIZE
            content_image_batch = read_images.fetch_batch(
                start_index, 'train')
            train_loss_summary_, _ = sess.run(
                    [train_loss_summary, train_op],
                    feed_dict={image_inputs:content_image_batch})

            train_writer.add_summary(
                train_loss_summary_, training_step)
            training_step += 1
            
            # Check how the validation images are looking
            if training_step % cfg.VALIDATION_STEPS == 0:
                content_image_batch = read_images.fetch_batch(
                    read_images.CHOSEN_IDX, 'val')
                image_summary_ = sess.run(
                    image_summary,
                    feed_dict={image_inputs:content_image_batch}
                    )
                train_writer.add_summary(
                    image_summary_, training_step)
                
        print('Epoch {} done'.format(i+1))
        
        # Go through one pass of the validation data
        val_loss_accum = 0
        for j in range(num_val_batches):

            start_index = j * cfg.BATCH_SIZE
            content_image_batch = read_images.fetch_batch(
                start_index, 'val')
            val_loss_accum  += sess.run(
                loss,
                feed_dict={image_inputs:content_image_batch})
        val_loss_summary_ = sess.run(
            val_loss_summary, 
            feed_dict={val_loss_ph:val_loss_accum})
        train_writer.add_summary(val_loss_summary_, training_step)
        