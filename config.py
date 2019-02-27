# Path to images uused during training and validation
STYLE_IMAGE_PATH = './images/starry_night.jpg'
TRAIN_IMAGES_DIR = './train2014/train2014/'
VAL_IMAGES_DIR = './val2014/val2014/'

# Seed for initializing numpy and tf
NP_SEED = 0
TF_SEED = 0

# Path to vgg19 checkpoint, must be downloaded separately
CHECKPOINT_PATH = './vgg_19.ckpt'

# Location of tensorboard summaries
TENSORBOARD_DIR = './tensorboard/'

# Path to directory used for storing images
DEBUG_DIR = './debug/'

# Dimensions desired for input, channels must be kept as 3
BATCH_SIZE = 4
HEIGHT = 256
WIDTH = 256
CHANNELS = 3

### These are when using vgg_16
# Layer being used for content component
# CONTENT_LAYER = 'vgg_16/conv4/conv2_2'

# # Layers that can be used for style component
# STYLE_LIST = [
# 'vgg_16/conv1/conv1_2',
# 'vgg_16/conv2/conv2_2',
# 'vgg_16/conv3/conv3_3',
# 'vgg_16/conv4/conv4_3'
# ]

### These are when using vgg_19
# Layer being used for content component
CONTENT_LAYER = 'vgg_19/conv4/conv4_2'

# Layers that can be used for style component
STYLE_LIST = [
'vgg_19/conv1/conv1_1',
'vgg_19/conv2/conv2_1', 
'vgg_19/conv3/conv3_1',
'vgg_19/conv4/conv4_1', 
'vgg_19/conv5/conv5_1'
]

# Chosen depth corresponds to how many feature layers you want to use
# for the style component
CHOSEN_DEPTH = 2

# Weights for each loss component
# CONTENT_WEIGHT = 7.5
# STYLE_WEIGHT = 100 
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = CONTENT_WEIGHT/8e-4
TV_WEIGHT = 2e-2 

# Learning rate for optimizer
LEARNING_RATE = 1e-1

# Number of training and validation step
# In this instance, validation refers to when we would like to examine:
# save currently optimized image and loss
TRAINING_EPOCHS = 20 
VALIDATION_STEPS = 100

# Offline debugging refers to images that will be saved to folder using plt,
# every validation step
DEBUG_OFFLINE = True

# This is how often training information will be printed to screen
DISPLAY_STEPS = 50

# Determines whether information is saved between runs
# for tensorboard
RESET_SAVES = True

### OVERFITTING MODE
# In this mode, we train on a smaller batch of data, treat that set as the 
# validation data, since it's unlikely the network will generalize, and 
# we also view information more frequently
OVERFITTING_MODE = True
if OVERFITTING_MODE:
	TRAINING_EPOCHS = 500
	VALIDATION_STEPS = 50
	DISPLAY_STEPS = 1
