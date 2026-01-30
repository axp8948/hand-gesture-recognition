# Big Picture

# Data(Images) --> DataLoader --> CNN --> Train --> Evaluate --> Save

import os 


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


# Defining Constants

IMG_SIZE = 128
BATCH_SIZE = 10
EPOCHS = 20
DATA_DIR = "../../data"

# Prepare the data 

# Scale the dataset so that we dont have large values

datagen = ImageDataGenerator(
    rescale = 1.0 / 255, 
    validation_split = 0.2, # 80% training and rest is for validation

    # data augmentation -> for model robustness
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
)

trainGenerator = datagen.flow_from_directory(
    DATA_DIR, # directory path
    target_size = (IMG_SIZE, IMG_SIZE), # size of the image
    batch_size = BATCH_SIZE,
    class_mode = "sparse", # since we are classifying into multiple classes
    subset = "training"
)

val_generator = datagen.flow_from_directory(
    DATA_DIR, 
    target_size = (IMG_SIZE, IMG_SIZE), 
    batch_size = BATCH_SIZE,
    class_mode = "sparse", 
    subset = "validation" # validation set
)

# BUILD THE CNN MODEL
# CNN is the area of deep learning that specializes in pattern recognition
# Filters are the fundamental of CNN # FILTERS / KERNELS / FEATURE DETECTORS
# Filters slide through the image matrix and match the image features 
# Filters start from small pattern recognitions to large patterns


# ARCHITECTURE OF A CONVOLUTIONAL LAYER

# CONV --> POOL --> CONV --> POOL --> FC --> FC

# Input -> matrix of pixels -> three 2d channels --> three 2d matrices stacked upon each other 
# CONV --> kernel a mini matrix -> dot product with the original matrix to produce a new matrix --> called feature mapping
# When kernel are initialized with certain values, they can be used to learn certain feature by transforming the original 
# matrix




model = models.Sequential() # sequential is a layer to layer model, where output from last layer goes as an input to the next layer

# FIRST CONVOLUTIONAL LAYER

# Conv2D -> our input is 3 layer of 2d matrices stacked together -> the kernel traverse all the matrix at the same time
# filters -> 32 different filters to extrat 32 different features
# relu -> for non-linearity
# shape of input -> image size + 3 color channels


# Take an RGB image of shape (IMG_SIZE × IMG_SIZE × 3), apply 32 learnable 3×3 convolutional filters across its spatial
# dimensions, combine information from all color channels, apply ReLU non-linearity, and output 32 feature maps.

model.add(
    layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        activation = "relu",
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
    )
)

# POOLING LAYER

# # After Conv2D, feature maps are:
# large (still close to 128×128)
# redundant (neighboring pixels often say similar things)

# Pooling helps us:
# reduce spatial size
# keep the strongest features
# make the model more robust to small shifts

model.add(
    layers.MaxPooling2D(
        pool_size = (2,2)
    )
) # 2 x 2 means that it looks at the 2 x 2 layer -> keep the max value 
# reducing it to 64 x 64 x 32


# model.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘



# SECOND CONV LAYER
model.add(
    layers.Conv2D(
        filters = 64, # more filters, better feature extraction
        kernel_size = (3, 3),
        activation="relu"
    )
)

# SECOND POOLING LAYER
model.add(
    layers.MaxPooling2D(
        pool_size = (2, 2)
    )
)

# model.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 61, 61, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 19,392 (75.75 KB)
#  Trainable params: 19,392 (75.75 KB)
#  Non-trainable params: 0 (0.00 B)


# THIRD CONV LAYER
model.add(
    layers.Conv2D(
        filters = 128,
        kernel_size = (3, 3),
        activation = "relu"
    )
)

# THIRD POOL LAYER
model.add(
    layers.MaxPooling2D(
        pool_size = (2, 2)
    )
)


# model.summary()

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 61, 61, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 28, 28, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 14, 14, 128)         │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 93,248 (364.25 KB)
#  Trainable params: 93,248 (364.25 KB)
#  Non-trainable params: 0 (0.00 B)

# WE MOVED FROM 128 X 128 X 3 TO 14 X 14 X 128 

# Fewer pixels → less where

# More channels → more what

# This now means:

# Rough location grid (14×14)

# At each location, 128 learned features

# So each spatial cell is no longer “a pixel” — it’s a semantic descriptor.



# NOW IT IS TIME TO MOVE FROM FEATURE EXTRACTION TO CLASSIFICATION


# FLATTEN THE FEATURE MAPS -> INTO ONE LONG VECTOR 
# Converts 14×14×128 into one long vector
# Prepares data for fully connected layers

model.add(
    layers.Flatten()
)


# ADD A FULLY CONNECTED DENSE LAYER

# 128 neurons -> design choice
# 14 × 14 × 128 = 25,088
# each of the 128 units has 25,088 inputs

model.add(
    layers.Dense(
        units = 128,
        activation = "relu"
    )
)


# DROPOUT 
# Randomly disables 50% of neurons during training to prevent memorization

model.add(
    layers.Dropout(
        rate = 0.5
    )
)

# OUTPUT LAYER (CLASSIFICATION)
# 3 units because 3 classes, and softmax for the prob distribution

model.add(
    layers.Dense(
        units = 3,
        activation = "softmax"
    )
)

# model.summary()


# FINAL MODEL:

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 61, 61, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 28, 28, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 14, 14, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 25088)               │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │       3,211,392 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 3)                   │             387 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 3,305,027 (12.61 MB)
#  Trainable params: 3,305,027 (12.61 MB)
#  Non-trainable params: 0 (0.00 B)





# MODEL COMPILATION

# loss function:
# sparse -> labels are integer, not one hot vector
# categorical -> multiclass

# optimizer -> Adam

# tracking metrics to compare the accuracy

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)



# TRAINING THE MODEL

history = model.fit(
    x = trainGenerator, # train data
    epochs = EPOCHS,
    validation_data = val_generator
)



# MODEL EVALUATION ON VALIDATION SET

val_loss, val_acc = model.evaluate(val_generator)
print("Validation Accuracy: ", val_acc)


# SAVE THE MODEL
model.save("hand_gesture_cnn.h5")

