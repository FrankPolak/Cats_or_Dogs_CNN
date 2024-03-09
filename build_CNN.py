import tensorflow as tf
# tf.__version__ = '2.12.0'
from keras.preprocessing.image import ImageDataGenerator

# Image preprocessing
# Apply transformations to the images of the training set 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Test set preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Initialise the CNN
cnn = tf.keras.models.Sequential()

### Build the CNN (architecture) ###

# 1. Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[64,64,3]
))

# 2. Pooling
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
))

# 3. Second layer (convolution and pooling)
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'
))

cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
))

# 4. Flattening
cnn.add(tf.keras.layers.Flatten())

# 5. Full connection
cnn.add(tf.keras.layers.Dense(
    units=128,
    activation='relu'
))

# 6. Output layer
cnn.add(tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'
))

### Train the CNN ###

# 1. Compile
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 2. Train
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)

# Save CNN in HDF5 format
cnn.save("cats_or_dogs.h5")


