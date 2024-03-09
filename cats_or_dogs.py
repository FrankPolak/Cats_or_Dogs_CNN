import sys
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.utils import load_img
import numpy as np

# Pass and check arguments
def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png']
    _, extension = os.path.splitext(filename)
    return extension.lower() in image_extensions

if len(sys.argv) == 2:
    test_image = sys.argv[1]
elif len(sys.argv) == 1:
    print("No arguments passed. Please provide an image.")
    sys.exit(1)
else:
    print("Too many arguments passed. Please provide only one image.")
    sys.exit(1)

if not is_image_file(test_image):
    print("Provided file is not an image. Please provide a .jpeg, .jpg, or .png file.")
    sys.exit(1)

if not os.path.exists(test_image):
    print("File not found.")
    sys.exit(1)

# Load model
cnn = keras.models.load_model("cats_or_dogs.h5")

# Modify test image
test_image = load_img(
    test_image,
    target_size=(64, 64)
)

test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction
result = cnn.predict(test_image)
print("###### Results ######")
if result[0][0] == 1:
    print("This is an image of a dog.")
else:
    print("This is an image of a cat.")