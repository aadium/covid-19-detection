import cv2
import numpy as np
import tensorflow as tf
import os

img_size = 64
test_dir = 'dataset\\val'

with open('classes.txt', 'r', encoding='utf-8') as f:
    labels = f.read().splitlines()


def get_image_paths(directory):
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')  # Add or remove extensions as needed
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Use the function to get all image paths
images_to_be_classified = get_image_paths(test_dir)

resized_images = []

for image_path in images_to_be_classified:
    image = cv2.imread(image_path)  # Load the image using OpenCV

    if image is not None:  # Check if the image was loaded successfully
        resized_image = cv2.resize(image, (img_size, img_size))  # Resize the image
        resized_images.append(resized_image)  # Append the resized image to the list
    else:
        print(f"Skipping invalid image: {image_path}")

# Continue with the rest of your script...

resized_images = np.array(resized_images)  # Convert the list of images to a NumPy array
resized_images = resized_images / 255.0

model = tf.keras.models.load_model('covid19_model.h5')
predictions = model.predict(resized_images)

for i in range(len(images_to_be_classified)):
    image_name = images_to_be_classified[i]
    prediction = predictions[i]
    rounded_prediction = np.round(prediction)
    label = labels[np.argmax(rounded_prediction)]
    print("Image:", image_name, ", Rounded Prediction:", rounded_prediction, ", Label:", label)
