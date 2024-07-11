import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

train_dir = 'dataset\\train'
test_dir = 'dataset\\test'

img_size = 64
num_classes = len(os.listdir(train_dir))
batch_size = 32


def load_dataset():
    loaded_images = []
    loaded_lbls = []
    classes = os.listdir(train_dir)
    label_binarizer = LabelBinarizer()
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):  # Check if the current item is a directory
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:  # Check if the image was loaded successfully
                        image = cv2.resize(image, (img_size, img_size))
                        loaded_images.append(image)
                        loaded_lbls.append(class_name)
                    else:
                        print(f"Skipping invalid image: {image_path}")
                except Exception as e:
                    print(f"Error loading image: {image_path}")
                    print(str(e))
    # Convert labels to unique binary arrays
    loaded_labels = label_binarizer.fit_transform(loaded_lbls)
    return loaded_images, loaded_lbls, loaded_labels


images, lbls, labels = load_dataset()

# Create a set to keep track of unique labels
printed_labels = set()
# Create a counter to keep track fo the label count
no_of_labels = 0
# Print lbls and corresponding labels
for labelName, arr in zip(lbls, labels):
    # Check if the label has already been printed
    if labelName not in printed_labels:
        # Print the disease and array
        print("Label:", labelName, ", Array:", arr)
        # Add the disease to the set of printed diseases
        printed_labels.add(labelName)
        # Increment the disease counter
        no_of_labels = no_of_labels + 1

print()
print("No. of labels:", no_of_labels)

def preprocess_dataset(images, labels):
    images = np.array(images)
    images = images.astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels

images, labels = preprocess_dataset(images, labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
data_generator.fit(train_images)

history = model.fit(data_generator.flow(train_images, train_labels, batch_size=batch_size), steps_per_epoch=len(train_images) // batch_size, epochs=50)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save('covid19_model.h5')