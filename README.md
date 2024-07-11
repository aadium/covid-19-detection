# COVID-19 detection
 
This is a Python program for training a Convolutional Neural Network (CNN) model to detect COVID-19 from chest X-ray images, and serving it on an API endpoint. Here is a detailed description of the files involved:

### web-service
<li>This is a web server built using Flask that serves the image classification model over a web service.
<li>You just have to specify the image url in the request body, and it will return the top 2 predicted classes along with their probabilities.

### train.py
<li>The necessary libraries are imported, including numpy, cv2, os, matplotlib.pyplot, random, sklearn.model_selection, keras.models, keras.layers, and keras.preprocessing.image.
<li>The dataset can be downloaded from here: https://www.kaggle.com/datasets/imdevskp/corona-virus-report
<li>The paths for the dataset and test images are set.
<li>The parameters such as image size, number of classes, and batch size are defined.
<li>The "load_dataset" function is defined to load the images and labels from the dataset directory. It iterates through the class directories, reads the images using OpenCV, resizes them, and appends them to the "images" list. The labels are also collected and converted to unique binary arrays using label binarization.
<li>The images and labels are preprocessed using the "preprocess_dataset" function. The images are converted to NumPy array, normalized to the range [0, 1], and labels are converted to NumPy array.
<li>The dataset is split into training and testing sets using the "train_test_split" function from sklearn.model_selection.
<li>A sequential CNN model is created using the Keras Sequential API. It consists of three convolutional layers with ReLU activation, followed by max-pooling layers, and a fully connected layer with ReLU activation. The final layer uses softmax activation for multi-class classification.
<li>The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
<li>Data augmentation is performed on the training images using the ImageDataGenerator from Keras. It applies random rotation, width and height shifts, and horizontal flip to generate augmented images.
<li>The model is trained on the augmented training data using the fit function. The training data is passed through the data generator, and the number of steps per epoch is calculated based on the batch size.
<li>The accuracy and loss history during training are stored in the "history" variable for plotting.
<li>The accuracy and loss graphs are plotted using matplotlib.
<li>The model is evaluated on the testing data using the evaluate function, and the test loss and accuracy are printed.
<li>The trained model is saved as an H5 file named "covid19_model.h5".

### test.py
<li>The image parameters are defined
<li>The images to be classified are loaded from the test directory, resized, and normalized.
<li>The model is loaded
<li>The model predicts the labels for the resized images using the predict function.
<li>The image names and their corresponding predicted labels are printed.
