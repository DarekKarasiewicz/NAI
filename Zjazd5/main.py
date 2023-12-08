import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
import seaborn as sns

"""
This program teach neural network to predict values from cifar 10 and Minst.

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

def seeds_dataset_prediction():
    """
    Performs binary classification on the seeds_dataset using a neural network.

    Loads the seeds_dataset, scales the features, splits the data into training and testing sets,
    builds and trains a neural network model, and prints the accuracy on the test set.

    Returns:
    None

    """
    data = np.loadtxt('./seeds_dataset.txt')

    X = data[:, :-1]
    y = data[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential()
    model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Accuracy: {accuracy}")

    

def load_mnist(path, kind='train'):
    """
    Loads the MNIST dataset from specified path.

    Parameters:
    - path (str): Path to the directory containing the MNIST dataset files.
    - kind (str, optional): Specifies whether to load the training or testing set ('train' or 'test').

    Returns:
    tuple: A tuple containing two elements - numpy arrays representing images and corresponding labels.

    """

    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels

def train_model_MINST_way(train_images, train_labels):
    """
    Trains a Feedforward Neural Network (FNN) model on the MNIST dataset.

    Parameters:
    - train_images (numpy array): Array containing training images (shape: [num_samples, width, height]).
    - train_labels (numpy array): Array containing corresponding training labels (shape: [num_samples]).

    Returns:
    tf.keras.models.Sequential: Trained FNN model.

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    return model

def train_model_CIFAR10(train_images, train_labels):
    """
    Trains a Convolutional Neural Network (CNN) model on CIFAR-10 dataset.

    Parameters:
    - train_images (numpy array): Array containing training images (shape: [num_samples, width, height, channels]).
    - train_labels (numpy array): Array containing corresponding training labels (shape: [num_samples]).

    Returns:
    tf.keras.models.Sequential: Trained CNN model.

    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  
        tf.keras.layers.Dense(10),
    ])
    # Kompiluj model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=20)
    return model
  

def plot_image(i, predictions_array, true_label, img, train_labels=None):
    
    """
    Plots an image along with its predicted and true labels.

    Parameters:
    - i (int): Index of the image to be plotted.
    - predictions_array (numpy array): Array containing predicted label probabilities for each class.
    - true_label (int): True label of the image.
    - img (numpy array): Image data represented as a NumPy array.
    - train_labels (list or None, optional): List of label names for the training dataset. 
      If provided, it will be used to display label names instead of numerical labels.

    Returns:
    None
    """

    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    if train_labels == None:
      plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                            100 * np.max(predictions_array),
                                            true_label),
                color=color)
    else:
      plt.xlabel("{} {:2.0f}% ({})".format(train_labels[predicted_label],
                                    100*np.max(predictions_array),
                                    train_labels[true_label]),
                                    color=color)


def plot_value_array(i, predictions_array, true_label):
    """
    Plots the predicted label probabilities as a bar chart with highlighting for the predicted and true labels.

    Parameters:
    - i (int): Index of the image for which the values are plotted.
    - predictions_array (numpy array): Array containing predicted label probabilities for each class.
    - true_label (int): True label of the image.

    Returns:
    None

    """
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def predict_visulisation(predictions, test_labels, test_images):

  """
  Visualizes a set of predicted labels and their corresponding true labels and images.

  Parameters:
  - predictions (numpy array): Array containing predicted label probabilities for each test image.
  - test_labels (numpy array): Array containing true labels for the test images.
  - test_images (numpy array): Array containing test images.

  Returns:
  None
  """
  num_rows = 5
  num_cols = 3
  num_images = num_rows * num_cols
  plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
  for i in range(num_images):
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
      plot_image(i, predictions[i], test_labels, test_images)
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
      plot_value_array(i, predictions[i], test_labels)
  plt.tight_layout()
  plt.show()

def unpickle(file):
    """
    Unpickles and loads data from a binary file.

    Parameters:
    - file (str): Path to the binary file to be unpickled.

    Returns:
    object: Unpickled data.
    """
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def animals():
  """
  Loads CIFAR-10 dataset, trains a model, evaluates it on the test set, and visualizes predictions.

  Loads training data from multiple batches, concatenates them, trains a CNN model using train_model_CIFAR10 function,
  loads the test data, evaluates the model on the test set, and visualizes predictions using predict_visualization function.

  Returns:
  None
  """
  train_images_list = []
  train_labels_list = []

  for i in range(1, 6):
      filename = f"./cifar-10-batches-py/data_batch_{i}"
      data_batch = unpickle(filename)
      train_images_list.append(data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0)
      train_labels_list.append(np.array(data_batch[b'labels']))

  train_images = np.concatenate(train_images_list, axis=0)
  train_labels = np.concatenate(train_labels_list, axis=0)

  model = train_model_CIFAR10(train_images, train_labels)

  test_data = unpickle("./cifar-10-batches-py/test_batch")
  test_images = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
  test_labels = np.array(test_data[b'labels'])

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc}')

  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

  predictions = probability_model.predict(test_images)

  predict_visulisation(predictions, test_labels, test_images)



def fashion():
  """
  Loads the Fashion MNIST dataset, trains a model, evaluates it on the test set, 
  and visualizes predictions including a confusion matrix.

  Loads Fashion MNIST training and test data, preprocesses the data, trains a model 
  using train_model_MINST_way function, evaluates the model on the test set, 
  visualizes a confusion matrix, and displays predictions using predict_visualization function.
  
  Returns:
  None
  """
  path = './MINST-fashion'

  train_images, train_labels = load_mnist(path, kind='train')

  test_images, test_labels = load_mnist(path, kind='t10k')

  train_images = train_images / 255.0

  test_images = test_images / 255.0

  model = train_model_MINST_way(train_images, train_labels)
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

  print('\nTest accuracy:', test_acc)
  probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_images)

  predictions_single = np.argmax(predictions, axis=1)

  cm = confusion_matrix(test_labels, predictions_single)

  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
  plt.xlabel('Predicted')
  plt.ylabel('True value')
  plt.title('Confusion Matrix')
  plt.show()


  predict_visulisation(predictions, test_labels, test_images)


def MINST():
  """
  Loads the MNIST dataset, trains a model, evaluates it on the test set, 
  and visualizes predictions.

  Loads MNIST training and test data, preprocesses the data, trains a model 
  using train_model_MINST_way function, evaluates the model on the test set, 
  and visualizes predictions using predict_visualization function.

  Returns:
  None
  """
  path = './MINST'

  train_images, train_labels = load_mnist(path, kind='train')

  test_images, test_labels = load_mnist(path, kind='t10k')

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  model = train_model_MINST_way(train_images, train_labels)
  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)

  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_images)

  predict_visulisation(predictions, test_labels, test_images)

animals()
fashion()
MINST()
seeds_dataset_prediction()