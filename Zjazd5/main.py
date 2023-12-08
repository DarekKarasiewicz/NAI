import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels

def train_model_MINST_way(train_images, train_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Kompiluj model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Trenuj model
    model.fit(train_images, train_labels, epochs=10)
    return model

def train_model_CIFAR10(train_images, train_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dodanie warstwy Dropout
        tf.keras.layers.Dense(10),
    ])
    # Kompiluj model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=20)
    return model
  

def plot_image(i, predictions_array, true_label, img, train_labels=None):
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
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def animals():
  # Wczytaj dane treningowe z plików data_batch_1 do data_batch_5
  train_images_list = []
  train_labels_list = []

  for i in range(1, 6):
      filename = f"./cifar-10-batches-py/data_batch_{i}"
      data_batch = unpickle(filename)
      train_images_list.append(data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0)
      train_labels_list.append(np.array(data_batch[b'labels']))

  # Połącz dane treningowe z różnych plików
  train_images = np.concatenate(train_images_list, axis=0)
  train_labels = np.concatenate(train_labels_list, axis=0)

  model = train_model_CIFAR10(train_images, train_labels)

  # Wczytaj dane testowe
  test_data = unpickle("./cifar-10-batches-py/test_batch")
  test_images = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
  test_labels = np.array(test_data[b'labels'])

  # Ocena na danych testowych
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc}')

  # Przekształć model na model prawdopodobieństwa
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

  # Sprawdź predykcje dla pierwszego obrazu ze zbioru testowego
  predictions = probability_model.predict(test_images)

  predict_visulisation(predictions, test_labels, test_images)



def fashion():
  fashion_mnist = tf.keras.datasets.fashion_mnist

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


  train_images = train_images / 255.0

  test_images = test_images / 255.0

  model = train_model_MINST_way(train_images, train_labels)
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

  print('\nTest accuracy:', test_acc)
  probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_images)

  predict_visulisation(predictions, test_labels, test_images)


def MINST():

  # Podaj ścieżkę do folderu zawierającego pliki danych
  path = './MINST'

  # Wczytaj dane treningowe
  train_images, train_labels = load_mnist(path, kind='train')

  # Wczytaj dane testowe
  test_images, test_labels = load_mnist(path, kind='t10k')

  # Reskaluj obrazy do zakresu [0, 1]
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  model = train_model_MINST_way(train_images, train_labels)
  # Ocena modelu na zbiorze testowym
  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  print('\nTest accuracy:', test_acc)

  # Utwórz model prawdopodobieństwa dla predykcji
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  predictions = probability_model.predict(test_images)

  # Wizualizacja predykcji
  predict_visulisation(predictions, test_labels, test_images)

animals()
fashion()
MINST()