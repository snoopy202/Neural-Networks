We work for CC: ConscientiousCars, where we help self-driving vehicles be more conscientious of their surroundings. Our cars have been very good at recognizing and avoiding humans. They haven't, however, been capable of recognizing dogs. Since dogs are man's best friend and will always be where we humans are, we want our cars to know if a dog is on the road in front of them and avoid the dog!

The first step to avoiding these cute puppies is **knowing if a puppy is in front of the car**. So today we will **build a detector that can tell when our car sees a dog or not**!

In this notebook, you'll:
- Explore the dogs vs. roads dataset
- Train a simple K-neighbors classifier for computer vision
- Train neural nets to tell dogs from roads
- Improve your model with convolutional neural networks!
- (Optional challenge) Use a saliency map to implement explainable AI
"""

#@title Run this to load some packages and data! { display-mode: "form" }
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def categorical_to_onehot(labels_in):
  labels = []
  for label in labels_in:
    if label == 'dog':
      labels.append(np.array([1, 0]))
    else:
      labels.append(np.array([0, 1]))
  return np.array(labels)

def one_hot_encoding(input):
  output = np.zeros((input.size, input.max()+1))
  output[np.arange(input.size), input] = 1

  return output


def load_data():
  # Run this cell to download our data into a file called 'cifar_data'
  !wget -O cifar_data https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%204%20_%205%20-%20Neural%20Networks%20_%20CNN/dogs_v_roads

  # now load the data from our cloud computer
  import pickle
  data_dict = pickle.load(open( "cifar_data", "rb" ));

  data   = data_dict['data']
  labels = data_dict['labels']

  return data, labels

def plot_one_image(data, labels, img_idx):
  from google.colab.patches import cv2_imshow
  import cv2
  import matplotlib.pyplot as plt

  my_img   = data[img_idx, :].reshape([32,32,3]).copy()
  my_label = labels[img_idx]
  print(f'label: {my_label}')

  fig, ax = plt.subplots(1,1)
  img = ax.imshow(my_img.astype('uint8'), extent=[-1,1,-1,1])

  x_label_list = [0, 8, 16, 24, 32]
  y_label_list = [0, 8, 16, 24, 32]

  ax.set_xticks([-1, -0.5, 0, 0.5, 1])
  ax.set_yticks([-1, -0.5, 0, 0.5, 1])

  ax.set_xticklabels(x_label_list)
  ax.set_yticklabels(y_label_list)

  fig.show(img)

def logits_to_one_hot_encoding(input):
    """
    Converts softmax output (logits) to a one-hot encoded format.

    This function takes an array of softmax output probabilities
    (usually from a neural network's output layer) and converts
    each row to a one-hot encoded vector. The highest probability
    in each row is marked as 1, with all other values set to 0.

    Parameters:
    input (numpy.ndarray): A 2D array where each row contains softmax probabilities for each class.
                            The shape of the array is (n_samples, n_classes).

    Returns:
    numpy.ndarray: A 2D array of the same shape as the input, where each row is the one-hot encoded representation
                   of the class with the highest probability in the original row.
    """

    output = np.zeros_like(input, dtype=int)
    output[np.arange(len(input)), np.argmax(input, axis=1)] = 1
    return output


class CNNClassifier:
    """
    A Convolutional Neural Network (CNN) classifier using Keras, customized for binary classification tasks.

    This class wraps a Keras Sequential model with a specific architecture suitable for image classification tasks.
    It includes a custom `predict` method that outputs one-hot encoded predictions, and other standard Keras model
    methods are accessible as well. This was done to override the need for the SciKeras wrappers that is frequently
    incompatible with Google Colab versions of Keras & Tensorflow. Feel free to modify as needed.

    Attributes:
        num_epochs (int): The number of training epochs.
        layers (int): The number of convolutional layers in the model.
        dropout (float): The dropout rate used in dropout layers for regularization.
        model (keras.models.Sequential): The underlying Keras Sequential model.

    Methods:
        build_model(): Constructs the CNN model with the specified architecture and compiles it.

        fit(*args, **kwargs): Trains the model. Accepts arguments compatible with the Keras `fit` method.

        predict(*args, **kwargs): Predicts labels for the input data. Converts the softmax output of the model
                                  to one-hot encoded format using `logits_to_one_hot_encoding`. Necessary to match
                                  accuracy_score function expected arguments.

        predict_proba(*args, **kwargs): Predicts labels for the input data and returns the raw output of the softmax.
                                        Used when wanting to inspect the raw probabilistic scoring of the model.

    Usage:
        cnn_classifier = CNNClassifier(num_epochs=30, layers=4, dropout=0.5)
        cnn_classifier.fit(X_train, y_train)
        predictions = cnn_classifier.predict(X_test)

    Note:
        The `__getattr__` method is overridden to delegate attribute access to the underlying Keras model,
        except for the `predict` method which is customized.
    """
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))

        for i in range(self.layers):
          model.add(Conv2D(32, (3, 3), padding='same'))
          model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, epochs=self.num_epochs, batch_size=10, verbose=2, **kwargs)

    #NOTE: WRITTEN TO RETURN ONE HOT ENCODINGS FOR ACCURACY
    def predict(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return logits_to_one_hot_encoding(predictions)

    def predict_proba(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def __getattr__(self, name):
        if name != 'predict' and name != 'predict_proba':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def plot_acc(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
    ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')
    ax.legend(loc = 7)
    ax.set_ylim([0.4, 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')

    plt.show()

"""# Understanding our data

Our cars are very attentive and always have their eyes on the road.

Every second, they're streaming in data about the street, including video.

From this video data, we want our car to tell: is there 'road' or 'dog' in front of it?

Lucky for us, we have a dataset of dog and road images already prepared! Let's start by reading that *labeled* data in.
"""

# load our data
data, labels = load_data()

"""Let's look at an image of a dog!

Try changing the number below. What does it do?


"""

plot_one_image(data, labels, 0) #change this number

"""### 💡 Discussion Question

Why might we be using such blurry images?

### Instructor Solution

<details><summary> click to reveal! </summary>

Due to ease of processing and storage limitations, we use 32x32 pixel RGB images for our dataset of 1200 images. This choice also accommodates varying camera quality and speeds up model training due to the smaller number of weights needed to process lower resolution images.

Next, let's try a road image. Again, try changing the number:
"""

plot_one_image(data, labels, 700) #change this number

"""How many images do we have?"""

print(len(data))
print(Counter(labels))

"""The dataset is organized such that there are 600 images of dogs and 600 images of roads.

#### Optional Exercise: Examining More Images

**Look at a few more images of both classes.**

Try using a `for` loop to look at 5 images!
"""

### YOUR CODE HERE


### END CODE

#@title Instructor Solution { display-mode: "form" }
for i in range(5):
  plot_one_image(data, labels, i)

for i in range(700, 705):
  plot_one_image(data, labels, i)

"""##Understanding our Data Representation

In an image each pixel is denoted by 3 numbers that represent the intensity value of that pixel (0 - 255) for each color channel (R, G, and B). Below we
see a list of numbers for each image that represent the intensity values.

"""

print('One image looks like:\n', data[0], '\n')
print("Length of list:", len(data[0]))

"""### 💡 Discussion Question

What does each number mean? Can you explain the length of the list?

### Instructor Solution

<details><summary> click to reveal! </summary>

Each number represents the intensity of a color channel. We have 8 bits of info per channel or 256 possible values (ranging 0-255). Three color channels make up one color pixel (R G B). We have images that are 32 pixels wide and 32 pixels tall so $32 * 32 = 1024$ gives the total number of pixels, and multiplying this by the number of channels gives $3 * 1024 = 3072$, the total number of intensity values!
"""

#@title Exercise: Fill in the correct values for each image's height, width, and number of color channels:

img_height =  None#@param {type:"integer"}
img_width =  None#@param {type:"integer"}
color_channels =  None#@param {type:"integer"}

if img_height == 32 and img_width == 32 and color_channels == 3:
  print("Correct!")
  print ("Each image is", img_height, 'x', img_width, 'pixels.')
  print ("Each pixel has", color_channels, "channels for red, green, blue.")
  print ("This gives a total of", img_height * img_width * color_channels, "intensity values per image.")
else:
  print("Those aren't quite the values.")
  print("Your values give a total of", img_height * img_width * color_channels, "intensity values per image.")
  print("Discuss with your group and try again!")

"""### Instructor Solution

<details><summary> click to reveal! </summary>

**img_height:** 32<br>
**img_width:** 32<br>
**color_channels:** 3

We use these values as **inputs** to predict an **output** label: 'dog' or 'road'!

Here's what our entire dataset looks like:
"""

print ('Data shape:', data.shape)
print ('Data:\n', data)

"""#A Simple Machine Learner

We want to create a machine learning model that can tell us whether a new image is either a dog or a road.

We will give our model a training manual of data and labels that it will study or train on.

We then check how well our model is doing on a test, where it is given data and told to predict their labels.

##Building a KNN##

Let's start by using the `KNeighborsClassifier` model.

**Playground:** Explore [this demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/) to understand what the KNN model is doing!

**Exercise:** Below, please build, train, and measure the accuracy of your own KNN model. Experiment with changing the number of neighbors!
"""

# Preparing data and create training and test inputs and labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=1)

### YOUR CODE HERE

# Initialize our model
knn_model = None # Change this!

# Train our model

# Test our model

# Print the score on the testing data

### END CODE

#@title Instructor Solution { display-mode: "form" }

# Preparing data and create training and test inputs and labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=1)

# Initializing our model
knn_model = KNeighborsClassifier(n_neighbors=3)

# Training our model with its training input data and labels
knn_model.fit(X_train, y_train)

# Predict what the classes are based on the testing data
predictions = knn_model.predict(X_test)

# Print the score on the testing data
print("KNN Testing Set Accuracy:")
print(accuracy_score(y_test, predictions)*100)

"""**(Optional):** After you've built your KNN model, remove ```random_state=1``` and re-run the cells above. How does removing ```random_state=1``` affect your accuracy? Why?

### Instructor Solution

<details><summary> click to reveal! </summary>

Eliminating the `random_state=1` parameter changes how the data is split between the training and testing sets. Since computers cannot truly be random in that they execute specific calculations asked of them, computers cannot generate truly random numbers and instead generate pseudorandom numbers. The generator algorithms take in a "seed" that will always result in the same "random" numbers produced. This means using the same seed will always give you the same train-test split!

For extra exploration, try changing the `shuffle` parameter, which is `True` by default, to see what happens when the data isn't randomly split between the training and testing sets.

##Predicting on images

We can use our trained model to predict whether our car is seeing a `dog` or `road`. Let's try this out - experiment with different images!
"""

# Specify which image you want to show
image_id = 210 #Change this!

# Visualize the image
plot_one_image(X_test, y_test, image_id)

# Use the model to predict what this might be and print it
print('prediction:', knn_model.predict([X_test[image_id]])[0])

"""## ✍ Exercise: Choosing a value of k

Determine the optimal value of "k" for our data. Use a for loop to loop through different values of "k". In particular, *at the very least* try k = 1, 3, 5, 10, 20, and 30. For each of these values of "k", define a new KNN model, train it, and evaluate the accuracy.
"""

### YOUR CODE HERE

### END CODE

#@title Instructor Solution { display-mode: "form" }

for i in [1, 3, 5, 10, 20, 30]:
  # Defining our classifier
  knn_model = KNeighborsClassifier(n_neighbors=i)

  # Training our model with its training input data and labels
  knn_model.fit(X_train, y_train)

  # predictions for test
  predictions = knn_model.predict(X_test)

  # Print the score on the testing data
  print(f"KNN testing set accuracy for {i} neighbors: {accuracy_score(y_test, predictions)*100}")

"""**Discuss:** What are the advantages and disadvantages of using a bigger vs. smaller **k**? What is the optimal value?

### Instructor Solution

<details><summary> click to reveal! </summary>

A smaller 'k' value is effective for distinct clusters but will be sensitive to outliers and noise. A larger 'k' suits datasets with overlapping clusters but will decrease the influence of the closest data.

For this dataset, an optimal 'k' is typically found to be either 2 or between 11-16.

## (Optional Exercise) Understanding our mistakes

Our classifications are OK, but are they good enough for our conscientious cars?

Let's put on our detective hats to determine the root causes of the incorrect classifications!

Below, please print out 4 images of true positives, 4 images of true negatives, 4 images of false positives, and 4 images of false negatives. What are the reasons for failure (both for false positives and false negatives)?
"""

#True Positives (code provided)
print ("TRUE POSITIVES")
tp_count = 0
i = 0
while tp_count < 4 and i < len(X_test):
  prediction = knn_model.predict([X_test[i]])[0]
  if prediction == y_test[i] and prediction == 'dog':
    plot_one_image(X_test, y_test, i)
    tp_count += 1
  i += 1

#False Positives
#YOUR CODE HERE

#True Negatives
#YOUR CODE HERE

#False Negatives
#YOUR CODE HERE

#@title Instructor Solution

# True Positives
print ("TRUE POSITIVES")
tp_count = 0
i = 0
while tp_count < 4 and i < len(X_test):
  prediction = knn_model.predict([X_test[i]])[0]
  if prediction == y_test[i] and prediction == 'dog':
    plot_one_image(X_test, y_test, i)
    tp_count += 1
  i += 1

# False Positives
print ("FALSE POSITIVES")
fp_count = 0
i = 0
while fp_count < 4 and i < len(X_test):
  prediction = knn_model.predict([X_test[i]])[0]
  if prediction != y_test[i] and prediction == 'dog':
    plot_one_image(X_test, y_test, i)
    fp_count += 1
  i += 1

# True Negatives
print ("TRUE NEGATIVES")
tn_count = 0
i = 0
while tn_count < 4 and i < len(X_test):
  prediction = knn_model.predict([X_test[i]])[0]
  if prediction == y_test[i] and prediction == 'road':
    plot_one_image(X_test, y_test, i)
    tn_count += 1
  i += 1


# False Negatives
print ("FALSE NEGATIVES")
fn_count = 0
i = 0
while fn_count < 4 and i < len(X_test):
  prediction = knn_model.predict([X_test[i]])[0]
  if prediction != y_test[i] and prediction == 'road':
    plot_one_image(X_test, y_test, i)
    fn_count += 1
  i += 1

"""### 💡 Discussion Question
What patterns did you notice? What are some reasons that the model makes mistakes?

### Instructor Solution

<details><summary> click to reveal! </summary>

Generally the false classifications are images that are much closer to the subject. One reason the model may make mistakes is that these images are on the boundaries of their clusters.

#Neural Networks
Now, let's create some new models using neural networks!

You can play around with [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.62283&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&regularization_hide=true&regularizationRate_hide=true&learningRate_hide=true&batchSize_hide=true&stepButton_hide=true&activation_hide=true) to get a feel for how neural nets work.

To build a simple neural network, we use `MLPClassifier` from scikit-learn. We will play with the **number of neurons** and the **number of hidden layers** to adjust the complexity of our model, just like we did in Playground!

**Example 1:**
Here's how we create a neural network with 1 hidden layer of 3 neurons.

```python
nnet = MLPClassifier(hidden_layer_sizes=(3))
```

**Example 2:**

Here's how we create a neural network with 2 hidden layers: one of 3 neurons and one of 4 neurons.

```python
nnet = MLPClassifier(hidden_layer_sizes=(3, 4))
```

### ✍ Exercise
How might you build a neural network with 3 hidden layers? Run the code below and modify it!
"""

# Create and train our multi layer perceptron model
nnet = MLPClassifier(hidden_layer_sizes=(3), random_state=1, max_iter=10000000)  ## How many hidden layers? How many neurons does this have?
nnet.fit(X_train, y_train)

# Predict what the classes are based on the testing data
predictions = nnet.predict(X_test)

# Print the score on the testing data
print("MLP Testing Accuracy:")
print(accuracy_score(y_test, predictions)*100)

#@title Instructor Solution (3-Layer MLP Example)
# Note that this is an example- have your students find a better architecture.

# Create and train our multi layer perceptron model
nnet = MLPClassifier(hidden_layer_sizes=(10, 5, 4), random_state=1, max_iter= 10000)  ## How many hidden layers? How many neurons does this have?
nnet.fit(X_train, y_train)

# Predict what the classes are based on the testing data
predictions = nnet.predict(X_test)

# Print the score on the testing data
print("MLP Testing Accuracy:")
print(accuracy_score(y_test, predictions)*100)

"""**How well did your neural network perform?**

Multilayer perceptrons are more complex models and it can be difficult to find the right "settings" for them. It takes some trial and error!

**Exercise: try the following out and see how well you can get your network to do!**
* Train a 1 layer, 10 neuron network for practice
* Change the number of neurons and/or add layers to see how well you can do
* Increase or decrease the number of iterations
"""

#YOUR CODE HERE

"""### ✍ Exercise: Automating our Experiments

Similar to what you did for KNNs, use a for loop to automate your investigation. Explore different numbers of hidden layers, the size of the hidden layers, and the number of iterations! How well can you get your network performing?
"""

### YOUR CODE HERE

### END CODE

#@title Instructor Solution { display-mode: "form" }
for layers in [(1,1), (3,3), (5,5), (8,6), (10,10,10), (10,10,5)]:

  print('Layer params are ...')
  print(layers)
  nnet = MLPClassifier(hidden_layer_sizes=layers, random_state=1, max_iter=100)  ## How many hidden layers? How many neurons does this have?

  nnet.fit(X_train, y_train)

  # Predict what the classes are based on the testing data
  predictions = nnet.predict(X_test)

  # Print the score on the testing data
  print("MLP Testing Accuracy:")
  print(accuracy_score(y_test, predictions) * 100)
  print()

"""# Models for Vision: Convolutional Neural Networks
There is a famous type of neural network known as convolutional neural networks (CNNs). These types of neural networks work particularly well on problems to do with computer vision. Let's try one out!

# CNNClassifier

## Overview
The `CNNClassifier` is a custom class designed for Inspirit AI focusing on teaching the application of Convolutional Neural Networks (CNN) for binary classification tasks using Keras. This class is unique and is not available from any standard libraries or repositories.

## Description
This class encapsulates a Keras Sequential model tailored for image classification. It features a customized `predict` method for one-hot encoded outputs, which bypasses the need for SciKeras wrappers that often present compatibility issues with certain versions of Keras and TensorFlow on Google Colab. The design encourages experimentation and modification to suit different learning or project needs. Take a look in the large import box above to see its definition!

## Attributes
- `num_epochs` (int): Number of training epochs.
- `layers` (int): Number of convolutional layers.
- `dropout` (float): Dropout rate for regularization.
- `model` (keras.models.Sequential): The base Keras Sequential model.

## Methods
- `build_model()`: Sets up the CNN architecture and compiles the model.
- `fit(*args, **kwargs)`: Trains the model using parameters compatible with Keras’s `fit` method.
- `predict(*args, **kwargs)`: Outputs one-hot encoded predictions.
- `predict_proba(*args, **kwargs)`: Provides raw softmax output for detailed probabilistic analysis.

### Training Your CNN with One Hot Encoded Labels
For initiating a basic CNN in Keras, execute the following command:

`cnn = CNNClassifier(num_epochs=N)`

Here, `num_epochs` denotes the number of complete passes the neural network will make through the training dataset.

Before training, it's crucial to preprocess our data. Specifically, we need to convert the data to floating-point (decimal) numbers. Additionally, our current labels, which are string categories like "dog" or "road", must be transformed into one-hot encodings. This conversion is essential for the neural network to process them correctly.

**Exercise:** Convert your string labels to one-hot encodings using `categorical_to_onehot(data)`, then proceed to train and test your CNN using the modified data. Make sure to save these variables as `y_train_onehot` and `y_test_onehot` We've taken care of changing the data (`X_train` & `X_test`) to decimal numbers.
"""

# convert our data to floats for our CNN
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# convert our labels to one-hot vectors!


### YOUR CODE HERE
# Create and train our cnn

# Predict what the classes are based on the testing data

# Print the score on the testing data


### END CODE

#@title Instructor Solution { display-mode: "form" }
# convert our data to floats for our CNN
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# convert our labels to one-hot vectors!
y_test_onehot = categorical_to_onehot(y_test)
y_train_onehot = categorical_to_onehot(y_train)

# Create and train our CNN model
cnn = CNNClassifier(num_epochs=40)

cnn.fit(X_train, y_train_onehot)

# Predict what the classes are based on the testing data
predictions = cnn.predict(X_test)

# Print the score on the testing data
print("CNN Testing Set Score:")
print(accuracy_score(y_test_onehot, predictions)*100)

"""**Discuss: Is this CNN good enough to use in practice?**

CNNs typically perform better than basic Neural Networks on vision problems - but like basic Neural Networks, they aren't always consistent in their results and are sensitive to a number of factors.

If you're interested in learning more about CNNs, spend some time exploring the [CNN Explainer](https://poloclub.github.io/cnn-explainer/)!

**Report to the class your highest model accuracy.**

**Bonus Question:** Each of you might see a different max accuracy. Can you think of why that might be?

### Instructor Solution

<details><summary> click to reveal! </summary>

Hint: *Consider the stages in the machine learning process.* The data is split into training and testing sets randomly, so the model learns from varying data each time. Additionally, neural network performance is influenced by multiple parameters, such as initialization, which can result in different outcomes.

## Training and Validation Curves

An important aspect of training neural networks is to prevent overfitting. **How do you know when your model is overfitting?**

To plot our model's history, we can train it with
```
history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot))
```

and then use
```
plot_acc(history)
```
Don't forget to change ```model``` to be the name of your model!

**Exercise:** Train a CNN model and plot a train vs. test curve.

**After how many epochs does the model begin to overfit?** Overfitting occurs when the validation accuracy starts to drop below the training accuracy.
"""

### YOUR CODE HERE

### END CODE

#@title Instructor Solution { display-mode: "form" }

cnn = CNNClassifier(num_epochs=20)

history = cnn.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot))

plot_acc(history)

"""### Hopefully your CNN worked *very* well! We want to keep the doggos as safe as they can be.

![](https://images.pexels.com/photos/316/black-and-white-animal-dog-pet.jpg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

# Challenge Exercise: Explainability through Saliency Maps

Neural networks have achieved incredible results in many fields. But they have a big problem: it’s very difficult to explain exactly why a neural network makes the decisions it does. This makes it difficult to trust them in high-stakes applications like medicine, self-driving cars, and criminal justice - would you trust an AI that diagnosed you with a disease, but couldn’t explain why?

Other classifiers are much more explainable:

*   With logistic regression, we can see the coefficient (importance) attached to each input feature.
*   With a decision tree, we can trace a particular decision down the tree.
*   With KNN, we can examine the nearby neighbors.

Our CNN, above, works well. For example, let's try choosing an image from our dataset and classifying it.
"""

image_index = 220 # pick any image you'd like
input_image = X_test[image_index]
print(input_image.shape)
print(input_image) # How many numbers are there? What does each represent?

plt.imshow(input_image.reshape(32,32,3).astype(int))
plt.show()

print('Classification:')
if(np.argmax(cnn.predict(np.array([input_image]))) == 0):
  print("Predicted: Dog")
else:
  print("Predicted: Road")
# 0 means dog, 1 means road

"""But why did the CNN reach that decision? It’s really hard to give a clear answer! The CNN relies on multiplying input features by the weights it has set. You can print out and look at the hundreds of weights:

"""

# Warning: expect a large output!
for layer_weights in history.model.weights:
  print (layer_weights)

"""Unfortunately, that probably didn’t help you make a useful explanation.

Researchers are currently studying ways to make neural networks more explainable. One approach is using **saliency maps** to figure out the saliency (importance) of each individual pixel. Check out a demo [here](https://lrpserver.hhi.fraunhofer.de/image-classification). Intuitively, we're trying to understand the neural network by tracking what it "pays attention" to, in the same way that psychologists study babies' cognition by [tracking what babies look at](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3259733/).

In this exercise, we're going to build a simple version of a saliency map for the image you chose above. We'll see what pixels were most important in helping the network make its classification.

To do this, we'll investigate the effects of changing each pixel a little bit. If changing a particular pixel changes the result a lot, we conclude that pixel must be important for classifying. If changing that pixel doesn't change the result, we conclude that pixel is unimportant.

We're going to use the raw predicted probabilities, rather than the final classification.
"""

pred = cnn.predict_proba(np.array([input_image])) # What does each number mean?
print(pred)
dog_prob = pred[0][0] # This is the probability we'll use (if we know dog prob, we know the classification)

print('Probability of dog:')
print(dog_prob)

"""Now, we need to calculate the saliency for each pixel (really, each RGB value). The core idea is that a pixel's saliency is the average value of

 $D = \left|\frac{\Delta probability}{\Delta pixel}\right|$

 where $\Delta$ is the amount of change. If a small change in the pixel value results in a large change in the probability (either up or down), we know this pixel is important. If you've seen derivatives in calculus, this idea should feel familiar.

Here's the game plan:

*   Consider each pixel value in turn: R, G, B, then the next pixel.
*   Make a copy of the image array before you change anything!
*   Make the pixel value larger or smaller by various amounts. Each time, find the CNN's prediction with the changed value, and calculate the value of D.
*   Repeat the previous step a few times, and calculate the pixel's saliency: the average value of D.
*   Store the saliency of each pixel in a list, so that we can visualize it later.

Try it below! (Warning: this code might be very slow. As a further challenge, try to speed it up!)

"""

saliencies = [] # eventually, will be the same size as input_image

for index, pixel in enumerate(input_image):
  # index counts up from 0, pixel is between 0 and 255

  if index % 100 == 0: # will track progress - this might take a while
    print (index)

  changed_input = input_image.copy() # make sure not to change the original input_image!

  # YOUR CODE HERE:
  # In changed_input, change the value of this pixel by some amount.
  # Use the CNN to classify changed_input.
  # Calculate the value of D.
  # Repeat with various-size changes, and calculate saliency as the average D.
  saliency = 0 # Change this!

  saliencies.append(saliency)

print(saliencies)

#@title Faster Student submitted solution
# Thanks to Paul Cherian (Inspirit AI Winter 2020 Student) for this solution!

saliencies = [] #eventually, will be the same size as input_image
all_changed_pixels = []
pixel_differences = []
for index, pixel in enumerate(input_image):
  #index counts up from 0, pixel is between 0 and 255
  # if index%100 == 0: #will track progress - this might take a while
    # print (index)
  # if index>500:
  #   break
  changed_input = input_image.copy() #make sure not to change the original input_image!

  # A much faster approach would be vectorize - create an array with all the changed
  #versions so that we can feed them all into the CNN at the same time.
  D_list = []
  changed_versions_of_pixel = []

  for pixel_change in [-50, -30, -10, 10, 30, 50]:
    changed_pixel = pixel + pixel_change

    if 0 <= changed_pixel <= 255:
      #add all the changed pixels to a list
      changed_versions_of_pixel.append(changed_pixel)
      pixel_differences.append(pixel- changed_pixel)

  # add the list of changed pixels to another list
  all_changed_pixels.append(changed_versions_of_pixel)

# make a 'stack' of all images with their changed pixel in each by using the
#changed input as the template and then reverting it back to the original when
# we move to the next pixel
changed_images = []

for j in range(len(changed_input)):
    for i in range(len(all_changed_pixels[j])):
      changed_input[j]= all_changed_pixels[j][i]
      changed_images.append(changed_input)
      changed_input = input_image.copy()

a = cnn.predict_proba(np.array([input_image]))
b = cnn.predict_proba(np.array(changed_images))

a = np.log(a)
b = np.log(b)

dog_prob = a[0][0]
new_b = []
for i in b:
  new_b.append(abs(i[0]) + abs(i[1]))
new_b = np.array(new_b)
probability_changes = new_b - dog_prob
d_total = abs(probability_changes/pixel_differences)

# the d_total list is all the values of D for each pixel with it's changes, it
# needs to be averaged, but because we ommitted some changed values that were not
# in the 0-255 range, use 'start' to 'end' to splice the array.

start = 0
end = 0
for i in all_changed_pixels:
  end += len(i)
  saliency = (np.mean(d_total[start:end]))
  start = end
  saliencies.append(saliency)

print("Non-Normalized Saliencies: \n", saliencies)

#@title Instructor Solution
#Slow, simpler version.
#A much faster approach would be vectorize - create an array with all the changed
#versions so that we can feed them all into the CNN at the same time.
"""
saliencies = []
for index, pixel in enumerate(input_image):
  if index%100 == 0: #track progress
    print (index)
  changed_input = input_image.copy()
  D_list = []
  for pixel_change in [-50, -30, -10, 10, 30, 50]:
    changed_pixel = pixel + pixel_change
    if 0 <= changed_pixel <= 255:
      changed_input[index] = changed_pixel
      changed_pred = cnn.predict_proba(np.array([changed_input]))
      changed_dog_prob = changed_pred[0][0]
      D = (changed_dog_prob - dog_prob)/pixel_change
      D_list.append(np.abs(D))
  saliency = np.mean(D_list)
  saliencies.append(saliency)

print (saliencies)
"""

"""You'll notice that your saliencies are probably very small values, since each individual pixel has a small effect on the output.
Here are the current min and max:
"""

sal_array = np.array(saliencies)
print (sal_array.min(), sal_array.max())
print (sal_array.shape)

"""To plot the saliencies, we need to do some arithmetic to transform them to a range of 0 to 1. Can you explain the function of each line?"""

sal_array = np.array(saliencies)
sal_array = sal_array - sal_array.min()
#TODO print min and max

sal_array = sal_array / sal_array.max()
#TODO print min and max

#Can you perform this transformation in a single line of code?

print (sal_array.shape)

#@title Instructor Solution
print (sal_array.min(), sal_array.max())
sal_array = (sal_array - sal_array.min()) / (sal_array.max() - sal_array.min())
print (sal_array.min(), sal_array.max())
sal_array.shape

"""Finally, we can plot our saliency map!

If you're not getting great results, try experimenting with how much you're changing the pixel values.
"""

#Plot our original image
plt.imshow(input_image.reshape(32,32,3).astype(int))
plt.show()

#Plot our saliency map: the brighter, the higher the saliency
plt.imshow(sal_array.reshape(32,32,3))
plt.show()

#Plot our saliency map superimposed on the image
plt.imshow(input_image.reshape(32,32,3).astype(int))
plt.imshow(sal_array.reshape(32,32,3),alpha=0.6)
plt.show()

"""We now have some insight into our neural network! We know which pixels matter in its decisions.

You can experiment with the definition of saliency we used above; you might come up with a better way to measure it!
"""
