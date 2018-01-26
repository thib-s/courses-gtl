
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib import DistToAverageClassifier


# In[ ]:

# load our train and test datasets
train_df = pd.read_csv("datasets/fashion_mnist/fashion-mnist_train.csv")
test_df = pd.read_csv("datasets/fashion_mnist/fashion-mnist_test.csv")


# In[67]:

categories = ["T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot" ]


# In[61]:

def gen_image(arr):
    """
    function to display an mnist image
    """
    image = np.array(arr, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

classifier = DistToAverageClassifier.DistToAverageClassifier()
classifier.fit(train_df)

print (classifier.score(test_df.values[:, 1:785], test_df.values[:, 0]))