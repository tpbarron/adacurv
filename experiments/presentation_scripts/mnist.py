import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n = 10

import numpy as np
import matplotlib.pyplot as plt

im1 = x_train[0]
im2 = x_train[1]
im3 = x_train[2]
im4 = x_train[3]
im5 = x_train[4]
plt.imshow(np.concatenate([im1, im2, im3, im4, im5], axis=1), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('mnist.pdf')
# plt.show()
