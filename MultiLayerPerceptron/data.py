import idx2numpy
import numpy as np

images = list(idx2numpy.convert_from_file('t10k-images-idx3-ubyte')/255)
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
new_labels = np.zeros((len(labels), 10))
new_labels[np.arange(len(labels)), labels] = 1
# new_labels = [np.zeros(10, dtype=int)]*len(labels)
# for k in range(len(labels)):
#     new_labels[k][labels[k]] = 1
train_set = list(zip(images, new_labels))
# print(train_set[0])

np.save(open('test.npy', 'wb'), train_set)

# arr = np.load(open('train.npy', 'rb'))

# print(arr[0])