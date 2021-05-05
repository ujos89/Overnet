import h5py
import numpy as np

# data from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
path_data = '../../dataset/nyu_depth_v2/labeled/raw/nyu_depth_v2_labeled.mat'
raw_data = h5py.File(path_data, mode='r+')

# print(raw_data.keys())
# KeysViewHDF5 ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']


## <names>: 1xC, names to 1d-array
# raw_data (h5py) -> names
names_h5py = raw_data['names']
names = np.array([])
for i in range(names_h5py.shape[1]):
    name = ''.join(chr(_) for _ in raw_data[names_h5py[0][i]])
    names = np.append(names, name)
# print(names.shape)
#(894, )

## <images>: Nx3XWxH, 1449x3X640x480
print(raw_data['images'].shape)
img = raw_data['images'][0]


# HxW: 1...C, 0:unlabeled
labels = raw_data['labels'][0]

# print(type(labels), type(names), type(img))
# type: numpy.ndarray

# print(labels.shape, names.shape, img.shape)
# (640,480) (894,) (3, 640, 480)

