import h5py
import numpy as np
import skimage.io as io
import pandas as pd

# data from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
path_data = '../../dataset/nyu_depth_v2/labeled/raw/nyu_depth_v2_labeled.mat'
raw_data = h5py.File(path_data, mode='r')

################


######################


# print(raw_data.keys())
# KeysViewHDF5 ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']

## functions

# <HDF5 object reference> to string
def h5py2str(raw_data, col_name):
    data_h5py = raw_data[col_name]
    data = np.array([])
    for idx in range(data_h5py.shape[1]):
        _ = ''.join(chr(_) for _ in raw_data[data_h5py[0][idx]])
        data = np.append(data, _)
    
    #type: numpy array
    return data

# names to title
def names2title(name):
    return name.split('.',)[0]


## <names>: 1xC-, names to 1d-array
# print(names.shape) -> (894, )
obj_names = h5py2str(raw_data, 'names')

## <rawDepthFilenames>: Nx1->1449x1
depth_names = h5py2str(raw_data, 'rawDepthFilenames')

## <rawRgbFilenames>: Nx1 
rgb_names = h5py2str(raw_data, 'rawRgbFilenames')

depth_title, rgb_title = np.array([]), np.array([]) 
for _ in range(len(depth_names)):
    depth_title = np.append(depth_title, names2title(depth_names[_]))
    rgb_title = np.append(rgb_title, names2title(rgb_names[_]))




# HxW: 1...C, 0:unlabeled
labels = raw_data['labels'][0]

# print(type(labels), type(names), type(img))
# type: numpy.ndarray

# print(labels.shape, names.shape, img.shape)
# (640,480) (894,) (3, 640, 480)


## <images>: Nx3XWxH->(1449x3X640x480), conevert to 

img_h5py = raw_data['images'][0]
img = np.empty([480, 640, 3])
img[:,:,0] = img_h5py[0,:,:].T
img[:,:,1] = img_h5py[1,:,:].T
img[:,:,2] = img_h5py[2,:,:].T

img = img.astype('float32')
# show image
# io.imshow(img/255.0)
# io.show()

# save image to png
# io.imsave('./test.png', img/255.0)

## <depths>: NxWxH->1449x640x480 [m]

depth_h5py = raw_data['depths'][0]

depth = np.empty([480, 640, 3])
depth[:,:,0] = depth_h5py[:,:].T
depth[:,:,1] = depth_h5py[:,:].T
depth[:,:,2] = depth_h5py[:,:].T

# show depth(depth visualization)
io.imshow(depth/4.0)
io.show()

# save depth to csv
depth_df = pd.DataFrame(depth_h5py)
depth_df.to_csv('test.csv', index=False)


