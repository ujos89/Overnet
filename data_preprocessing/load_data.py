import h5py
import numpy as np
import skimage.io as io

# data from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
path_data = '../../dataset/nyu_depth_v2/labeled/raw/nyu_depth_v2_labeled.mat'
raw_data = h5py.File(path_data, mode='r+')

################


## <depths>: NxWxH->1449x640x480 [m]
depth_h5py = raw_data['depths']




######################


# print(raw_data.keys())
# KeysViewHDF5 ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']

## functions
def h5py2str(raw_data, col_name):
    data_h5py = raw_data[col_name]
    data = np.array([])
    for idx in range(data_h5py.shape[1]):
        _ = ''.join(chr(_) for _ in raw_data[data_h5py[0][idx]])
        data = np.append(data, _)
    
    #type: numpy array
    return data

print(h5py2str(raw_data, 'names'))


## <names>: 1xC, names to 1d-array
# raw_data (h5py) -> names
names = h5py2str(raw_data, 'names')
# print(names.shape)
#(894, )

## <images>: Nx3XWxH->(1449x3X640x480), conevert to 

img_h5py = raw_data['images'][0]
img = np.empty([480, 640, 3])
img[:,:,0] = img_h5py[0,:,:].T
img[:,:,1] = img_h5py[1,:,:].T
img[:,:,2] = img_h5py[2,:,:].T

img = img.astype('float32')
io.imshow(img/255.0)
io.show()


## <rawDepthFilenames>: Nx1 
depth_names_h5py = raw_data['rawDepthFilenames']
depth_names = np.array([])

for i in range(depth_names_h5py.shape[1]):
    depth_name = ''.join(chr(_) for _ in raw_data[depth_names_h5py[0][i]])
    depth_names = np.append(depth_names, depth_name)
print(depth_names.shape)
print(depth_names)

## <rawRgbFilenames>: Nx1 
rgb_names = h5py2str(raw_data, 'rawRgbfilenames')

## error on use function
##KeyError: "Unable to open object (object 'rawRgbfilenames' doesn't exist)"

# e.g.
# 1448 -th, name:  dining_room_0036/d-1315329855.768027-4071441432.pgm dining_room_0036/r-1315329855.782488-4072768344.ppm



# HxW: 1...C, 0:unlabeled
labels = raw_data['labels'][0]

# print(type(labels), type(names), type(img))
# type: numpy.ndarray

# print(labels.shape, names.shape, img.shape)
# (640,480) (894,) (3, 640, 480)

