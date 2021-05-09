import h5py
import numpy as np
import skimage.io as io
import pandas as pd

# data from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
path_data = '../../dataset/nyu_depth_v2/labeled/raw/nyu_depth_v2_labeled.mat'
raw_data = h5py.File(path_data, mode='r')

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

# name to title
def name2title(name):
    #image file title
    title = name.split('.')[0]
    
    return ''.join(_ for _ in title.split('/r'))

# image data from h5py to png file
def h5py2img(img_h5py, title, path):
    img = np.empty([480, 640, 3])
    img[:,:,0] = img_h5py[0,:,:].T
    img[:,:,1] = img_h5py[1,:,:].T
    img[:,:,2] = img_h5py[2,:,:].T
    img = img.astype('float32')
    io.imsave(path+title+'.png', img/255.0)

    # show image
    # io.imshow(img/255.0)
    # io.show()

    # guide
    # h5py2img(raw_data['images'][0], 'test_img','./')

# show depth(depth visualization)
# depth_h5py = raw_data['depths'][0]
# depth = np.empty([480, 640, 3])
# depth[:,:,0] = depth_h5py[:,:].T
# depth[:,:,1] = depth_h5py[:,:].T
# depth[:,:,2] = depth_h5py[:,:].T
# io.imshow(depth/4.0)
# io.show()

def h5py2csv(h5py, title, path):
    data = pd.DataFrame(h5py)
    data.to_csv(path+title+'.csv', index=False)

## <rawDepthFilenames>: Nx1->1449x1
#depth_names = h5py2str(raw_data, 'rawDepthFilenames')

## <rawRgbFilenames>: Nx1 
rgb_names = h5py2str(raw_data, 'rawRgbFilenames')

title = np.array([]) 
for _ in range(1449):
    title = np.append(title, name2title(rgb_names[_]))

## <names>: 1xC-, names to 1d-array
# print(names.shape) -> (894, )
label_title = h5py2str(raw_data, 'names')

label_title_df = pd.DataFrame(label_title)
label_title_df.to_csv('../../dataset/nyu_depth_v2/labeled/title/label_title.csv', index=False)

title_df = pd.DataFrame(title)
title_df.to_csv('../../dataset/nyu_depth_v2/labeled/title/title.csv', index=False)

## <labels>: HxW: 1...C, 0:unlabeled
## <images>: Nx3XWxH->(1449x3X640x480)
## <depths>: NxWxH->1449x640x480 [m]

for idx in range(1449):
    img_h5py = raw_data['images'][idx]
    depth_h5py = raw_data['depths'][idx]
    label_h5py = raw_data['labels'][idx]

    h5py2img(img_h5py, title[idx], '../../dataset/nyu_depth_v2/labeled/image/')
    h5py2csv(depth_h5py, title[idx], '../../dataset/nyu_depth_v2/labeled/depth/')
    h5py2csv(label_h5py, title[idx], '../../dataset/nyu_depth_v2/labeled/label/')


