import pandas as pd
import numpy as np
import skimage.io as io

title = pd.read_csv('../../dataset/nyu_depth_v2/labeled/title/title.csv')
depth_path = '../../dataset/nyu_depth_v2/labeled/depth/'
depthimg_path = '../../dataset/nyu_depth_v2/labeled/depth_img/'

# experiment: find difference 
# depth = pd.read_csv(depth_path+'bathroom_0013-1300633382.csv').to_numpy()
# io.imsave('./1.png',depth)
# io.imsave('./1000.png',depth*1000)
# io.imsave('./1000int.png',depth.astype(np.uint16))

for idx in range(len(title['0'])):
    depth = pd.read_csv(depth_path+title['0'][idx]+'.csv').to_numpy()
    io.imsave(depthimg_path+title['0'][idx]+'.png',depth)