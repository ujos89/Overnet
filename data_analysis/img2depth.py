import pandas as pd
import numpy as np
import os
import skimage.io as io
from pypfm import PFMLoader
# graph visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

path_depth_img = '/home/ujos89/Desktop/Vision/DPT/output_monodepth/'
path_depth_true = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/depth/'
file_names = os.listdir(path_depth_img)
loader = PFMLoader(color=False, compress=False)

def plot3d(npData):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # set coordinate : set (0,0) to upper left  
    X = np.arange(npData.shape[0])
    Y = np.arange(npData.shape[1])
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.transpose(npData), cmap=cm.coolwarm, linewidth=0, antialiased=False, rcount=200, ccount=200)

    # label to axis
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth[m]")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot3d_contour(npData):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # set coordinate : set (0,0) to upper left  
    X = np.arange(npData.shape[1])
    Y = np.arange(npData.shape[0])
    X, Y = np.meshgrid(X, Y)

    depth3D = ax.contour3D(Y, X, npData, 20, cmap=cm.jet)
    
    # label to axis
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth[m]")

    clabel = plt.colorbar(depth3D, ax=ax, shrink =.7)
    plt.show()

for file_name in file_names:
    if file_name.endswith('.png'):
        print("png file name:",file_name)
        # type: numpy ndarray
        depth_img_png = io.imread(path_depth_img+file_name)
        plot3d_contour(depth_img_png)

    elif file_name.endswith('.pfm'):
        print("pfm file name",file_name)
        # type: numpy ndarray
        depth_img_pfm = loader.load_pfm(path_depth_img+file_name)
        plot3d_contour(depth_img_pfm)

        # true value of depth from nyu_depth_v2
        file_name_csv = file_name.replace('.pfm','.csv')
        print("csv file name", file_name_csv)
        depth_true_value = pd.read_csv(path_depth_true+file_name_csv).to_numpy()
        plot3d_contour(depth_true_value)
