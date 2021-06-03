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
path_label = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/label/'
path_save_output = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/dpt/'
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
    #plt.show()

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

def plot3d_contours(npData1, npData2, npData3):
    fig = plt.figure()
    
    # set coordinate : set (0,0) to upper left  
    X = np.arange(npData1.shape[1])
    Y = np.arange(npData1.shape[0])
    X, Y = np.meshgrid(X, Y)

    # Graph1
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    # depth3D_1 = ax1.contour3D(Y, X, npData1, 20, cmap=cm.RdPu)
    # rstride:row step size, cstride:column step size, cmap:colour map, linewidth: wireframe line width
    depth3D_1 = ax1.plot_surface(Y, X, npData1, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax1.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='label')

    # Graph2
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    depth3D_2 = ax2.plot_surface(Y, X, npData2, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax2.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='Output_pfm')
    
    # Graph3
    ax3 = fig.add_subplot(1,3,3, projection='3d')
    depth3D_3 = ax3.plot_surface(Y, X, npData3, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax3.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='True value')

    plt.show()

def main():
    for file_name in file_names:
        if file_name.endswith('.png'):
            ##file name
            file_name_pfm = file_name.replace('.png','.pfm')
            file_name_csv = file_name.replace('.png','.csv')

            ## png file
            print()
            print("png file name:",file_name)
            # type: numpy ndarray
            depth_img_png = io.imread(path_depth_img+file_name)
            print("Range: ", np.min(depth_img_png), " ~ ", np.max(depth_img_png))
            print("Shape: ", depth_img_png.shape)
            #plot3d_contour(depth_img_png)

            #save output of png
            df_dpt_png = pd.DataFrame(depth_img_png)
            df_dpt_png.to_csv(path_save_output+'/dpt_png/'+file_name_csv,index = False)

            ## pfm file
            print("pfm file name",file_name_pfm)
            # type: numpy ndarray
            depth_img_pfm = loader.load_pfm(path_depth_img+file_name_pfm)
            depth_img_pfm = depth_img_pfm[::-1]
            print("Range: ", np.min(depth_img_pfm), " ~ ", np.max(depth_img_pfm))
            print("Shape: ", depth_img_pfm.shape)
            #plot3d_contour(depth_img_pfm)

            # save output of pfm
            df_dpt_pfm = pd.DataFrame(depth_img_pfm)
            df_dpt_pfm.to_csv(path_save_output+'/dpt_pfm/'+file_name_csv, index = False)

            # true value of depth from nyu_depth_v2
            print("csv file name", file_name_csv)
            depth_true_value = pd.read_csv(path_depth_true+file_name_csv).to_numpy()
            depth_true_value = np.rot90(depth_true_value)
            depth_true_value = depth_true_value[::-1]
            print("Range: ", np.min(depth_true_value), " ~ ", np.max(depth_true_value))
            print("Shape: ", depth_true_value.shape)
            #plot3d_contour(depth_true_value)

            #label 
            img_label = pd.read_csv(path_label+file_name_csv).to_numpy()
            img_label = np.rot90(img_label)
            img_label = img_label[::-1]

            np.unique

            print("Range: ", np.min(img_label), " ~ ", np.max(img_label))
            print("Shape: ", img_label.shape)

            #graph visualization
            plot3d_contours(img_label, depth_img_pfm, depth_true_value)

if __name__=='__main__':
    main()
