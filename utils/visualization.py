import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def plot3d_contours(npData1, npData2, npData3, file_name="graph"):
    fig = plt.figure()
    
    # Graph1
    # set coordinate : set (0,0) to upper left  
    X = np.arange(npData1.shape[1])
    Y = np.arange(npData1.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    # depth3D_1 = ax1.contour3D(Y, X, npData1, 20, cmap=cm.RdPu)
    # rstride:row step size, cstride:column step size, cmap:colour map, linewidth: wireframe line width
    depth3D_1 = ax1.plot_surface(Y, X, npData1, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax1.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='True value from nyu_v2_depth')

    # Graph2
    X = np.arange(npData2.shape[1])
    Y = np.arange(npData2.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    depth3D_2 = ax2.plot_surface(Y, X, npData2, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax2.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='Prediction (Single image)')
    
    # Graph3
    X = np.arange(npData3.shape[1])
    Y = np.arange(npData3.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax3 = fig.add_subplot(1,3,3, projection='3d')
    depth3D_3 = ax3.plot_surface(Y, X, npData3, rstride=20, cstride=20, cmap=cm.RdPu, linewidth=10, antialiased=True, label='label')
    ax3.set(xlabel='Height', ylabel='Width', zlabel='Depth', title='Prediction (Overlapped cropped image)')

    plt.show()
    plt.savefig('graph/'+file_name+'.png')
