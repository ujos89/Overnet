from pypfm import PFMLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from matplotlib import rc
from matplotlib import cm 

plt.rcParams['font.family'] = 'Times New Roman'
font = {'size'   : 30}
rc('text', usetex=True)
 


fPath = '../data/test1.pfm'
loader = PFMLoader(color=False, compress=False)
npData = loader.load_pfm(fPath)

xDataLen = npData.shape[0]
yDataLen = npData.shape[1]
dData = npData

xData = np.linspace(0, xDataLen, yDataLen)
yData = np.linspace(0, yDataLen, xDataLen)


x, y = np.meshgrid(xData, yData)
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection='3d')

depth3D = ax.contour3D(x, y, dData, 20, cmap=cm.jet, label='depth map')

ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
ax.set_zlabel(r"depth (mm)",  fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.zaxis.set_tick_params(labelsize=20)

clabel = plt.colorbar(depth3D, ax=ax, shrink=0.7)
clabel.set_label(label = r'depth (mm)',  fontsize=20)
depth3D.ax.tick_params(labelsize=20) 
plt.show()