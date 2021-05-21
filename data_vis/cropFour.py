from PIL import Image
from pypfm import PFMLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from matplotlib import rc
from matplotlib import cm 
import pandas as pd

fPath = '../DPT/output_monodepth/c1.pfm'
fPathTot ='../DPT/output_monodepth/test.pfm' 
loader = PFMLoader(color=False, compress=False)

npData = loader.load_pfm(fPath)
# print(npData.shape)
npData = np.flipud(npData)
# npData = npData.transpose()
xDataLen = npData.shape[0]
yDataLen = npData.shape[1]

dEData = pd.DataFrame(npData)


npDataTot = loader.load_pfm(fPathTot)
# print(npDataTot.shape)

dEDataTot = npDataTot
# dEDataTot = npDataTot.transpose()
pdDataTot = pd.DataFrame(dEDataTot)
pdDataTot = pdDataTot.iloc[:240, :320]
# print(pdDataTot)

xData = np.linspace(0, xDataLen, yDataLen)
yData = np.linspace(0, yDataLen, xDataLen)
    
x, y = np.meshgrid(xData, yData)
# gPath = '../DPT/ground_monodepth/test.csv'
# dGtData = pd.read_csv(gPath)
# dGtData = dGtData * 1000.
# dGtData = dGtData.iloc[:320, :240]

# print(dGtData.shape)
# print(dEData.shape)    

dError = pdDataTot.values - dEData
# dError = np.abs(dError) 
# print(dError.shape)
rmsError = np.sqrt(np.mean((dError)**2))
print(rmsError) 


# fig1 = plt.figure(figsize=(24, 12))
# ax = fig1.gca(projection='3d')
# depth3D = ax.plot_surface(x, y, dEData,
#                        rstride=20,                   # row step size
#                        cstride=20,                  # column step size
#                        cmap=cm.RdPu,       # colour map
#                        linewidth=10,               # wireframe line width
#                        antialiased=True,
#                        label='Estimation depth map - crop')

# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Crop Image - Estimated depth (mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depth3D, ax=ax, shrink=0.7)
# clabel.set_label(label = r'depth (mm)',  fontsize=20)
# # depth3D.ax.tick_params(labelsize=20) 

# fig2 = plt.figure(figsize=(24, 12))
# ax = fig2.gca(projection='3d')
# depthEr = ax.plot_surface(x, y, pdDataTot.values,
#                        rstride=20,                   # row step size
#                        cstride=20,                  # column step size
#                        cmap=cm.RdPu,       # colour map
#                        linewidth=10,               # wireframe line width
#                        antialiased=True,
#                        label='Estimation depth map - crop')
# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Total Image - Estimated depth(mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depthEr, ax=ax, shrink=0.7)
# clabel.set_label(label = r' Total Image (mm)',  fontsize=20)
# # depthEr.ax.tick_params(labelsize=20) 


# fig3 = plt.figure(figsize=(24, 12))
# ax = fig3.gca(projection='3d')
# depthEr = ax.plot_surface(x, y,dError,
#                        rstride=20,                   # row step size
#                        cstride=20,                  # column step size
#                        cmap=cm.RdPu,       # colour map
#                        linewidth=10,               # wireframe line width
#                        antialiased=True,
#                        label='Estimation depth map - crop')
# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Error (mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depthEr, ax=ax, shrink=0.7)
# clabel.set_label(label = r'Error Truth (mm)',  fontsize=20)
# depthEr.ax.tick_params(labelsize=20) 
    



# fig1 = plt.figure(figsize=(24, 12))
# ax = plt.axes(projection='3d')
# depth3D = ax.contour3D(x, y, dEData, 20, cmap=cm.jet, label='Estimation depth map - crop')
# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Estimated depth (mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depth3D, ax=ax, shrink=0.7)
# clabel.set_label(label = r'depth (mm)',  fontsize=20)
# depth3D.ax.tick_params(labelsize=20) 

# fig2 = plt.figure(figsize=(24, 12))
# ax = plt.axes(projection='3d')
# depthEr = ax.contour3D(x, y, dGtData.values, 20, cmap=cm.jet, label='Ground Truth depth map- crop')
# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Ground Truth depth(mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depthEr, ax=ax, shrink=0.7)
# clabel.set_label(label = r'Ground Truth (mm)',  fontsize=20)
# depthEr.ax.tick_params(labelsize=20) 
    
# fig3 = plt.figure(figsize=(24, 12))
# ax = plt.axes(projection='3d')
# depthEr = ax.contour3D(x, y, dError, 20, cmap=cm.jet, label='Ground Truth depth map- crop')
# ax.set_xlabel(r"x (mm)"      ,  fontsize=20)
# ax.set_ylabel(r"y (mm)"      ,  fontsize=20)
# ax.set_zlabel(r"Error (mm)",  fontsize=20)
# ax.xaxis.set_tick_params(labelsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.zaxis.set_tick_params(labelsize=20)
# clabel = plt.colorbar(depthEr, ax=ax, shrink=0.7)
# clabel.set_label(label = r'Ground Truth (mm)',  fontsize=20)
# depthEr.ax.tick_params(labelsize=20) 
    
    
plt.show()