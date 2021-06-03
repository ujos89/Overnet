import pandas as pd
import numpy as np
import os

from data_analysis.imgCrop import rawcsv2np, dptcsv2np
from data_analysis.img2depth import plot3d_contours

def summary_np(npData):
    print("Shape:",npData.shape)
    print("Range:",np.min(npData), "~", np.max(npData))
    print()

def extract_raw(npData, boundary):
    l, r, u, d = boundary[0,0], boundary[1,0], boundary[2,0], boundary[3,0]

    return npData[u:d, l:r]

def extract_cropped(npData, boundary, methods='avg'):
    np_ul, np_ur, np_bl, np_br = npData
    l, r, u, d = boundary[0,0], boundary[1,0], boundary[2,0], boundary[3,0]

    over_ul = np_ul[u:, l:]
    over_ur = np_ur[u:, :(r-l)]
    over_bl = np_bl[:(d-u), l:]
    over_br = np_br[:(d-u), :(r-l)]

    if methods == 'avg':
        return (over_ul+over_ur+over_bl+over_br)/4


def main():
    path_depth_true = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/depth/'
    path_dpt_img = 'dpt/image/'
    path_dpt_depth = 'dpt/depth/'
    path_crop_boundary = 'cropped_img/boundary/'
    path_crop_img = 'cropped_img/image/'
    path_crop_depth = 'cropped_img/depth/'

    file_names_raw = os.listdir(path_dpt_depth)
    file_names_crop = os.listdir(path_crop_depth)

    for file_name in file_names_raw:
        # true depth value from nyu_depth_v2
        depth_true = rawcsv2np(path_depth_true+file_name)
        # depth estimated from dpt
        depth_dpt = dptcsv2np(path_dpt_depth+file_name)
        #numpy shape: (4,1)
        boundary = dptcsv2np(path_crop_boundary+file_name)

        # summary_np(depth_true)
        # summary_np(depth_dpt)

        depth_ul = dptcsv2np(path_crop_depth+file_name.replace('.csv','_ul.csv'))
        depth_ur = dptcsv2np(path_crop_depth+file_name.replace('.csv','_ur.csv'))
        depth_bl = dptcsv2np(path_crop_depth+file_name.replace('.csv','_bl.csv'))
        depth_br = dptcsv2np(path_crop_depth+file_name.replace('.csv','_br.csv'))

        # compare overlapped parts
        over_true = extract_raw(depth_true, boundary)
        over_dpt = extract_raw(depth_dpt, boundary)
        over_crop = extract_cropped([depth_ul, depth_ur, depth_bl, depth_br], boundary)
        
        print("overlapped parts from true value")
        summary_np(over_true)
        print("overlapped parts from output of dpt")
        summary_np(over_dpt)
        print("overlapped parts from cropped image")
        summary_np(over_crop)

        plot3d_contours(over_true, over_dpt, over_crop)

        error_dpt = over_dpt-over_true
        error_crop = over_crop-over_true

        plot3d_contours(over_true, error_dpt, error_crop)

        mse_dpt = (error_dpt**2).mean()
        mse_crop = (error_crop**2).mean()

        print("MSE for dpt", mse_dpt)
        print("MSE for cropped image", mse_crop)



if __name__=="__main__":
    main()