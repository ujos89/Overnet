import numpy as np
import pandas as pd
import os
from scipy import stats
#Crop image to 4 piece

def rawcsv2np(file_name):
    npData = pd.read_csv(file_name).to_numpy()
    npData = np.rot90(npData)
    npData = npData[::-1]

    return npData

def dptcsv2np(file_name):
    return pd.read_csv(file_name).to_numpy()

def get_mode(npData):
    np_flat = npData.flatten()
    np_no0 = np.delete(np_flat, np.where(np_flat==0))
    print(stats.mode(np_no0))
    mode = stats.mode(np_no0)[0]
    return mode

def find_obj_bound(npData, value):
    positons = np.array(list(zip(*np.where(npData == value))))
    
    ## how to find largest portion of labels in npData????
    
    #limits of [left, right, up, down]
    bounds = [10, 500, 40, 320]
    return bounds

def imgCrop_depth(img_np, bounds):
    l_limit, r_limit, u_limit, d_limit = bounds

    img_upperleft = img_np[:d_limit+1, :r_limit+1]
    img_upperright = img_np[:d_limit+1, l_limit:]
    img_bottomleft = img_np[u_limit:, :r_limit+1]
    img_bottomright = img_np[u_limit:, l_limit:]

    return img_upperleft, img_upperright, img_bottomleft, img_bottomright

def imgCrop_rgb(img, bounds):
    pass


def main():
    path_label = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/label/'
    path_depth_img = '/home/ujos89/Desktop/Vision/DPT/output_monodepth/'
    path_title = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/title/'
    path_dpt_pfm = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/dpt/dpt_pfm/'
    label_name = pd.read_csv(path_title+'label_title.csv').reset_index(drop=True).to_numpy().flatten()
    file_names = os.listdir(path_depth_img)

    for file_name in file_names:
        if file_name.endswith('.png'):
            file_name_csv = file_name.replace('.png', '.csv')
            npData_label = rawcsv2np(path_label+file_name_csv)
            print()
            mode = get_mode(npData_label)
            print(file_name_csv,"( mode:",mode,", label: ", label_name[mode],")")

            #Crop img example
            npData_dpt_pfm = dptcsv2np(path_dpt_pfm+file_name_csv)
            bounds = find_obj_bound(npData_dpt_pfm, 0)
            npData_dpt_pfm_ul, npData_dpt_pfm_ur, npData_dpt_pfm_bl, npData_dpt_pfm_br = imgCrop_depth(npData_dpt_pfm, bounds)

            print(npData_dpt_pfm_ul.shape)
            print(npData_dpt_pfm_ur.shape)
            print(npData_dpt_pfm_bl.shape)
            print(npData_dpt_pfm_br.shape)



if __name__=='__main__':
    main()