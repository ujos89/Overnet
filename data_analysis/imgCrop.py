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

def find_obj(npData, value):
    positons = np.array(list(zip(*np.where(npData == value))))
    
    ## how to find largest portion of labels in npData????
    
    #limits of [left, right, up, down]
    bounds = [10, 500, 40, 320]
    return bounds



def main():
    path_label = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/label/'
    path_depth_img = '/home/ujos89/Desktop/Vision/DPT/output_monodepth/'
    path_title = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/title/'
    label_name = pd.read_csv(path_title+'label_title.csv').reset_index(drop=True).to_numpy().flatten()
    file_names = os.listdir(path_depth_img)
    print(label_name.shape)

    for file_name in file_names:
        if file_name.endswith('.png'):
            file_name_csv = file_name.replace('.png', '.csv')
            npData = rawcsv2np(path_label+file_name_csv)
            print(npData.shape)
            mode = get_mode(npData)
            print(file_name_csv,"( mode:",mode,", label: ", label_name[mode],")")



if __name__=='__main__':
    main()