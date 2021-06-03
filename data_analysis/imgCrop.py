import numpy as np
import pandas as pd
import os
import skimage.io as io
from scipy import stats
#Crop image to 4 piece

def rawcsv2np(file_name):
    npData = pd.read_csv(file_name).to_numpy()
    npData = np.rot90(npData)
    npData = npData[::-1]

    return npData

def dptcsv2np(file_name):
    return pd.read_csv(file_name).to_numpy()

def png2np(file_name):
    return io.imread(file_name)

def get_mode(npData):
    np_flat = npData.flatten()
    np_no0 = np.delete(np_flat, np.where(np_flat==0))
    print(stats.mode(np_no0))
    mode = stats.mode(np_no0)[0]
    return mode

def find_obj_bound(npData, value):
    l_limit, r_limit, u_limit, d_limit = npData.shape[1], 0, npData.shape[0], 0
    positions = np.array(list(zip(*np.where(npData == value))))
    
    for pos in positions:

        l_limit = min(l_limit, pos[1])
        r_limit = max(r_limit, pos[1])
        u_limit = min(u_limit, pos[0])
        d_limit = max(d_limit, pos[0])
        
    bounds = [l_limit, r_limit, u_limit, d_limit]

    bounds = [200,500,200,400]
    return bounds

def imgCrop_depth(img_np, bounds):
    l_limit, r_limit, u_limit, d_limit = bounds

    img_upperleft = img_np[:d_limit, :r_limit]
    img_upperright = img_np[:d_limit, l_limit:]
    img_bottomleft = img_np[u_limit:, :r_limit]
    img_bottomright = img_np[u_limit:, l_limit:]

    return img_upperleft, img_upperright, img_bottomleft, img_bottomright

def imgCrop_rgb(img_np, bounds, path, file_name):
    #input shape: 480x640x3
    file_name = file_name.split('.')[0]
    l_limit, r_limit, u_limit, d_limit = bounds
    
    img_upperleft = img_np[:d_limit, :r_limit, :]
    img_upperright = img_np[:d_limit, l_limit:, :]
    img_bottomleft = img_np[u_limit:, :r_limit, :]
    img_bottomright = img_np[u_limit:, l_limit:, :]    
    
    io.imsave(path+file_name+'_ul.png', img_upperleft)
    io.imsave(path+file_name+'_ur.png', img_upperright)
    io.imsave(path+file_name+'_bl.png', img_bottomleft)
    io.imsave(path+file_name+'_br.png', img_bottomright)


def main():
    path_label = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/label/'
    path_dpt_input = '../dpt/image/'
    path_dpt_output = '../dpt/depth/'
    path_title = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/title/'
    path_depth_true = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/dpt/labeled/depth/'
    path_img = '/home/ujos89/Desktop/Vision/dataset/nyu_depth_v2/labeled/image/'
    path_crop_img ='../cropped_img/image/'
    path_crop_depth ='../cropped_img/depth/'
    path_crop_boundary = '../cropped_img/boundary/'

    label_name = pd.read_csv(path_title+'label_title.csv').reset_index(drop=True).to_numpy().flatten()
    file_names = os.listdir(path_dpt_input)


    for file_name in file_names:
        if file_name.endswith('.png'):
            file_name_png = file_name
            file_name_csv = file_name.replace('.png', '.csv')
            
            #get boundary for crop
            np_label = rawcsv2np(path_label+file_name_csv)
            mode = get_mode(np_label)
            print(file_name_csv,"( mode:",mode,", label: ", label_name[mode],")")
            bounds = find_obj_bound(np_label, mode)
            print("bounds:", bounds)
            df_bounds = pd.DataFrame(bounds)
            df_bounds.to_csv(path_crop_boundary+file_name_csv, index=False)

            #Crop img
            np_rawimg = png2np(path_img+file_name_png)
            imgCrop_rgb(np_rawimg, bounds, path_crop_img, file_name_png)
            print("save cropped image :", file_name_png)
            

if __name__=='__main__':
    main()