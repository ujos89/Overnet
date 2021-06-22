import numpy as np
import skimage.io as io
import pandas as pd
from scipy import stats

def crop_depth(depth_np, bounds, path, file_name):
    file_name = file_name.split('.')[0]
    l_limit, r_limit, u_limit, d_limit = bounds

    depth_upperleft = depth_np[:d_limit, :r_limit]
    depth_upperright = depth_np[:d_limit, l_limit:]
    depth_bottomleft = depth_np[u_limit:, :r_limit]
    depth_bottomright = depth_np[u_limit:, l_limit:]

    df_depth_ul = pd.DataFrame(depth_upperleft)
    df_depth_ul.to_csv(path+file_name+'_ul.csv', index=False)
    df_depth_ur = pd.DataFrame(depth_upperright)
    df_depth_ur.to_csv(path+file_name+'_ur.csv', index=False)
    df_depth_bl = pd.DataFrame(depth_bottomleft)
    df_depth_bl.to_csv(path+file_name+'_bl.csv', index=False)
    df_depth_br = pd.DataFrame(depth_bottomright)
    df_depth_br.to_csv(path+file_name+'_br.csv', index=False)

def crop_image(img_np, bounds, path, file_name):
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

def find_obj_bound(npData):
    ## need to modify
    
    l_limit, r_limit, u_limit, d_limit = npData.shape[1], 0, npData.shape[0], 0

    np_flat = npData.flatten()
    np_no0 = np.delete(np_flat, np.where(np_flat==0))
    mode = stats.mode(np_no0)[0]

    positions = np.array(list(zip(*np.where(npData == mode))))
    
    for pos in positions:

        l_limit = min(l_limit, pos[1])
        r_limit = max(r_limit, pos[1])
        u_limit = min(u_limit, pos[0])
        d_limit = max(d_limit, pos[0])
        
    bounds = [l_limit, r_limit, u_limit, d_limit]

    bounds = [200,550,200,400]
    return bounds