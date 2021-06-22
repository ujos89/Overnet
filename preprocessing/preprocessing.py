import numpy as np
import pandas as pd
import skimage.io as io

def nyucsv2np(file):
    npData = pd.read_csv(file).to_numpy()
    npData = np.rot90(npData)
    npData = npData[::-1]

    return npData

def img2np(file):
    return io.imread(file)