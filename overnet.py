import numpy as np
import pandas as pd
import argparse
import torch
import os
import cv2
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from preprocessing.preprocessing import *
from utils.crop import *

def dpt(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load network: model_type == "dpt_hybrid_nyu"
    net_w = 640
    net_h = 480
    model = DPTDepthModel(
        path="dpt/weights/dpt_hybrid_nyu-2ce69ec7.pt",
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    
    model.eval()

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # predict depth for single image
    img_names = os.listdir(input_path+"single/")
    num_images = len(img_names)
    
    print("Monocular Depth Prediction (Single Images)")
    for idx, img_name in enumerate(img_names):
        # progress
        print("  processing {} ({}/{})".format(img_name, idx + 1, num_images))
        
        img = img2np(input_path+"single/"+img_name)
        img_input = transform({"image": img})["image"]

        # prediction
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        
        df_pred = pd.DataFrame(prediction)
        df_pred.to_csv(output_path+"single/"+img_name.replace(".png",".csv"), index=False)

    # predict depth for single image
    img_names = os.listdir(input_path+"crop/")
    num_images = len(img_names)
    
    print("Monocular Depth Prediction (Cropped Images)")
    for idx, img_name in enumerate(img_names):
        # progress
        print("  processing {} ({}/{})".format(img_name, idx + 1, num_images))
        
        img = img2np(input_path+"crop/"+img_name)
        img_input = transform({"image": img})["image"]

        # prediction
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        
        df_pred = pd.DataFrame(prediction)
        df_pred.to_csv(output_path+"crop/"+img_name.replace(".png",".csv"), index=False)

def imgcrop(input_path, output_path, true_value_path):
    rawdepth_path = "../dataset/nyu_depth_v2/labeled/depth/"
    bounds_path = output_path+"bounds/"
    single_img_path = input_path+"single/"
    crop_img_path = input_path+"crop/"
    single_true_value_path = true_value_path+"single/"
    crop_true_value_path = true_value_path+"crop/"

    img_names = os.listdir(single_img_path)
    num_images = len(img_names)

    print("Crop Image")
    for idx, img_name in enumerate(img_names):
        # progress
        print("  processing {} ({}/{})".format(img_name, idx + 1, num_images))

        img = img2np(single_img_path+img_name)
        
        # store true value(depth) single images (from raw depth nyu_depth_v2)
        csv_name = img_name.replace('.png', '.csv')
        depth = nyucsv2np(rawdepth_path+csv_name)
        df_depth = pd.DataFrame(depth)
        df_depth.to_csv(single_true_value_path+csv_name, index=False)
        
        # get bounds
        bounds = find_obj_bound(img)
        df_bounds = pd.DataFrame(bounds)
        df_bounds.to_csv(bounds_path+csv_name, index=False)

        # crop image by bounds
        crop_image(img, bounds, crop_img_path, img_name)
        
        # store true value(depth) crop images
        crop_depth(depth, bounds, crop_true_value_path, csv_name)
        

def main(input_path, output_path, true_value_path):
    """
    input_path : input/
    output_path: output/
    true_value_path: true_value/
    """
    
    # crop image & depth
    imgcrop(input_path, output_path, true_value_path)

    # predict depth with dpt
    dpt(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i",
        "--input_path",
        default="input/",
        help="path with input images"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output/",
        help="path with output images"
    )
    parser.add_argument(
        "-t",
        "--true_value_path",
        default="true_value/",
        help="path with true value(depth) csv files"
    )
    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main(args.input_path, args.output_path, args.true_value_path)