"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import pickle
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

# Create directories if they do not exist
result_folder = './result/'
picklefiles = './pickle'
if not os.path.isdir(picklefiles):
    os.mkdir(picklefiles)

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # Resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # Coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # Render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

# if __name__ == '__main__':
#     # Load net
#     net = CRAFT()  # Initialize

#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     if args.cuda:
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
#     else:
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

#     if args.cuda:
#         net = net.cuda()
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False

#     net.eval()

#     # Link Refiner
#     refine_net = None
#     if args.refine:
#         from refinenet import RefineNet
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
#         if args.cuda:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
#             refine_net = refine_net.cuda()
#             refine_net = torch.nn.DataParallel(refine_net)
#         else:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

#         refine_net.eval()
#         args.poly = True

#     t = time.time()

#     # Load data
#     box_data = []
#     image_list, _, _ = file_utils.get_files(args.test_folder)
    
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        
#         try:
#             image = imgproc.loadImage(image_path)
#         except (OSError, ValueError) as e:
#             print(f"Error loading image {image_path}: {e}")
#             continue

#         bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        
#         file_name = os.path.basename(image_path)
#         box_data.append({
#             'file_name': file_name,
#             'bboxes': bboxes
#         })

#         # Save results
#         file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

#     # Save bounding boxes to pickle file after processing all images
#     with open(picklefiles + '/bbox_list.pkl', 'wb') as f:
#         pickle.dump(box_data, f)

#     print("Elapsed time : {}s".format(time.time() - t))
if __name__ == '__main__':
    # Load net
    net = CRAFT()  # Initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # Link Refiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # Load data
    box_data = []
    image_list, _, _ = file_utils.get_files(args.test_folder)

    # List of filenames to skip
    skip_files = ['410+-FsVe-L.jpg', '311PssCBQIL.jpg','311e8av9T8L.jpg','51q6DNtF9sL.jpg','41-UpuYc8mL.jpg','41--Lur7UYL.jpg','41BXCQpIpxL.jpg']
    
    for k, image_path in enumerate(image_list):
        file_name = os.path.basename(image_path)
        
        # Check if the filename is in the skip list
        if any(file_name.endswith(f) for f in skip_files):
            print(f"Skipping file: {file_name}")
            continue
        
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        
        try:
            image = imgproc.loadImage(image_path)
        except (OSError, ValueError) as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        
        box_data.append({
            'file_name': file_name,
            'bboxes': bboxes
        })

        # Save results
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    # Save bounding boxes to pickle file after processing all images
    with open(picklefiles + '/bbox_list.pkl', 'wb') as f:
        pickle.dump(box_data, f)

    print("Elapsed time : {}s".format(time.time() - t))

