import torch
from torch.optim import Adam, SGD
import time
import asyncio
import copy
import json

import numpy as np

from data.data import *
from data_al.data_al_evanet import get_dataset_al
from data_al.data_remaker_al import remake_data
from eva_net_model import *
from loss import *
from metrics import *

from tqdm import tqdm
# from osgeo import gdal

import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch
import itertools
import time
from collections import defaultdict
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# these variables should just be updated in each iteration
labeled_superpixels = {}

def get_meta_data(DATASET_PATH):
    
    DATASET = os.listdir(DATASET_PATH)
    DATASET = [file for file in DATASET if  file.endswith(".npy") and re.search("Features", file)]

    META_DATA = dict()

    for file_name in DATASET:
        file = np.load(os.path.join(DATASET_PATH, file_name))
        #print(file.shape)
        file_height, file_width, _ = file.shape
        #print(file_height)
        #print(file_width)

        elev_data = file[:, :, 3]
        file_elev_max = np.max(elev_data)
        file_elev_min = np.min(elev_data)
        # print(file_elev_max)
        # print(file_elev_min)

        if file_elev_max>config.GLOBAL_MAX:
            config.GLOBAL_MAX = file_elev_max
        if file_elev_min<config.GLOBAL_MIN:
            config.GLOBAL_MIN = file_elev_min


        META_DATA[file_name] = {"height": file_height,
                                "width": file_width}
        
    return META_DATA

def run_pred_al_probability(model, data_loader, TRANSFORMATION_SCORE):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    final_pred_patches_dict = dict()
    variance_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        logits_, pred = model(rgb_data, norm_elev_data)
        pred = torch.nn.functional.softmax(pred, dim = 1)

        # print('PRED SHAPE: ', pred.shape) # 4, 2, 128, 128

        rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
        rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
        rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
        rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
        rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)
        
        norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
        norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
        norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
        norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
        norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

        logits_flipx, pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
        logits_flipy, pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
        logits_rot90, pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
        logits_rot180, pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
        logits_ro7270, pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip
        
        pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
        pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
        pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
        pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
        pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

        pred_flipx_inv = torch.nn.functional.softmax(pred_flipx_inv, dim = 1)
        pred_flipy_inv = torch.nn.functional.softmax(pred_flipy_inv, dim = 1)
        pred_rot90_inv = torch.nn.functional.softmax(pred_rot90_inv, dim = 1)
        pred_rot180_inv = torch.nn.functional.softmax(pred_rot180_inv, dim = 1)
        pred_rot270_inv = torch.nn.functional.softmax(pred_rot270_inv, dim = 1)

        if config.ENT_VAR:
            # entropy variance
            all_logits = [pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]
            variance = compute_entropy_variance(all_logits, TRANSFORMATION_SCORE)
        else:
            # offset variance
            s1,s2,s3,s4 = pred.shape
            half_array = torch.tensor(np.full((s1, s2, s3, s4), 0.5)).to(DEVICE)
            
            # pred_abs = torch.abs(pred - half_array)
            # pred_abs = torch.sum(pred_abs, dim=1)/2
            # pred_flipx_abs = torch.abs(pred_flipx_inv - half_array)
            # pred_flipx_abs = torch.sum(pred_flipx_abs, dim=1)/2
            # pred_flipy_abs = torch.abs(pred_flipy_inv - half_array)
            # pred_flipy_abs = torch.sum(pred_flipy_abs, dim=1)/2
            # pred_rot90_abs = torch.abs(pred_rot90_inv - half_array)
            # pred_rot90_abs = torch.sum(pred_rot90_abs, dim=1)/2
            # pred_rot180_abs = torch.abs(pred_rot180_inv - half_array)
            # pred_rot180_abs = torch.sum(pred_rot180_abs, dim=1)/2
            # pred_rot270_abs = torch.abs(pred_rot270_inv - half_array)
            # pred_rot270_abs = torch.sum(pred_rot270_abs, dim=1)/2

            # avg_pred_abs = (pred_abs + pred_flipx_abs + pred_flipy_abs + pred_rot90_abs + pred_rot180_abs + pred_rot270_abs)/6

            # # compute variance of each patch as compared to average logits
            # var_orig = torch.pow(torch.abs(avg_pred_abs - pred_abs), 2)
            # var_flipx = torch.pow(torch.abs(avg_pred_abs - pred_flipx_abs), 2)
            # var_flipy = torch.pow(torch.abs(avg_pred_abs - pred_flipy_abs), 2)
            # var_rot90 = torch.pow(torch.abs(avg_pred_abs - pred_rot90_abs), 2)
            # var_rot180 = torch.pow(torch.abs(avg_pred_abs - pred_rot180_abs), 2)
            # var_rot270 = torch.pow(torch.abs(avg_pred_abs - pred_rot270_abs), 2)

            avg_pred = (pred[:,0,:,:] + pred_flipx_inv[:,0,:,:] + pred_flipy_inv[:,0,:,:] + pred_rot90_inv[:,0,:,:] + pred_rot180_inv[:,0,:,:] + pred_rot270_inv[:,0,:,:])/6

            # compute variance of each patch as compared to average logits
            var_orig = torch.pow(torch.abs(avg_pred - pred[:,0,:,:]), 2)
            var_flipx = torch.pow(torch.abs(avg_pred - pred_flipx_inv[:,0,:,:]), 2)
            var_flipy = torch.pow(torch.abs(avg_pred - pred_flipy_inv[:,0,:,:]), 2)
            var_rot90 = torch.pow(torch.abs(avg_pred - pred_rot90_inv[:,0,:,:]), 2)
            var_rot180 = torch.pow(torch.abs(avg_pred - pred_rot180_inv[:,0,:,:]), 2)
            var_rot270 = torch.pow(torch.abs(avg_pred - pred_rot270_inv[:,0,:,:]), 2)

            # aggregate the variance among 6 transformations
            if TRANSFORMATION_SCORE == "AVG":
                variance = torch.sum(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0) / 6
            elif TRANSFORMATION_SCORE == "MIN":
                variance = torch.min(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values
            elif TRANSFORMATION_SCORE == "MAX":
                variance = torch.max(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values
            
        pred_np = pred.detach().cpu().numpy()
        variance_np = variance.detach().cpu().numpy()

        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_np[idx, :, :, :]
            variance_patches_dict[filename[idx]] = variance_np[idx, :, :]
        
    return pred_patches_dict, variance_patches_dict

def run_pred_al_entropy(model, data_loader, TRANSFORMATION_SCORE):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    entropy_patches_dict = dict()
    variance_patches_dict = dict()
    
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        logits_, pred = model(rgb_data, norm_elev_data)
        pred = torch.nn.functional.softmax(pred, dim = 1)

        rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
        rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
        rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
        rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
        rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)
        
        norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
        norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
        norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
        norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
        norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

        logits_flipx, pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
        logits_flipy, pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
        logits_rot90, pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
        logits_rot180, pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
        logits_ro7270, pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip
        
        pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
        pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
        pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
        pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
        pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

        pred_flipx_inv = torch.nn.functional.softmax(pred_flipx_inv, dim = 1)
        pred_flipy_inv = torch.nn.functional.softmax(pred_flipy_inv, dim = 1)
        pred_rot90_inv = torch.nn.functional.softmax(pred_rot90_inv, dim = 1)
        pred_rot180_inv = torch.nn.functional.softmax(pred_rot180_inv, dim = 1)
        pred_rot270_inv = torch.nn.functional.softmax(pred_rot270_inv, dim = 1)

        if config.ENT_VAR:
            # entropy variance
            all_logits = [pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]
            variance = compute_entropy_variance(all_logits, TRANSFORMATION_SCORE)
        else:
            # prob variance
            s1,s2,s3,s4 = pred.shape
            half_array = torch.tensor(np.full((s1, s2, s3, s4), 0.5)).to(DEVICE)
            
            # pred_abs = torch.abs(pred - half_array)
            # pred_abs = torch.sum(pred_abs, dim=1)/2
            # pred_flipx_abs = torch.abs(pred_flipx_inv - half_array)
            # pred_flipx_abs = torch.sum(pred_flipx_abs, dim=1)/2
            # pred_flipy_abs = torch.abs(pred_flipy_inv - half_array)
            # pred_flipy_abs = torch.sum(pred_flipy_abs, dim=1)/2
            # pred_rot90_abs = torch.abs(pred_rot90_inv - half_array)
            # pred_rot90_abs = torch.sum(pred_rot90_abs, dim=1)/2
            # pred_rot180_abs = torch.abs(pred_rot180_inv - half_array)
            # pred_rot180_abs = torch.sum(pred_rot180_abs, dim=1)/2
            # pred_rot270_abs = torch.abs(pred_rot270_inv - half_array)
            # pred_rot270_abs = torch.sum(pred_rot270_abs, dim=1)/2

            # avg_pred_abs = (pred_abs + pred_flipx_abs + pred_flipy_abs + pred_rot90_abs + pred_rot180_abs + pred_rot270_abs)/6

            # # compute variance of each patch as compared to average logits
            # var_orig = torch.pow(torch.abs(avg_pred_abs - pred_abs), 2)
            # var_flipx = torch.pow(torch.abs(avg_pred_abs - pred_flipx_abs), 2)
            # var_flipy = torch.pow(torch.abs(avg_pred_abs - pred_flipy_abs), 2)
            # var_rot90 = torch.pow(torch.abs(avg_pred_abs - pred_rot90_abs), 2)
            # var_rot180 = torch.pow(torch.abs(avg_pred_abs - pred_rot180_abs), 2)
            # var_rot270 = torch.pow(torch.abs(avg_pred_abs - pred_rot270_abs), 2)

            avg_pred = (pred[:,0,:,:] + pred_flipx_inv[:,0,:,:] + pred_flipy_inv[:,0,:,:] + pred_rot90_inv[:,0,:,:] + pred_rot180_inv[:,0,:,:] + pred_rot270_inv[:,0,:,:])/6

            # compute variance of each patch as compared to average logits
            var_orig = torch.pow(torch.abs(avg_pred - pred[:,0,:,:]), 2)
            var_flipx = torch.pow(torch.abs(avg_pred - pred_flipx_inv[:,0,:,:]), 2)
            var_flipy = torch.pow(torch.abs(avg_pred - pred_flipy_inv[:,0,:,:]), 2)
            var_rot90 = torch.pow(torch.abs(avg_pred - pred_rot90_inv[:,0,:,:]), 2)
            var_rot180 = torch.pow(torch.abs(avg_pred - pred_rot180_inv[:,0,:,:]), 2)
            var_rot270 = torch.pow(torch.abs(avg_pred - pred_rot270_inv[:,0,:,:]), 2)

            # aggregate the variance among 5 transformations
            if TRANSFORMATION_SCORE == "AVG":
                variance = torch.sum(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0) / 6
            elif TRANSFORMATION_SCORE == "MIN":
                variance = torch.min(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values
            elif TRANSFORMATION_SCORE == "MAX":
                variance = torch.max(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values

        pred_np = pred.detach().cpu().numpy()
        variance_np = variance.detach().cpu().numpy()
 
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_np[idx, :, :, :]
            variance_patches_dict[filename[idx]] = variance_np[idx, :, :]

    return pred_patches_dict, variance_patches_dict

def run_pred_al_cod(models, data_loader):
    
    ## Model gets set to evaluation mode
    pred_patches_dict = dict()
    cod_loss_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        logits_backbone, pred_backbone = models['backbone'](rgb_data, norm_elev_data)
        logits_cod, pred_cod = models['cod'](rgb_data, norm_elev_data)

        pred_backbone = torch.nn.functional.softmax(pred_backbone, dim = 1)
        pred_cod = torch.nn.functional.softmax(pred_cod, dim = 1)

        # print("logits shape: ", logits_backbone.shape)

        if config.USE_LOGITS:
            cod_loss = (logits_backbone - logits_cod).pow(2)
        else:
            cod_loss = (pred_backbone - pred_cod).pow(2)
        
        cod_loss = torch.sum(cod_loss, dim=1)/2

        cod_loss_np = cod_loss.detach().cpu().numpy()

        pred_np = pred_backbone.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            cod_loss_patches_dict[filename[idx]] = cod_loss_np[idx, :, :]
            pred_patches_dict[filename[idx]] = pred_np[idx, :, :, :]

    return pred_patches_dict, cod_loss_patches_dict


# def compute_entropy(outputs, TRANSFORMATION_SCORE):

#     counter = 0
#     entropy_list = []
#     for i, output in enumerate(outputs):
#         # prob_out = torch.nn.functional.softmax(output, dim = 1)
#         prob_out = torch.clone(output)

#         log_out =-1* torch.log(prob_out)
#         log_out[log_out != log_out] = 0
#         log_out[log_out == float("Inf")] = 0
#         log_out[log_out == -float("Inf")] = 0
#         log_out[log_out == float("-Inf")] = 0

#         entropy_computed = log_out * prob_out

#         # print("entropy_computed_shape: ", entropy_computed.shape)
#         # print(torch.min(entropy_computed, dim=1).values, torch.max(entropy_computed, dim=1).values)
#         # entropy_map = torch.sum(entropy_computed, dim=1) # summation over c in paper

#         if TRANSFORMATION_SCORE != "AVG":
#             entropy_computed = entropy_computed.unsqueeze(dim = 1)
#         entropy_list.append(entropy_computed)

#         if i == 0:
#             numpy_ent_total = torch.clone(entropy_computed)
#             # stacked_entropy = entropy_computed
#         else:
#             numpy_ent_total = numpy_ent_total +  torch.clone(entropy_computed)
#             # stacked_entropy = torch.stack([stacked_entropy, entropy_computed])
        
#         counter += 1

#     stacked_entropy = torch.cat(entropy_list, dim=1)

#     if TRANSFORMATION_SCORE == "AVG":
#         return (numpy_ent_total / counter)
#     elif TRANSFORMATION_SCORE == "MIN":
#         return torch.min(stacked_entropy, dim=1).values
#     elif TRANSFORMATION_SCORE == "MAX":
#         return torch.max(stacked_entropy, dim=1).values
   
    # return numpy_ent_total 


def compute_entropy_variance(outputs, TRANSFORMATION_SCORE):

    counter = 0
    entropy_list = []
    for i, output in enumerate(outputs):
        # prob_out = torch.nn.functional.softmax(output, dim = 1)
        prob_out = torch.clone(output)

        log_out =-1* torch.log(prob_out)
        log_out[log_out != log_out] = 0
        log_out[log_out == float("Inf")] = 0
        log_out[log_out == -float("Inf")] = 0
        log_out[log_out == float("-Inf")] = 0

        entropy_computed = log_out * prob_out
        entropy_computed = torch.sum(entropy_computed, dim=1)
        entropy_list.append(entropy_computed)

    avg_entropy = (entropy_list[0] + entropy_list[1] + entropy_list[2] + entropy_list[3] + entropy_list[4] + entropy_list[5])/6

    # compute variance of each patch as compared to average logits
    var_orig = torch.pow(torch.abs(avg_entropy - entropy_list[0]), 2)
    var_flipx = torch.pow(torch.abs(avg_entropy - entropy_list[1]), 2)
    var_flipy = torch.pow(torch.abs(avg_entropy - entropy_list[2]), 2)
    var_rot90 = torch.pow(torch.abs(avg_entropy - entropy_list[3]), 2)
    var_rot180 = torch.pow(torch.abs(avg_entropy - entropy_list[4]), 2)
    var_rot270 = torch.pow(torch.abs(avg_entropy - entropy_list[5]), 2)

    # aggregate the variance among 6 transformations
    if TRANSFORMATION_SCORE == "AVG":
        entropy_variance = torch.sum(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0) / 6
    elif TRANSFORMATION_SCORE == "MIN":
        entropy_variance = torch.min(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values
    elif TRANSFORMATION_SCORE == "MAX":
        entropy_variance = torch.max(torch.stack([var_orig, var_flipx, var_flipy, var_rot90, var_rot180, var_rot270]), dim=0).values
    
    return entropy_variance
        

def run_pred_final(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        logits_, pred = model(rgb_data, norm_elev_data)

        # ## Remove pred and GT from GPU and convert to np array
        pred_labels_np = pred.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = pred_labels_np[idx, :, :, :]
        
    return pred_patches_dict 


def run_pred_final_avg(model, data_loader):
    
    ## Model gets set to evaluation mode
    model.eval()
    pred_patches_dict = dict()
    
    for data_dict in tqdm(data_loader):
        
        ## RGB data
        rgb_data = data_dict['rgb_data'].float().to(DEVICE)

        ## Elevation data
        norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)

        ## Get filename
        filename = data_dict['filename']

        ## Get model prediction
        logits_, pred = model(rgb_data, norm_elev_data)

        # flip and rotate
        rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
        rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
        rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
        rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
        rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)
        
        norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
        norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
        norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
        norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
        norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

        logits_flipx, pred_flipx = model(rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
        logits_flipy, pred_flipy = model(rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
        logits_rot90, pred_rot90 = model(rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
        logits_rot180, pred_rot180 = model(rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
        logits_ro7270, pred_rot270 = model(rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip
        
        pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
        pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
        pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
        pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
        pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

#         s1,s2,s3,s4 = pred.shape
#         half_array = torch.tensor(np.full((s1, s2, s3, s4), 0.5)).to(DEVICE)
        
#         pred_abs = torch.abs(pred - half_array)
#         pred_flipx_abs = torch.abs(pred_flipx_inv - half_array)
#         pred_flipy_abs = torch.abs(pred_flipy_inv - half_array)
#         pred_rot90_abs = torch.abs(pred_rot90_inv - half_array)
#         pred_rot180_abs = torch.abs(pred_rot180_inv - half_array)
#         pred_rot270_abs = torch.abs(pred_rot270_inv - half_array)

#         avg_prob = torch.sum(torch.stack([pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]), dim=0) / 6
        avg_prob = torch.sum(torch.stack([pred, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]), dim=0) / 6
        avg_prob_np = avg_prob.detach().cpu().numpy()

#         avg_prob_np = pred.detach().cpu().numpy()

        # ## Remove pred and GT from GPU and convert to np array
        # pred_labels_np = pred.detach().cpu().numpy() 
        # gt_labels_np = labels.detach().cpu().numpy()
        
        ## Save Image and RGB patch
        for idx in range(rgb_data.shape[0]):
            pred_patches_dict[filename[idx]] = avg_prob_np[idx, :, :, :]
        
    return pred_patches_dict 


def find_patch_meta(pred_patches_dict):
    y_max = 0
    x_max = 0

    for item in pred_patches_dict:

        temp = int(item.split("_")[3])
        if temp>y_max:
            y_max = temp

        temp = int(item.split("_")[5])
        if temp>x_max:
            x_max = temp


    y_max+=1
    x_max+=1
    
    return y_max, x_max


def stitch_patches_GT_labels(pred_patches_dict, TEST_REGION):
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            dict_key_label = f"Region_{TEST_REGION}_y_{i}_x_{j}_label.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            pred_patch = np.transpose(pred_patch, (1, 2, 0))

            label_patch = np.load(os.path.join(cropped_data_path, dict_key_label))

            if j == 0:
                label_x_patches = label_patch
                pred_x_patches = pred_patch
            else:
                label_x_patches = np.concatenate((label_x_patches, label_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            label_y_patches = label_x_patches
            pred_y_patches = pred_x_patches
        else:
            label_y_patches = np.vstack((label_y_patches, label_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    label_stitched = label_y_patches
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return label_stitched, pred_stitched

def stitch_patches(pred_patches_dict, TEST_REGION, var=False):
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    y_max, x_max = find_patch_meta(pred_patches_dict)
    
    for i in range(y_max):
        for j in range(x_max):
            dict_key = f"Region_{TEST_REGION}_y_{i}_x_{j}_features.npy"
            #print(dict_key)
        
            pred_patch = pred_patches_dict[dict_key]
            if not var:
                pred_patch = np.transpose(pred_patch, (1, 2, 0))

            rgb_patch = np.load(os.path.join(cropped_data_path, dict_key))[:, :, :3]


            if j == 0:
                rgb_x_patches = rgb_patch
                pred_x_patches = pred_patch
            else:
                rgb_x_patches = np.concatenate((rgb_x_patches, rgb_patch), axis = 1)
                pred_x_patches = np.concatenate((pred_x_patches, pred_patch), axis = 1)

            ## rgb_patches.append(rgb_patch)
            ## pred_patches.append(pred_patch)
    
        if i == 0:
            rgb_y_patches = rgb_x_patches
            pred_y_patches = pred_x_patches
        else:
            rgb_y_patches = np.vstack((rgb_y_patches, rgb_x_patches))
            pred_y_patches = np.vstack((pred_y_patches, pred_x_patches))
        

    rgb_stitched = rgb_y_patches.astype('uint8')
#     pred_stitched = np.argmax(pred_y_patches, axis = -1)
    pred_stitched = pred_y_patches.copy()
    
    return rgb_stitched, pred_stitched

def center_crop(stictched_data, original_height, original_width, image = False, var=False):
    # dict_key = f"Region_{TEST_REGION}_Features7Channel.npy"
    
#     if image:
    current_height, current_width = stictched_data.shape[0], stictched_data.shape[1]
#     else:
#         current_height, current_width = stictched_data.shape    
#     print("current_height: ", current_height)
#     print("current_width: ", current_width)
    
    # original_height = META_DATA[dict_key]['height']
    # original_width = META_DATA[dict_key]['width']
#     print("original_height: ", original_height)
#     print("original_width: ", original_width)
    
    height_diff = current_height-original_height
    width_diff = current_width-original_width
    
#     print("height_diff: ", height_diff)
#     print("width_diff: ", width_diff)
    
    if var:
        cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2]
    else:
        cropped = stictched_data[height_diff//2:current_height-height_diff//2, width_diff//2: current_width-width_diff//2, :]
    
    return cropped

def get_superpixel_scores(superpixels_group, logits, forest_prob, SUPERPIXEL_SCORE, method, use_forest=1):
    superpixel_scores = {}
    h,w = forest_prob.shape
    for sid, pixels in superpixels_group.items():
        total_score = 0
        total_pixels = len(pixels)
        min_score = 1e300
        max_score = -1e300

        for (row, col) in pixels:
            prob_score = logits[row][col]
            forest_score = forest_prob[row][col]

            # Vote down the superpixel based on forest model probability score
            # if config.PROBABILITY and config.TAKE_PRODUCT:
            #     prob_score *= forest_score
            # elif config.PROBABILITY:
            #     prob_score += forest_score
            # elif config.ENTROPY and config.TAKE_PRODUCT:
            #     prob_score *= (-math.log(forest_score))
            # elif config.ENTROPY:
            #     prob_score += forest_score
            # elif config.COD:
            #     prob_score += forest_score

            # if config.TAKE_PRODUCT:
            #     prob_score *= forest_score
            # else:

            if use_forest:
                prob_score = prob_score + config.LAMBDA_3 * forest_score

            # total_score += prob_score
            # if prob_score < min_score:
            #     min_score = prob_score
            
            # if prob_score > max_score:
            #     max_score = prob_score

            pixel_idx = row * w + col
            superpixel_scores[pixel_idx] = prob_score

        # if SUPERPIXEL_SCORE == "AVG":
        #     superpixel_scores[sid] = total_score / total_pixels
        # elif SUPERPIXEL_SCORE == "MIN":
        #     superpixel_scores[sid] = min_score
        # elif SUPERPIXEL_SCORE == "MAX":
        #     superpixel_scores[sid] = max_score
    
    return superpixel_scores


def get_superpixel_scores_min(superpixels_group, logits):
    superpixel_scores = {}
    for sid, pixels in superpixels_group.items():
        total_score = 0
        total_pixels = len(pixels)
        min_score = 100
        for (row, col) in pixels:
            prob_score = logits[row][col]
            if prob_score < min_score:
                min_score = prob_score
                
        superpixel_scores[sid] = min_score
    
    return superpixel_scores


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def ema_loss(backbone_scores, ema_scores):
    consistency_loss = F.mse_loss(backbone_scores, ema_scores)
    return consistency_loss


def loss_self_consistency(logits: list, labels):
    if not logits:
        return 0

    total_sum = 0
    counter = 0

    # Generate Pred Masks
    ones = torch.ones_like(labels)
    unknown_mask = torch.where(labels == 0, ones, 0) # only get the unlabeled pixels
    unknown_pixels_count = int(torch.sum(unknown_mask))
    
    avg_logits = (logits[0] + logits[1] + logits[2] + logits[3] + logits[4] + logits[5])/6
    
    original_logit = logits[0]
    # for transformed_logit in logits[1:]:
    for transformed_logit in logits:
        # l1, l2 = original_logit, transformed_logit
        l1, l2 = avg_logits[:,0,:,:], transformed_logit[:,0,:,:] # only consider flood class
        diff = torch.subtract(l1, l2)
        diff = unknown_mask * diff
        sq_l2_norm = torch.pow(torch.abs(diff), 2)
        sq_l2_norm = torch.sum(sq_l2_norm).item()
        total_sum += sq_l2_norm
        counter += 1

    _, C, W, H = logits[0].shape
    # loss = (total_sum)/(W * H * C * counter)
    # loss = (total_sum)/(unknown_pixels_count * C * counter)
    loss = (total_sum)/(unknown_pixels_count * counter)
    return loss

def loss_self_consistency_acquisition(logits: list):
    if not logits:
        return 0

    counter = 0    
    avg_logits = (logits[0] + logits[1] + logits[2] + logits[3] + logits[4] + logits[5])/6
    # avg_logits = (logits[0] + logits[1]) / 2
    
    original_logit = logits[0]

    s1, s2, s3, s4 = logits[0].shape
    # sum_array = torch.tensor(np.full((s1, s2, s3, s4), 0)).to(DEVICE)

    for i, transformed_logit in enumerate(logits):
        # l1, l2 = original_logit, transformed_logit
        l1, l2 = avg_logits, transformed_logit
        diff = torch.subtract(l1, l2)
        if i == 0:
            sum_array = diff
        else:
            sum_array += diff
        counter += 1

    loss_array = sum_array / counter
    return loss_array

def label_acquisition(selected_superpixels, elev_data, gt_labels, current_labels, superpixels_group):
    """
        Acquire labels of current_pixel as well as its BFS neighbors 

        Parameters:
            selected_pixels: 2D array where selected pixel = 1, others are 0
            elev_data: DEM of all pixels
            gt_labels: GT labels from our annotation
            current_labels: labels acquired till now; this needs to be updated and returned back                                    
    """

    updated_labels = current_labels.copy() # TODO: initially current labels should all be 0 (unknown)
    
    for sid in selected_superpixels:
        for (row, col) in superpixels_group[sid]:
            updated_labels[row][col] = gt_labels[row][col]
    
    return updated_labels

def select_superpixels(total_superpixels, superpixel_scores, forest_superpixels, annotated_superpixels):
    max_items = min(config.NUM_RECOMMEND, total_superpixels)
    select_count = 0
    selected_superpixels = []
    for i, (sid, prob_score) in enumerate(superpixel_scores.items()):
        is_labeled = labeled_superpixels.get(sid, False)
        is_forest = forest_superpixels.get(sid, False)
        is_annotated = annotated_superpixels.get(sid, False)
        # if not is_labeled and not is_forest and not is_annotated:
        if not is_forest and not is_annotated:
            if select_count < max_items:
                selected_superpixels.append(sid)
                labeled_superpixels[sid] = True
                select_count += 1
    
    return selected_superpixels, select_count

class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after
	certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is
			   not improving
		:param min_delta: minimum difference between new loss and old loss for
			   new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			# reset counter if validation loss improves
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			# logging.info(f"Early stopping counter {self.counter} of {self.patience}")
			if self.counter >= self.patience:
				# logging.info('Early stop!')
				self.early_stop = True

def convert_to_rgb(input_array):
    # Ensure the input array is in the range [0, 1]
    input_array = np.clip(input_array, 0, 1)

    # Create a colormap from blue to red
    cmap = plt.get_cmap('Reds')

    # Apply the colormap to the input array
    rgb_image = cmap(input_array)

    # Remove the alpha channel if it exists
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[:, :, :3]

    return rgb_image


def recommend_superpixels(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, updated_labels=None, use_forest=1):
    student_id = student_id.strip()

    if not os.path.exists(f"./users/{student_id}"):
        os.mkdir(f"./users/{student_id}")

    if not os.path.exists(f"./users/{student_id}/output"):
        os.mkdir(f"./users/{student_id}/output")

    if not os.path.exists(f"./users/{student_id}/output/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/output/Region_{TEST_REGION}_TEST")

    if not os.path.exists(f"./users/{student_id}/saved_models_al"):
        os.mkdir(f"./users/{student_id}/saved_models_al")
    
    if not os.path.exists(f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST")
    
    if not os.path.exists(f"./users/{student_id}/resume_epoch"):
        os.mkdir(f"./users/{student_id}/resume_epoch")

    # print("ent", "prob", "cod")
    # print(entropy, probability, cod)

    # fallback if user choose both to be 0
    if (entropy != 0 or probability != 0):
        print("HERE")
        config.ENTROPY = entropy
        config.PROBABILITY = probability

    if (config.ENTROPY == 0 and config.PROBABILITY == 0 and config.COD == 0):
        config.PROBABILITY = 1

    config.COD = cod

    print("ent", "prob", "cod")
    print(config.ENTROPY, config.PROBABILITY, config.COD)

    if transformation_agg.strip().lower() == 'avg':
        config.TRANSFORMATION_SCORE = 'AVG'
    else:
        config.TRANSFORMATION_SCORE = 'MAX'

    if superpixel_agg.strip().lower() == 'avg':
        config.SUPERPIXEL_SCORE = 'AVG'
    else:
        config.SUPERPIXEL_SCORE = 'MIN'

    # print(config.ENTROPY, config.PROBABILITY)
    print("trans_score", "superpixel_score")
    print(config.TRANSFORMATION_SCORE, config.SUPERPIXEL_SCORE)

    print("use Forest: ", use_forest)

    file_path_lambda_search = f"./users/{student_id}/output/Region_{TEST_REGION}_TEST/L1.{config.LAMBDA_1}_L2.{config.LAMBDA_2}_L3.{config.LAMBDA_3}_B1.{config.BETA_1}_B2.{config.BETA_2}_P.{config.PROBABILITY}_E.{config.ENTROPY}_C.{config.COD}_EVAR.{config.ENT_VAR}_TA.{config.TRANSFORMATION_SCORE}_SA.{config.SUPERPIXEL_SCORE}"
    print(file_path_lambda_search)
    if not os.path.exists(file_path_lambda_search):
        os.mkdir(file_path_lambda_search)


    start = time.time()
    DATASET_PATH = "./data_al/repo/Features_7_Channels"

    superpixels = np.load(f"./data_al/superpixels/Region_{TEST_REGION}/Region_{TEST_REGION}_superpixels.npy") # TODO: make is a global variable or store in cache
    forest_labels = np.load(f"./data_al/forest/R{TEST_REGION}_forest_labels.npy") # TODO: make is a global variable or store in cache
    forest_prob = np.load(f"./data_al/forest/R{TEST_REGION}_forest.npy") # TODO: make is a global variable or store in cache

    # print(forest_labels.shape)

    superpixels_group = defaultdict(list)
    forest_superpixels = {}
    annotated_superpixels = {}

    # Iterate through the NumPy array to group pixels
    height = superpixels.shape[0]
    width = superpixels.shape[1]

    config.HEIGHT = height
    config.WIDTH = width

    updated_labels = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_labels.png', TEST_REGION)
    print("updated_labels: ", updated_labels.shape)

    # print(height, width)
    for i in range(height):
        for j in range(width):
            pixel_value = superpixels[i][j]
            superpixels_group[pixel_value].append((i, j))

    accepted_sub_pixels = np.zeros((height, width), dtype='int')
        
    for sid, pixels in superpixels_group.items():
        forest_count = 0
        annotated_count = 0
        total_pixels = len(pixels)

        accepted_pixels = []
        for (row, col) in pixels:
            pixel_idx = row * width + col
            is_forest = forest_labels[row][col]
            is_annotated = updated_labels[row][col]
            if is_forest:
                # forest_count += 1
                forest_superpixels[pixel_idx] = True
            # else:
                # accepted_pixels.append((row, col))

            if is_annotated != 0:
                annotated_superpixels[pixel_idx] = True
                # annotated_count += 1

        # forest_fraction = forest_count / total_pixels
        # if forest_fraction == 1.0:
        #     forest_superpixels[sid] = True
        # else:
        #     forest_superpixels[sid] = False

        # if forest_fraction >= 0.8 and forest_fraction < 1.0:
            # accepted_pixels_np = np.array(accepted_pixels)
            # accepted_sub_pixels[accepted_pixels_np[:, 0], accepted_pixels_np[:, 1]] = 1

        # if annotated_count == total_pixels:
        #     annotated_superpixels[sid] = True
        # else:
        #     annotated_superpixels[sid] = False


    # print("SUM: ", np.sum(accepted_sub_pixels))

    # read resume epoch from text file if exists
    try:
        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0

    print("Starting from epoch: ", resume_epoch)

    test_filename = f"Region_{TEST_REGION}_Features7Channel.npy"

    VAL_FREQUENCY = 1
    SAVE_FREQUENCY = 1

    elev_data = np.load(f"./data_al/repo/Features_7_Channels/Region_{TEST_REGION}_Features7Channel.npy")[:,:,3]
    gt_labels = np.load(f"./data_al/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")

    total_superpixels = len(superpixels_group.items())

    ######### Pixel Selection using Active Learning #######################
    # model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    cod_model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    backbone_net = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)

    # models = {'original': model, 'backbone': backbone_net, 'cod': cod_model}
    models = {'backbone': backbone_net, 'cod': cod_model}

    # optimizer = SGD(model.parameters(), lr = 1e-7)
    criterion = ElevationLoss()
    elev_eval = Evaluator()


    model_path = f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_{resume_epoch}.ckpt"
    if not os.path.exists(model_path):
        print("Model path doesn't exist; using pretrained model!!!")
        model_path = f"./saved_models_evanet/initial_model/saved_model_AL_0.ckpt"
    else:
        print(f"Resuming from epoch {resume_epoch}")

    checkpoint_backbone = torch.load(model_path, map_location=torch.device(DEVICE))
    print("Pre-trained epoch: ", checkpoint_backbone['epoch'])
    models['backbone'].load_state_dict(checkpoint_backbone['model'])

    if al_cycle > 0:
        model_path_last_cycle = f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_cycle_{al_cycle}.ckpt"
        checkpoint = torch.load(model_path_last_cycle, map_location=torch.device(DEVICE))
        models['cod'].load_state_dict(checkpoint['model'])

    
    
    ## Model gets set to evaluation mode
    models['backbone'].eval()
    models['cod'].eval()

    pred_patches_dict = dict()

    META_DATA = get_meta_data(DATASET_PATH)
    
    ## Run prediciton
    cropped_data_path = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    test_dataset = get_dataset(cropped_data_path)
    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE)

    # TODO: initialize pred_unpadded and entropy unpadded to zeros
    pred_orig = np.zeros((height, width))
    entropy_unpadded = np.zeros((height, width))

    if config.PROBABILITY: # get A1 and B
        pred_patches_dict, variance_patches_dict = run_pred_al_probability(models['backbone'], test_loader, config.TRANSFORMATION_SCORE)

        # stitch pred patches
        _, pred_stitched = stitch_patches(pred_patches_dict, TEST_REGION)
        pred_orig = center_crop(pred_stitched, height, width, image = False)
        # print(pred_orig.shape)

        # get the offset of original prediction --> A1
        s1,s2,s3 = pred_orig.shape
        half_array = np.full((s1, s2, s3), 0.5)
        pred_orig_offset = np.abs(pred_orig - half_array)
        pred_orig_offset = pred_orig_offset[:,:,0] # only flood class
        # pred_orig_offset = np.sum(pred_orig_offset, axis=-1)/2

        # stitch variance patches --> B
        _, variance_stitched = stitch_patches(variance_patches_dict, TEST_REGION, var=True)
        variance_unpadded = center_crop(variance_stitched, height, width, image = False, var=True)
        # variance_unpadded = variance_unpadded[:,:,0]
        # variance_unpadded = np.sum(variance_unpadded, axis=-1)/2

        # print("offset: ", np.min(pred_orig_offset), np.max(pred_orig_offset))
        # print("variance: ", np.min(variance_unpadded), np.max(variance_unpadded))
        # print("pred_shape: ", pred_orig_offset.shape)
        # print("variance_shape: ", variance_unpadded.shape)

    elif config.ENTROPY: # get A2 and B
        pred_patches_dict, variance_patches_dict = run_pred_al_entropy(models['backbone'], test_loader, config.TRANSFORMATION_SCORE)

        # stitch pred patches
        _, pred_stitched = stitch_patches(pred_patches_dict, TEST_REGION)
        pred_orig = center_crop(pred_stitched, height, width, image = False)
        # print(pred_orig.shape)

        # get the entropy of original prediction --> A2
        pred_orig_clone = np.copy(pred_orig)
        log_out =-1* np.log(pred_orig_clone)
        log_out[log_out != log_out] = 0
        log_out[log_out == float("Inf")] = 0
        log_out[log_out == -float("Inf")] = 0
        log_out[log_out == float("-Inf")] = 0

        entropy_orig = log_out * pred_orig_clone
        entropy_orig = np.sum(entropy_orig, axis=-1)

        # stitch variance patches --> B
        _, variance_stitched = stitch_patches(variance_patches_dict, TEST_REGION, var=True)
        variance_unpadded = center_crop(variance_stitched, height, width, image = False, var=True)
        # variance_unpadded = variance_unpadded[:,:,0]
        # variance_unpadded = np.sum(variance_unpadded, axis=-1)/2

        # print("entropy: ", np.min(entropy_orig), np.max(entropy_orig))
        # print("variance: ", np.min(variance_unpadded), np.max(variance_unpadded))
        # print("pred_shape: ", entropy_orig.shape)
        # print("variance_shape: ", variance_unpadded.shape)


    # print("COD: ", config.COD)
    if config.COD: # A(original) + B(variance) + C(cod)
        pred_patches_dict, cod_loss_patches_dict = run_pred_al_cod(models, test_loader)

        # stitch pred patches
        _, pred_stitched = stitch_patches(pred_patches_dict, TEST_REGION)
        pred_orig = center_crop(pred_stitched, height, width, image = False)
        # print(pred_orig.shape)

        # get COD --> C
        _, cod_loss = stitch_patches(cod_loss_patches_dict, TEST_REGION, var=True)
        cod_loss_unpadded = center_crop(cod_loss, height, width, image = False, var=True)
        # cod_loss_unpadded = cod_loss_unpadded[:,:,0]
        # cod_loss_unpadded = np.sum(cod_loss_unpadded, axis=-1)/2

        # print("COD loss: ", np.min(cod_loss_unpadded), np.max(cod_loss_unpadded))
        # print("cod_loss_shape: ", cod_loss_unpadded.shape)

        if config.PROBABILITY:
            # compute the weighted sum to 2 uncertainty scores (probability offset and COD)
            # pred_offset_agg = pred_orig_offset + config.LAMBDA_1 * variance_unpadded + config.LAMBDA_2 * (1 - cod_loss_unpadded) # for PROB: 0 means uncertain; for COD: 1 means uncertain # A+B+C
            pred_offset_agg = pred_orig_offset - config.LAMBDA_1 * variance_unpadded - config.LAMBDA_2 * cod_loss_unpadded # for PROB: 0 means uncertain; for COD: 1 means uncertain # A+B+C

            # print("A1 + B + C: ", np.min(pred_offset_agg), np.max(pred_offset_agg))
        elif config.ENTROPY:
            # compute the weighted sum to 2 uncertainty scores (entropy and COD); higher entropy means recommend so we add (1 - pred_unpadded_cod)
            entropy_agg = -entropy_orig - config.LAMBDA_1_A2 * variance_unpadded - config.LAMBDA_2_A2 * cod_loss_unpadded # for ENT: 1 means uncertain; for COD: 1 means uncertain # A+B+C

            # print(np.min(entropy_agg), np.max(entropy_agg))
    else: # A(original) + B(variance)
        if config.PROBABILITY: # A1 + B
            pred_offset_agg = pred_orig_offset - config.LAMBDA_1 * variance_unpadded
        elif config.ENTROPY: # A2 + B
            entropy_agg = -entropy_orig - config.LAMBDA_1_A2 * variance_unpadded

    
    if config.PROBABILITY:
        # get aggregate score of each superpixel
        superpixel_scores = get_superpixel_scores(superpixels_group, pred_offset_agg, forest_prob, config.SUPERPIXEL_SCORE, method='offset', use_forest=use_forest)

        # sort by prob score in ascending order; most uncertain superpixel first (whichever is close to 0.5)
        superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]))
    elif config.ENTROPY:
        # get aggregate score of each superpixel
        superpixel_scores = get_superpixel_scores(superpixels_group, entropy_agg, forest_prob, config.SUPERPIXEL_SCORE, method='entropy', use_forest=use_forest)

        # sort by prob score in descending order; highest entropy first
        # superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]), reverse=True)

        # after the sign is reversed
        superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]))
    elif config.COD:
        superpixel_score = config.SUPERPIXEL_SCORE
        if config.SUPERPIXEL_SCORE == "MIN":
            superpixel_score = "MAX"
            
        # get aggregate score of each superpixel
        superpixel_scores = get_superpixel_scores(superpixels_group, cod_loss_unpadded, forest_prob, superpixel_score, method='cod')

        # sort by prob score in ascending order; most uncertain superpixel first (whichever is close to 0.5)
        # superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]), reverse=True)

        # after the sign is reversed
        superpixel_scores = dict(sorted(superpixel_scores.items(), key=lambda item: item[1]))

    # select top-N superpixels
    selected_superpixels, max_items = select_superpixels(total_superpixels, superpixel_scores, forest_superpixels, annotated_superpixels)

    np.save(f"./users/{student_id}/output/Region_{TEST_REGION}_pred.npy", pred_orig)

    ## Stitch pred patches back together
    # _, pred_stitched_2 = stitch_patches(pred_patches_dict, TEST_REGION)
    # pred_unpadded_2 = center_crop(pred_stitched_2, height, width, image = False)
    pred_final = 1 - np.argmax(pred_orig, axis=-1)
    # np.save(f"./users/{student_id}/output/Region_{TEST_REGION}_pred.npy", pred_final)
    pred_final = np.where(pred_final == 0, -1, pred_final)

    gt_labels = np.load(f"./data_al/repo/groundTruths/Region_{TEST_REGION}_GT_Labels.npy")
    metrices, metrices_unlabeled = elev_eval.run_eval(pred_final, gt_labels, updated_labels)

    # updated_labels = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_labels.png', TEST_REGION)

    annotated_pixels = np.where(updated_labels != 0, 1, 0)
    annotated_pixels_percent = (np.sum(annotated_pixels) / (updated_labels.shape[0] * updated_labels.shape[1])) * 100
    print("Ann pixels percent: ", annotated_pixels_percent)

    annotated_pixels_percent = float("{:.2f}".format(annotated_pixels_percent))

    metrices += "\n"
    metrices += f"Annotated Pixels: {annotated_pixels_percent} %"

    file_path_lambda_search = f"./users/{student_id}/output/Region_{TEST_REGION}_TEST/L1.{config.LAMBDA_1}_L2.{config.LAMBDA_2}_L3.{config.LAMBDA_3}_B1.{config.BETA_1}_B2.{config.BETA_2}_P.{config.PROBABILITY}_E.{config.ENTROPY}_C.{config.COD}_EVAR.{config.ENT_VAR}_TA.{config.TRANSFORMATION_SCORE}_SA.{config.SUPERPIXEL_SCORE}/Region_{TEST_REGION}_Metrics.txt"
    with open(file_path_lambda_search, "w") as fp:
        fp.write(metrices)

    file_path = f"./users/{student_id}/output/Region_{TEST_REGION}_Metrics_C{al_cycle}.txt"
    with open(file_path, "w") as fp:
        fp.write(metrices)
    
    # get the superpixels to be recommended in this iteration and save as png
    # interval = (139 - 25) / (max_items - 1)
    slot_values = np.linspace(0.1, 0.99, max_items)

    recommended_superpixels = np.zeros((height, width))
    recommended_superpixels_actual = np.zeros((height, width))
    for i, sid in enumerate(selected_superpixels):
        # pixels = superpixels_group[sid]
        print(i, sid)
        print(int(sid/width), sid%width)

        # pixels = [(int(sid/width), sid%width)]
        # slot_val = int(139 - i * interval)
        # recommended_superpixels[tuple(zip(*pixels))] = slot_values[i]
        # recommended_superpixels[tuple(zip(*pixels))] = slot_values[max_items - i - 1]

        he, we = int(sid/width), sid%width
        recommended_superpixels[he][we] = slot_values[max_items - i - 1]

        try:
            recommended_superpixels[he-1][we-1] = slot_values[max_items - i - 1]
        except IndexError:
            pass
        
        try:
            recommended_superpixels[he-1][we] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-1][we+1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he][we-1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he][we+1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+1][we-1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+1][we] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+1][we+1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        



        try:
            recommended_superpixels[he-2][we-2] = slot_values[max_items - i - 1]
        except IndexError:
            pass
        
        try:
            recommended_superpixels[he-2][we-1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-2][we] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-1][we+1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-2][we+2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-1][we-2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he][we-2] = slot_values[max_items - i - 1]
        except IndexError:
            pass
        
        try:
            recommended_superpixels[he+1][we-2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+2][we-2] = slot_values[max_items - i - 1]
        except IndexError:
            pass
        
        try:
            recommended_superpixels[he+2][we-1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+2][we] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+2][we+1] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he+2][we+2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he-1][we+2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        try:
            recommended_superpixels[he][we+2] = slot_values[max_items - i - 1]
        except IndexError:
            pass
        
        try:
            recommended_superpixels[he+1][we+2] = slot_values[max_items - i - 1]
        except IndexError:
            pass

        recommended_superpixels_actual[he][we] = slot_values[max_items - i - 1]
    
    mask = np.where(recommended_superpixels > 0, 1, 0)
    # mask_blue = np.where((accepted_sub_pixels == 1) & (recommended_superpixels > 0)) # For blue holes in superpixel !!!!!!
    mask_blue = np.where(recommended_superpixels_actual > 0)
    mask = np.expand_dims(mask, axis=-1)
    recommended_superpixels = recommended_superpixels.astype('float32')
    result_array = convert_to_rgb(recommended_superpixels)
    result_array = result_array * mask

    result_array[mask_blue] = [0.0, 0.0, 1.0] # For blue holes in superpixel !!!!!!

    # print(result_array.shape)
    plt.imsave(f'./users/{student_id}/output/R{TEST_REGION}_superpixels_test.png', result_array)

    # user's annotation should stay the same in the predicted result even though model predicts otherwise
    if updated_labels is not None:
        flood_pixels = np.where(updated_labels == 1)
        dry_pixels = np.where(updated_labels == -1)
        pred_final[flood_pixels] = 1
        pred_final[dry_pixels] = -1

    # save current prediction as png
    flood_labels = np.where(pred_final > 0.5, 1, 0)
    dry_labels = np.where(pred_final <= 0.5, 1, 0)
    flood_labels = np.expand_dims(flood_labels, axis=-1)
    dry_labels = np.expand_dims(dry_labels, axis=-1)
    flood_labels = flood_labels*np.array([ [ [255, 0, 0] ] ])
    dry_labels = dry_labels*np.array([ [ [0, 0, 255] ] ])
    pred_labels = (flood_labels + dry_labels).astype('uint8')
    pim = Image.fromarray(pred_labels)
    pim.convert('RGB').save(f'./users/{student_id}/output/R{TEST_REGION}_pred_test.png')

    return metrices, metrices_unlabeled, selected_superpixels, superpixels_group


def ann_to_labels(png_image, TEST_REGION):
    if not os.path.exists(png_image):
        final_arr = np.zeros((config.HEIGHT, config.WIDTH))
        return final_arr
    
    ann = cv2.imread(png_image)
    ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

    flood = ann[:, :, 0] == 255
    dry = ann[:, :, 2] == 255

    flood_arr = np.where(flood, 1, 0)
    dry_arr = np.where(dry, -1, 0)

    final_arr = flood_arr + dry_arr

    # forest = np.load(f"./data_al/forest/Region_{TEST_REGION}_forest.npy") # TODO
    # accept_mask = np.where(forest == 0, 1, 0)
    # final_arr = final_arr * accept_mask
    
    return final_arr
        


def train(TEST_REGION, entropy, probability, cod, transformation_agg, superpixel_agg, student_id, al_cycle, al_iters, use_sc_loss=1, use_cod_loss=1, use_forest=1):
    start_time = time.time()

    student_id = student_id.strip()

    # print("Retraining the Model with new labels")

    if not os.path.exists(f"./users/{student_id}"):
        os.mkdir(f"./users/{student_id}")

    if not os.path.exists(f"./users/{student_id}/output"):
        os.mkdir(f"./users/{student_id}/output")

    if not os.path.exists(f"./users/{student_id}/output/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/output/Region_{TEST_REGION}_TEST")

    if not os.path.exists(f"./users/{student_id}/saved_models_al"):
        os.mkdir(f"./users/{student_id}/saved_models_al")
    
    if not os.path.exists(f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST")
    
    if not os.path.exists(f"./users/{student_id}/resume_epoch"):
        os.mkdir(f"./users/{student_id}/resume_epoch")

    if not os.path.exists(f"./users/{student_id}/al_iters"):
        os.mkdir(f"./users/{student_id}/al_iters")

    if not os.path.exists(f"./users/{student_id}/al_cycles"):
        os.mkdir(f"./users/{student_id}/al_cycles")


    config.ENTROPY = entropy
    config.PROBABILITY = probability
    config.COD = cod

    if transformation_agg.strip().lower() == 'avg':
        config.TRANSFORMATION_SCORE = 'AVG'
    else:
        config.TRANSFORMATION_SCORE = 'MAX'

    if superpixel_agg.strip().lower() == 'avg':
        config.SUPERPIXEL_SCORE = 'AVG'
    else:
        config.SUPERPIXEL_SCORE = 'MIN'

    # fallback if user choose both to be 0
    # if (entropy == 0 and probability == 0):
    #     config.PROBABILITY = 1
        
    print("USE_SC_LOSS: ", use_sc_loss)
    print("USE_COD_LOSS: ", use_cod_loss)

    # model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    cod_model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    ema_model = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)
    backbone_net = EvaNet(config.BATCH_SIZE, config.IN_CHANNEL, config.N_CLASSES, ultrasmall = True).to(DEVICE)

    models = {'backbone': backbone_net, 'ema': ema_model, 'cod': cod_model}

    # optimizer = SGD(models['original'].parameters(), lr = 1e-7)
    optimizer_backbone = SGD(models['backbone'].parameters(), lr = 1e-7)

    # models = {'original': model, 'backbone': backbone_net, 'ema': ema_model, 'cod': cod_model}
    # optimizers = {'original': optimizer, 'backbone': optimizer_backbone}

    optimizers = {'backbone': optimizer_backbone}
    
    criterion = ElevationLoss()
    elev_eval = Evaluator()

    # read resume epoch from text file if exists
    try:
        with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'r') as file:
            content = file.read()
            resume_epoch = int(content) 
    except FileNotFoundError:
        resume_epoch = 0
    
    model_path = f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_{resume_epoch}.ckpt"

    # load COD model 
    if al_cycle > 0:
        model_path_last_cycle = f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_cycle_{al_cycle}.ckpt"
        checkpoint = torch.load(model_path_last_cycle, map_location=torch.device(DEVICE))
        models['cod'].load_state_dict(checkpoint['model'])

    if not os.path.exists(model_path):
        print("Model path doesn't exist; using pretrained model!!!")
        model_path = f"./saved_models_evanet/initial_model/saved_model_AL_{resume_epoch}.ckpt"
    else:
        print(f"Resuming from epoch {resume_epoch}")

    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    models['backbone'].load_state_dict(checkpoint['model'])

    updated_labels = ann_to_labels(f'./users/{student_id}/output/R{TEST_REGION}_labels.png', TEST_REGION)

    # need to remake labels after getting updated labels
    remake_data(updated_labels, TEST_REGION)
    
    cropped_data_path_al = f"./data_al/Region_{TEST_REGION}_TEST/cropped_data_val_test_al"
    elev_val_test_dataset_al = get_dataset_al(cropped_data_path_al)
    elev_val_test_seq = np.arange(0, len(elev_val_test_dataset_al), dtype=int)
    elev_val_test_dataset_al = torch.utils.data.Subset(elev_val_test_dataset_al, elev_val_test_seq)
    al_loader = DataLoader(elev_val_test_dataset_al, batch_size = config.BATCH_SIZE)
    
    # Retrain
    ################################## Training Loop#####################################    
    al_loss_dict = dict()
    val_loss_dict = dict()
    min_val_loss = 1e10   
    
    early_stop = EarlyStopping(patience=7) # TODO: this should be a parameter
    VAL_FREQUENCY = 1

    total_epochs = resume_epoch + config.EPOCHS

    last_epoch = resume_epoch
    for epoch in range(resume_epoch, total_epochs):
        print(f"EPOCH: {epoch+1}/{total_epochs} \r")

        ## Models gets set to training mode
        # models['original'].train()
        models['backbone'].train()
        models['ema'].train()

        al_loss = 0 

        for data_dict in tqdm(al_loader):
            al_iters += 1 # al_iters is for the entire steps the model is trained for

            ## RGB data
            rgb_data = data_dict['rgb_data'].float().to(DEVICE)
            rgb_data.requires_grad = True

            ## Elevation data
            elev_data = data_dict['elev_data'].float().to(DEVICE)
            elev_data.requires_grad = False

            norm_elev_data = data_dict['norm_elev_data'].float().to(DEVICE)
            norm_elev_data.requires_grad = False

            """
            ## Data labels
            Elev Loss function label format: Flood = 1, Unknown = 0, Dry = -1 
            """
            labels = data_dict['labels'].float().to(DEVICE)
            labels.requires_grad = False  

            ## Get model prediction
            # pred = models['original'](rgb_data, norm_elev_data)
            logits_backbone, pred_backbone = models['backbone'](rgb_data, norm_elev_data)
            logits_ema, pred_ema = models['ema'](rgb_data, norm_elev_data)
            
            rgb_data_flipx = torch.flip(rgb_data, dims=(-1,))
            rgb_data_flipy = torchvision.transforms.functional.vflip(rgb_data)
            rgb_data_rot90 = torchvision.transforms.functional.rotate(rgb_data, angle=90)
            rgb_data_rot180 = torchvision.transforms.functional.rotate(rgb_data, angle=180)
            rgb_data_rot270 = torchvision.transforms.functional.rotate(rgb_data, angle=270)

            norm_elev_data_flipx = torch.flip(norm_elev_data, dims=(-1,))
            norm_elev_data_flipy = torchvision.transforms.functional.vflip(norm_elev_data)
            norm_elev_data_rot90 = torchvision.transforms.functional.rotate(norm_elev_data, angle=90)
            norm_elev_data_rot180 = torchvision.transforms.functional.rotate(norm_elev_data, angle=180)
            norm_elev_data_rot270 = torchvision.transforms.functional.rotate(norm_elev_data, angle=270)

            logits_flipx, pred_flipx = models['backbone'](rgb_data_flipx, norm_elev_data_flipx) # get pred for horizontal flip
            logits_flipy, pred_flipy = models['backbone'](rgb_data_flipy, norm_elev_data_flipy) # get pred for horizontal flip
            logits_rot90, pred_rot90 = models['backbone'](rgb_data_rot90, norm_elev_data_rot90) # get pred for horizontal flip
            logits_rot180, pred_rot180 = models['backbone'](rgb_data_rot180, norm_elev_data_rot180) # get pred for horizontal flip
            logits_rot270, pred_rot270 = models['backbone'](rgb_data_rot270, norm_elev_data_rot270) # get pred for horizontal flip

            pred_flipx_inv = torch.flip(pred_flipx, dims=(-1,)) # flip back to original orientation
            pred_flipy_inv = torchvision.transforms.functional.vflip(pred_flipy) # flip back to original orientation
            pred_rot90_inv = torchvision.transforms.functional.rotate(pred_rot90, angle=270)
            pred_rot180_inv = torchvision.transforms.functional.rotate(pred_rot180, angle=180)
            pred_rot270_inv = torchvision.transforms.functional.rotate(pred_rot270, angle=90)

            ## Backprop Loss
            loss1 = criterion.forward(pred_backbone, elev_data, labels)
            loss2 = criterion.forward(pred_flipx_inv, elev_data, labels)
            loss3 = criterion.forward(pred_flipy_inv, elev_data, labels)
            loss4 = criterion.forward(pred_rot90_inv, elev_data, labels)
            loss5 = criterion.forward(pred_rot180_inv, elev_data, labels)
            loss6 = criterion.forward(pred_rot270_inv, elev_data, labels)

            # flip and rotate; add all 6
            all_logits = [pred_backbone, pred_flipx_inv, pred_flipy_inv, pred_rot90_inv, pred_rot180_inv, pred_rot270_inv]

            # print("PRED_SHAPE: ", pred_backbone.shape)
            # print("PRED_flipy_inv_shape: ", pred_flipy_inv.shape)

            # create a mask of unlabeled pixels
            ones = torch.ones_like(labels)
            unknown_mask = torch.where(labels == 0, ones, 0)
            unknown_pixels_count = int(torch.sum(unknown_mask))

            # Calculate 3 loss and aggregate them
            supervised_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            self_consistency_loss = loss_self_consistency(all_logits, labels)

            if config.USE_LOGITS:
                ema_loss = F.mse_loss(logits_backbone * unknown_mask, logits_ema * unknown_mask)
            else:
                ema_loss = F.mse_loss(pred_backbone * unknown_mask, pred_ema * unknown_mask) # unknown mask prevents known pixels from being included in the loss computation

            if use_sc_loss and use_cod_loss:
                total_loss = supervised_loss + config.BETA_1 * self_consistency_loss + config.BETA_2 * ema_loss
            elif use_sc_loss:
                total_loss = supervised_loss + config.BETA_1 * self_consistency_loss
            elif use_cod_loss:
                total_loss = supervised_loss + config.BETA_2 * ema_loss

            # backpropagate the total loss
            # optimizers['original'].zero_grad()
            optimizers['backbone'].zero_grad()

            total_loss.backward()

            # optimizers['original'].step()
            optimizers['backbone'].step()

            ## Record loss for batch
            al_loss += total_loss.item()

            update_ema_variables(models['backbone'], models['ema'], 0.999, al_iters)

        al_loss /= len(al_loader)
        al_loss_dict[epoch+1] = al_loss
        last_epoch = epoch + 1
        print(f"Epoch: {epoch+1} AL Loss: {al_loss}" )

    # Skipping model validation, just save model after all the epochs are trained
    print("Saving Model")
    torch.save({'epoch': last_epoch,  # when resuming, we will start at the next epoch
                'model': models['backbone'].state_dict(),
                'optimizer': optimizers['backbone'].state_dict()}, 
                f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_{last_epoch}.ckpt")
    
    
    with open(f"./users/{student_id}/resume_epoch/R{TEST_REGION}.txt", 'w') as file:
        file.write(str(last_epoch))

    with open(f"./users/{student_id}/al_iters/R{TEST_REGION}.txt", 'w') as file:
        file.write(str(al_iters))

    with open(f"./users/{student_id}/al_cycles/R{TEST_REGION}.txt", 'w') as file:
        file.write(str(al_cycle + 1))
    

    # call AL pipeline once the model is retrained
    metrices, metrices_unlabeled, _, _ = recommend_superpixels(TEST_REGION, config.ENTROPY, config.PROBABILITY, config.COD, transformation_agg, superpixel_agg, student_id, al_cycle, updated_labels=updated_labels, use_forest=use_forest)

    torch.save({'epoch': last_epoch,  # when resuming, we will start at the next epoch
                'model': models['backbone'].state_dict(),
                'optimizer': optimizers['backbone'].state_dict()}, 
                f"./users/{student_id}/saved_models_al/Region_{TEST_REGION}_TEST/saved_model_AL_cycle_{al_cycle + 1}.ckpt")
    
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    elapsed_time = float("{:.2f}".format(elapsed_time))

    metrices += "\n"
    metrices += f"Elapsed Time: {elapsed_time} minutes"

    metrices += "\n"
    metrices += "\n"
    metrices += metrices_unlabeled


    file_path = f"./users/{student_id}/output/Region_{TEST_REGION}_Metrics_C{al_cycle}.txt"
    with open(file_path, "w") as fp:
        fp.write(metrices)
    
    return metrices


if __name__ == "__main__":
    TEST_REGION = "1"

    







