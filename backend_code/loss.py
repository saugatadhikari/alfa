import torch
from torch import nn


class ElevationLoss(nn.Module):
    
    def __init__(self):
        super(ElevationLoss, self).__init__()
        
        ## Initialize unfold function
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), padding = 1)
        self.relu = torch.nn.ReLU()
        

    def forward(self, pred_labels, heights, gt_labels):
        
        """
        INPUT:
            pred_label: The predicted labels for each pixel. Must be of shape (Batch, Channel, Height, Width)
            h: The GT elevation map (height of pixels). Must be of shape (Batch, Channel, Height, Width)
            gt_label: The GT labels for each pixel. Must be of shape (Batch, Channel, Height, Width). FLood = 1, Unknown = 0, Dry = -1 
        OUTPUT:
            final_loss: A single value.
        """

        # print("pred_labels: ", pred_labels.shape)
        ## Get Argmax Index for each prediction channel 
        pred_label_idx = torch.argmax(pred_labels, dim = 1)
        pred_label_idx = torch.unsqueeze(pred_label_idx, dim = 1)
        # print("pred_label_idx: ", pred_label_idx.shape)
        
        
        ## Split the prediciton channels
        flood_pred, dry_pred = torch.tensor_split(pred_labels, 2, dim = 1)
        # print("flood_pred: ", flood_pred.shape)

        
        ## Generate Pred Masks
        ones = torch.ones_like(gt_labels)
        
        flood_mask = torch.where(pred_label_idx == 0, ones, 0)
        dry_mask = torch.where(pred_label_idx == 1, ones, 0)
        # print("flood_mask: ", flood_mask.shape)
            
            
        ## Appropiately Mask Each Prediciton channel
        flood_pred = flood_pred*flood_mask
        dry_pred = -dry_pred*dry_mask
        
        ## Combine all prediciton  channels to One
        unified_pred = flood_pred+dry_pred
        # print("unified_pred: ", unified_pred.shape)
            
            
        pred_flat = torch.reshape(unified_pred, (unified_pred.shape[0], unified_pred.shape[1], -1, 1))
        # print("pred_flat: ", pred_flat.shape)
        
        
        gt_unfolded = self.unfold(gt_labels)
        gt_unfolded = gt_unfolded.permute(0, 2, 1)
        gt_unfolded = torch.unsqueeze(gt_unfolded, dim = 1)
        # print("gt_unfolded: ", gt_unfolded.shape)
        
        
        h_flat = heights.clone()
        h_flat = torch.reshape(h_flat, (h_flat.shape[0], 1, -1, 1))
        # print("h_flat: ", h_flat.shape)
        
        
        h_unfolded = self.unfold(heights)
        h_unfolded = h_unfolded.permute(0, 2, 1)
        h_unfolded = torch.unsqueeze(h_unfolded, dim = 1)
        # print("h_unfolded: ", h_unfolded.shape)
        
        
        ## Calculate Score
        score = 1 - (gt_unfolded*pred_flat)
    
        # Calculate Elevation Delta
        delta_unfolded = (h_flat - h_unfolded)
        
        # Calculate Weight
        weight = torch.ones_like(gt_unfolded)
        # print("weight: ", weight.shape)
        
        ones = torch.ones_like(gt_unfolded)
        zeros = torch.zeros_like(gt_unfolded)
        
        unknown_mask = torch.where(gt_unfolded == 0, zeros, 1)
        pos_elev_mask = torch.where(delta_unfolded > 0, ones, 0)
        neg_elev_mask = torch.where(delta_unfolded < 0, ones, 0)
        gt_flood_mask = torch.where(gt_unfolded == 1, ones, 0)
        gt_dry_mask = torch.where(gt_unfolded == -1, ones, 0)
        
        flood_pos_elev_mask = 1 - (gt_flood_mask*pos_elev_mask)
        dry_neg_elev_mask = 1 - (gt_dry_mask*neg_elev_mask)
        
        loss = (weight*unknown_mask*flood_pos_elev_mask*dry_neg_elev_mask)*score
        # print("loss: ", loss.shape)
                
        
        return torch.sum(loss)
        