import statistics as stats
import numpy as np
from sklearn.metrics import confusion_matrix



class Evaluator():
    """
    The Evaluator currently can do the following metrics:
        - Precision
        - Recall
        - Fscore
    """

    def __init__(self):

        # Declare Metrics
        self.DRY_ACC = 0
        self.FLOOD_ACC = 0
        
        self.DRY_PRECISION = 0
        self.FLOOD_PRECISION = 0
        
        self.DRY_RECALL = 0
        self.FLOOD_RECALL = 0
        
        self.DRY_FSCORE = 0
        self.FLOOD_FSCORE = 0
    
    def run_eval(self, pred_unpadded, gt_labels, updated_labels):
        
        # cm = confusion_matrix(gt_labels.flatten(), pred_unpadded.flatten(), labels = [0, 1, -1])
        cm = confusion_matrix(gt_labels.flatten(), pred_unpadded.flatten(), labels = [-1, 1, 0])
        TP_0 = cm[0][0]
        FP_0 = cm[1][0]
        FN_0 = cm[0][1]
        TN_0 = cm[1][1]
        
        
        TP_1 = cm[1][1]
        FP_1 = cm[0][1]
        FN_1 = cm[1][0]
        TN_1 = cm[0][0]
        
        
        ####DRY
        self.DRY_ACC = ((TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0))*100
        # print("Dry Accuracy: ", self.DRY_ACC)
        self.DRY_PRECISION = ((TP_0)/(TP_0+FP_0))*100
        # print("Dry Precision: ", self.DRY_PRECISION)
        self.DRY_RECALL = ((TP_0)/(TP_0+FN_0))*100
        # print("Dry Recall: ", self.DRY_RECALL)
        self.DRY_FSCORE = ((2*self.DRY_PRECISION*self.DRY_RECALL)/(self.DRY_PRECISION+self.DRY_RECALL))
        # print("Dry F1 score: ", self.DRY_FSCORE)
        self.DRY_IOU = (TP_0)/(TP_0+FP_0+FN_0)
        # print("Dry IoU: ", self.DRY_IOU)
        
        # print("\n")
        
        ####FLOOD
        self.FLOOD_ACC = ((TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1))*100
        # print("Flood Accuracy: ", self.FLOOD_ACC)
        self.FLOOD_PRECISION = ((TP_1)/(TP_1+FP_1))*100
        # print("Flood Precision: ", self.FLOOD_PRECISION)
        self.FLOOD_RECALL = ((TP_1)/(TP_1+FN_1))*100
        # print("Flood Recall: ", self.FLOOD_RECALL)
        self.FLOOD_FSCORE = ((2*self.FLOOD_PRECISION*self.FLOOD_RECALL)/(self.FLOOD_PRECISION+self.FLOOD_RECALL))
        # print("Flood F1 score: ", self.FLOOD_FSCORE)
        self.FLOOD_IOU = (TP_1)/(TP_1+FP_1+FN_1)
        # print("Flood IoU: ", self.FLOOD_IOU)

        metrices = {
            "Accuracy": float("{:.2f}".format(self.DRY_ACC)),
            "Dry Precision": float("{:.2f}".format(self.DRY_PRECISION)),
            "Dry Recall": float("{:.2f}".format(self.DRY_RECALL)),
            "Dry F1 score": float("{:.2f}".format(self.DRY_FSCORE)),
            "Dry IOU": float("{:.2f}".format(self.DRY_IOU)),
            "Flood Precision": float("{:.2f}".format(self.FLOOD_PRECISION)),
            "Flood Recall": float("{:.2f}".format(self.FLOOD_RECALL)),
            "Flood F1 score": float("{:.2f}".format(self.FLOOD_FSCORE)),
            "Flood IOU": float("{:.2f}".format(self.FLOOD_IOU)),
        }

        dry_acc = float("{:.2f}".format(self.DRY_ACC))
        dry_precision = float("{:.2f}".format(self.DRY_PRECISION))
        dry_recall = float("{:.2f}".format(self.DRY_RECALL))
        dry_f1 = float("{:.2f}".format(self.DRY_FSCORE))
        dry_iou = float("{:.2f}".format(self.DRY_IOU))
        flood_precision = float("{:.2f}".format(self.FLOOD_PRECISION))
        flood_recall = float("{:.2f}".format(self.FLOOD_RECALL))
        flood_f1 = float("{:.2f}".format(self.FLOOD_FSCORE))
        flood_iou = float("{:.2f}".format(self.FLOOD_IOU))

        metrices_str = "   Metrics (Unit: %)    "
        metrices_str += "\n\n"
        metrices_str += f"Accuracy         : {dry_acc}"
        metrices_str += "\n\n"
        metrices_str += f"Dry Precision    : {dry_precision}"
        metrices_str += "\n"
        metrices_str += f"Dry Recall       : {dry_recall}"
        metrices_str += "\n"
        metrices_str += f"Dry F1 score     : {dry_f1}"
        # metrices_str += "\n"
        # metrices_str += f"Dry IOU          : {dry_iou}"
        metrices_str += "\n\n"
        metrices_str += f"Flood Precision  : {flood_precision}"
        metrices_str += "\n"
        metrices_str += f"Flood Recall     : {flood_recall}"
        metrices_str += "\n"
        metrices_str += f"Flood F1 score   : {flood_f1}"
        # metrices_str += "\n"
        # metrices_str += f"Flood IOU        : {flood_iou}"

        # indices_to_remove = np.where(updated_labels != 0)
        # mask = np.ones(updated_labels.shape, dtype=bool)
        # mask[indices_to_remove] = False

        # gt_masked = gt_labels[mask]
        # pred_masked = pred_unpadded[mask] 

        condition = (updated_labels != 0)
        pred_with_ann = pred_unpadded.copy()
        pred_with_ann = np.where(condition, updated_labels, pred_with_ann)

        cm = confusion_matrix(gt_labels.flatten(), pred_with_ann.flatten(), labels = [-1, 1, 0])
        TP_0 = cm[0][0]
        FP_0 = cm[1][0]
        FN_0 = cm[0][1]
        TN_0 = cm[1][1]
        
        
        TP_1 = cm[1][1]
        FP_1 = cm[0][1]
        FN_1 = cm[1][0]
        TN_1 = cm[0][0]
        
        
        ####DRY
        self.DRY_ACC = ((TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0))*100
        # print("Dry Accuracy: ", self.DRY_ACC)
        self.DRY_PRECISION = ((TP_0)/(TP_0+FP_0))*100
        # print("Dry Precision: ", self.DRY_PRECISION)
        self.DRY_RECALL = ((TP_0)/(TP_0+FN_0))*100
        # print("Dry Recall: ", self.DRY_RECALL)
        self.DRY_FSCORE = ((2*self.DRY_PRECISION*self.DRY_RECALL)/(self.DRY_PRECISION+self.DRY_RECALL))
        # print("Dry F1 score: ", self.DRY_FSCORE)
        self.DRY_IOU = (TP_0)/(TP_0+FP_0+FN_0)
        # print("Dry IoU: ", self.DRY_IOU)
        
        # print("\n")
        
        ####FLOOD
        self.FLOOD_ACC = ((TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1))*100
        # print("Flood Accuracy: ", self.FLOOD_ACC)
        self.FLOOD_PRECISION = ((TP_1)/(TP_1+FP_1))*100
        # print("Flood Precision: ", self.FLOOD_PRECISION)
        self.FLOOD_RECALL = ((TP_1)/(TP_1+FN_1))*100
        # print("Flood Recall: ", self.FLOOD_RECALL)
        self.FLOOD_FSCORE = ((2*self.FLOOD_PRECISION*self.FLOOD_RECALL)/(self.FLOOD_PRECISION+self.FLOOD_RECALL))
        # print("Flood F1 score: ", self.FLOOD_FSCORE)
        self.FLOOD_IOU = (TP_1)/(TP_1+FP_1+FN_1)
        # print("Flood IoU: ", self.FLOOD_IOU)

        metrices_unlabeled = {
            "Accuracy": float("{:.2f}".format(self.DRY_ACC)),
            "Dry Precision": float("{:.2f}".format(self.DRY_PRECISION)),
            "Dry Recall": float("{:.2f}".format(self.DRY_RECALL)),
            "Dry F1 score": float("{:.2f}".format(self.DRY_FSCORE)),
            "Dry IOU": float("{:.2f}".format(self.DRY_IOU)),
            "Flood Precision": float("{:.2f}".format(self.FLOOD_PRECISION)),
            "Flood Recall": float("{:.2f}".format(self.FLOOD_RECALL)),
            "Flood F1 score": float("{:.2f}".format(self.FLOOD_FSCORE)),
            "Flood IOU": float("{:.2f}".format(self.FLOOD_IOU)),
        }

        dry_acc = float("{:.2f}".format(self.DRY_ACC))
        dry_precision = float("{:.2f}".format(self.DRY_PRECISION))
        dry_recall = float("{:.2f}".format(self.DRY_RECALL))
        dry_f1 = float("{:.2f}".format(self.DRY_FSCORE))
        dry_iou = float("{:.2f}".format(self.DRY_IOU))
        flood_precision = float("{:.2f}".format(self.FLOOD_PRECISION))
        flood_recall = float("{:.2f}".format(self.FLOOD_RECALL))
        flood_f1 = float("{:.2f}".format(self.FLOOD_FSCORE))
        flood_iou = float("{:.2f}".format(self.FLOOD_IOU))

        metrices_str_unlabeled = "   Metrics (Unit: %)    "
        metrices_str_unlabeled += "\n\n"
        metrices_str_unlabeled += f"Accuracy         : {dry_acc}"
        metrices_str_unlabeled += "\n\n"
        metrices_str_unlabeled += f"Dry Precision    : {dry_precision}"
        metrices_str_unlabeled += "\n"
        metrices_str_unlabeled += f"Dry Recall       : {dry_recall}"
        metrices_str_unlabeled += "\n"
        metrices_str_unlabeled += f"Dry F1 score     : {dry_f1}"
        # metrices_str_unlabeled += "\n"
        # metrices_str_unlabeled += f"Dry IOU          : {dry_iou}"
        metrices_str_unlabeled += "\n\n"
        metrices_str_unlabeled += f"Flood Precision  : {flood_precision}"
        metrices_str_unlabeled += "\n"
        metrices_str_unlabeled += f"Flood Recall     : {flood_recall}"
        metrices_str_unlabeled += "\n"
        metrices_str_unlabeled += f"Flood F1 score   : {flood_f1}"
        # metrices_str_unlabeled += "\n"
        # metrices_str_unlabeled += f"Flood IOU        : {flood_iou}"

        return metrices_str, metrices_str_unlabeled

        
    
    
    @property
    def f_accuracy(self):        
        if self.FLOOD_ACC > 0:
            return self.FLOOD_ACC
        else:
            return 0.0

    @property
    def f_precision(self):        
        if self.FLOOD_PRECISION > 0:
            return self.FLOOD_PRECISION
        else:
            return 0.0

 
    @property
    def f_recall(self):
        if self.FLOOD_RECALL > 0:
            return self.FLOOD_RECALL
        else:
            return 0.0
        
        
    @property
    def f_fscore(self):
        if self.FLOOD_FSCORE > 0:
            return self.FLOOD_FSCORE
        else:
            return 0.0
    
    @property
    def f_iou(self):
        if self.FLOOD_IOU > 0:
            return self.FLOOD_IOU
        else:
            return 0.0
    
    
    
    @property
    def d_accuracy(self):        
        if self.DRY_ACC > 0:
            return self.DRY_ACC
        else:
            return 0.0
    
    @property
    def d_precision(self):        
        if self.DRY_PRECISION > 0:
            return self.DRY_PRECISION
        else:
            return 0.0

 
    @property
    def d_recall(self):
        if self.DRY_RECALL > 0:
            return self.DRY_RECALL
        else:
            return 0.0
        
        
    @property
    def d_fscore(self):
        if self.DRY_FSCORE > 0:
            return self.DRY_FSCORE
        else:
            return 0.0

    @property
    def d_iou(self):
        if self.DRY_IOU > 0:
            return self.DRY_IOU
        else:
            return 0.0