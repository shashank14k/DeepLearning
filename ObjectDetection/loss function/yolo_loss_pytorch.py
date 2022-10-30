import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
def calc_iou(b_pred, b_true, img_size):
    if type(b_pred) == list:
        b_pred = torch.unsqueeze(torch.tensor(b_pred), 0)
        b_true = torch.unsqueeze(torch.tensor(b_true), 0)
    box1_x1 = b_pred[..., 0:1] - b_pred[..., 2:3] * img_size / 2
    box1_y1 = b_pred[..., 1:2] - b_pred[..., 3:4] * img_size / 2
    box1_x2 = b_pred[..., 0:1] + b_pred[..., 2:3] * img_size / 2
    box1_y2 = b_pred[..., 1:2] + b_pred[..., 3:4] * img_size / 2
    box2_x1 = b_true[..., 0:1] - b_true[..., 2:3] * img_size / 2
    box2_y1 = b_true[..., 1:2] - b_true[..., 3:4] * img_size / 2
    box2_x2 = b_true[..., 0:1] + b_true[..., 2:3] * img_size / 2
    box2_y2 = b_true[..., 1:2] + b_true[..., 3:4] * img_size / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_supression(bboxes, iou_threshold, img_size):
    """
    accepts [class_pred,obj_prob,x1,y1,width,height] as input
    """
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # Store bboxes after nms in a list
    bboxes_nms = []
    while bboxes:
        # Select box with highest obj_pred initially
        box = bboxes.pop(0)
        bboxes = [
            b
            for b in bboxes
            if b[3] != box[3]
            or calc_iou(
            b[5:],box[5:],img_size
            ) < iou_threshold#keep the subsequent box if the class predictions are diiferent
        ]
        bboxes_nms.append(box)

    return bboxes_nms


class YoloLoss(nn.Module):
    def __init__(self, img_size, window_size, n_boxes, n_classes, obj_loss_wt,
                 nobj_loss_wt):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.img_size = img_size
        self.s = window_size
        self.b = n_boxes
        self.c = n_classes
        self.obj_loss_wt = obj_loss_wt
        self.nobj_loss_wt = nobj_loss_wt

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, self.s, self.s, self.c + self.b * 5)
        y_pred = torch.nan_to_num(y_pred, 0)
        y_true = torch.nan_to_num(y_true, 0)
        box_list = []
        start_index = self.c + 1
        end_index = self.c + 5  # 21:25,26:30,
        for box in range(self.b):
            iou = calc_iou(y_pred[..., start_index:end_index],
                           y_true[..., self.c + 1:self.c + 5],
                           self.img_size)  # shape = (batch,s,s,1)
            start_index = end_index + 1
            end_index = end_index + 5
            box_list.append(iou.unsqueeze(0))
            # iou_b2 = calc_iou(y_pred[...,self.c+5:self.c+10],y_true[...,self.c+1:self.c+5])#shape = (batch,s,s,1)
        """
        Next, we need to concatenate iou_b1 and iou_b2. But we'll first unsqueeze them to add extra dim as
        the two sets are different
        """
        ious = torch.cat(box_list, dim=0)  # (2,batch,s,s,1)
        """
        To select boxes with max iou, we'll do a torch max at dim 0. This way, for each location of each input in
        batch, we'll get only one of the two bounding boxes
        """
        max_iou, bbox = torch.max(ious, dim=0)  # bbox shape (batch,s,s,1)
        identity_obj_f = y_true[..., self.c:self.c + 1]

        """
                BOX COORDINATE LOSS - y_pred[...,21:25]/y_pred[...,25:30] => shape = (batch,s,s,4)
                bbox will return a 0,1 values based on which box has the maximum iou. We'll consider only the best box 
                for computing loss. We'll multiply the bbox with y_pred coordinates. This way, only the best bounding box
                at each cell will be considered. Note y_pred[...,21:25] are predicted coordinates for first bounding box
                """

        bbox_coordinates = bbox * y_pred[..., 26:30] + (1 - bbox) * y_pred[...,
                                                                    21:25]
        # Multiply this with identity_obj_f to compute only when object is present in reality
        bbox_preds = identity_obj_f * bbox_coordinates
        bbox_targets = identity_obj_f * y_true[..., 21:25]
        # square root for width and height
        bbox_preds[..., 2:4] = torch.sign(bbox_preds[..., 2:4]) * torch.sqrt(
            torch.abs(bbox_preds[..., 2:4] + 1e-6))
        bbox_targets[..., 2:4] = torch.sqrt(bbox_targets[..., 2:4])
        # Flatten all examples to calculate mse
        coordinates_loss = self.mse(torch.flatten(bbox_preds, end_dim=2),
                                    torch.flatten(bbox_targets, end_dim=2))
        """
        OBJECT LOSS - y_pred[...,20:21]/y_pred[...,25:26]
        Compute loss related to existence of object
        """
        obj_exist_loss = bbox * y_pred[..., 25:26] + (1 - bbox) * y_pred[...,
                                                                  20:21]
        obj_exist_loss = identity_obj_f * obj_exist_loss
        obj_exist_loss = self.mse(torch.flatten(obj_exist_loss, end_dim=2),
                                  torch.flatten(
                                      identity_obj_f * y_true[..., 20:21],
                                      end_dim=2))
        """
        No object loss, mse score at locations where object is not present
        """
        identity_no_obj_f = 1 - identity_obj_f
        no_obj_loss = self.mse(
            torch.flatten(identity_no_obj_f * y_pred[..., 20:21], end_dim=2),
            torch.flatten(identity_no_obj_f * y_true[..., 20:21], end_dim=2)
        )
        no_obj_loss += self.mse(
            torch.flatten(identity_no_obj_f * y_pred[..., 25:26], end_dim=2),
            torch.flatten(identity_no_obj_f * y_true[..., 20:21], end_dim=2)
        )
        """
        Class Loss
        """
        class_loss = self.mse(
            torch.flatten(identity_obj_f * y_pred[..., :20], end_dim=2),
            torch.flatten(identity_obj_f * y_true[..., :20], end_dim=2)
        )
        loss = self.obj_loss_wt * coordinates_loss + obj_exist_loss + self.nobj_loss_wt * no_obj_loss + class_loss
        return loss