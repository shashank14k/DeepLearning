import tensorflow as tf
import numpy as np

def calc_box_coords(box):
    box_x1= box[... ,0] - box[... ,2 ] /2
    box_y1= box[... ,1] - box[... ,3 ] /2
    box_x2= box[... ,0] + box[... ,2 ] /2
    box_y2= box[... ,1] + box[... ,3 ] /2
    return [box_x1 ,box_x2 ,box_y1 ,box_y2]

def iou(box_pred ,box_target):
    """
    box = (batch, 4) (x,y,w,h)
    box coordinates from midpoint, height and width
    """
    box1 = calc_box_coords(box_pred)
    box2 = calc_box_coords(box_target)

    x1 = tf.math.maximum(box1[0], box2[0])
    y1 = tf.math.maximum(box1[2], box2[2])
    x2 = tf.math.minimum(box1[1], box2[1])
    y2 = tf.math.minimum(box1[2], box2[2])

    intersection = tf.clip_by_value((x2 - x1) ,clip_value_min=0
                                    ,clip_value_max=1) * tf.clip_by_value \
        ((y2 - y1) ,clip_value_min=0 ,clip_value_max=1)

    box1_area = abs((box1[1] - box1[0]) * (box1[3] - box1[2]))
    box2_area = abs((box2[1] - box2[0]) * (box2[3] - box2[2]))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes ,threshold ,iou_threshold):
    """
        bboxes = [confidence,xcenter,ycenter,height,width,class-prob-score]
    """
    bboxes =[bbox for bbox in bboxes if bbox[-1 ] >threshold]
    bboxes =sorted(bboxes, key=lambda x: x[-1] ,reverse=True)
    nms =[]
    while bboxes:
        high =bboxes.pop(0)
        bboxes =[
            bbox for bbox in bboxes
            if iou(bbox[1:-1] ,high[1:-1]) < iou_threshold
        ]
        nms.append(high)

    return nms


def yolo_loss(y_pred, y_true):
    """
    y - shape (s*s*(B*5+C))
    B = 2 is number of bounding boxes
    C = 2 is number of classes car/non car
    """
    obj_para = 5
    noobj_para = 0.5
    print(y_pred.shape)
    print(y_true.shape)
    # Box coordinate loss if object exists
    # Input is (batch,7,7,4) returns the box iou scores at each of the 49 locations
    b1_iou = iou(y_pred[..., 1:5], y_true[..., 1:5])
    b2_iou = iou(y_pred[..., 6:10], y_true[..., 1:5])
    """
        Next step is to find the max iou box at each location from box1,box2.
        For that we concat b1_iou,b2_iou. Box_iou shapes are (batch,7,7).
        Expand each box along axis 0 and then concat to get max iou box at each
        location separately for each sample. Otherwise reduce_max will give a
        single (7,7) output for all samples combined.
    """
    ious = tf.concat([tf.expand_dims(b1_iou, 0), tf.expand_dims(b2_iou, 0)],
                     axis=0)
    best_boxes = tf.argmax(ious, 0)
    best_boxes = tf.cast(tf.expand_dims(best_boxes, axis=3), 'float32')
    exists_boxes = y_true[..., 0]  # batch,7,7
    exists_boxes = tf.cast(tf.expand_dims(exists_boxes, 3),
                           'float32')  # batch,7,7,1

    """For multiplication with y_[....,] of shape batch,7,7,4 we need to 
    expand best_boxes and exists_boxes along dim 3 to make batch,7,7,1
    """

    """
     best_boxes will return a 0,1 array indicating which box was best.
     Shape will be (batch,7,7). With that done, we need to consider the best
     boxes only and convert everything in the bad box to 0 to avoid loss 
     computation from that box. Also, we need to consider only those boxes
     which have an object in target
    """
    box_preds = best_boxes * y_pred[..., 5:10] + (1 - best_boxes) * y_pred[...,
                                                                    0:5]
    box_preds = exists_boxes * box_preds
    box_targets = exists_boxes * y_true[..., 0:5]

    # box shapes now : batch,7,7,4

    box_p = np.zeros(shape=box_preds.shape)
    box_t = np.zeros(shape=box_targets.shape)
    box_p[..., 1:3] = box_preds[..., 1:3]
    box_t[..., 1:3] = box_targets[..., 1:3]
    box_p[..., 3:5] = tf.math.sqrt(tf.math.abs(box_preds[..., 3:5]))
    box_t[..., 3:5] = tf.math.sqrt(tf.math.abs(box_targets[..., 3:5]))
    box_p = np.asarray(box_p).astype('float32')
    box_t = np.asarray(box_t).astype('float32')
    box_p = tf.convert_to_tensor(box_p)
    box_t = tf.convert_to_tensor(box_t)
    loc_loss = obj_para(
        tf.reduce_sum(tf.square(box_p[..., 1:5] - box_t[..., 1:5])))
    loc_loss += tf.reduce_sum(tf.square(box_p[..., 0] - box_t[..., 0]))
    """
    No object loss - only on confidence scores. Penalize no-object predictions
    from both boxes at each location
    """

    nobj_loss = tf.reduce_sum(tf.square(
        (1 - exists_boxes) * y_pred[..., 0:1] - (1 - exists_boxes) * y_true[...,
                                                                     0:1]))

    nobj_loss = nobj_loss + tf.reduce_sum(tf.square(
        (1 - exists_boxes) * y_pred[..., 5:6] - (1 - exists_boxes) * y_true[...,
                                                                     0:1]))
    nobj_loss = nobj_loss * noobj_para

    """
    Class prediction is common for each box
    """
    class_loss = tf.reduce_sum(tf.square(
        exists_boxes * y_pred[..., 10:12] - exists_boxes * y_true[..., 10:12]
    ))

    loc_loss = tf.cast(loc_loss, 'float32')
    nobj_loss = tf.cast(nobj_loss, 'float32')
    class_loss = tf.cast(class_loss, 'float32')

    return loc_loss + nobj_loss + class_loss