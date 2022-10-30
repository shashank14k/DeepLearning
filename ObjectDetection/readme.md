You Only Look Once / YOLO are a series of CNN architectures used for object detection. Object detection entails constructing rectangular boxes (bounding boxes) around
objects of interest in images and, needless to say, has diverse applications in computer vision. Since its release in 2015, yolo has had several advances,
and yolov7 was released in 2022. This, however, is an implementation of the first version [yolov1](https://arxiv.org/abs/1506.02640).

## How yolo works? 
Object detection models prior to yolo functioned extremely slowly. The latency was primarily due to their requirement to make multiple passes on each images.
Yolo, as you might have reckoned from its full name, was groundbreaking as it detected objects in a single pass. It consisits of a series of convolution
blocks followed by a fully connected layer which flattens the image. The fcn layer, with 4096 activations, is then reshaped into a (7,7,30) tensor.

<img width="715" alt="image" src="https://user-images.githubusercontent.com/98767932/198864471-a4b77465-a877-4a2f-b1dc-b513ed78a399.png">

Intuitively, one can think of the model architecture to be dividing an image into 49 cells. Each cell is responsible for detecting whether it has an object.
An object's presence in a cell is dependent on whether its centre lies in that cell. Each cell has 30 indexes and gives two different bounding box predictions.
The idea is that different bounding boxes would enable the model to adjust to different orientations of an object.

<img width="612" alt="image" src="https://user-images.githubusercontent.com/98767932/198864700-7c6b4173-fce7-4d70-bda4-2090fbebf13e.png">

In my implementation: 
1. The first 20 indices (0-19) include class probability scores of the present object. Yolov1 was trained to detect objects of 20 classes.
2. Index 20 and 25 include confidence score about whether there's an image in bounding box
3. Indices 21 and 22 include the object's centroid coordinates reltive to bounding box1 and 26,27 hold the same information but for bounding box 2
4. Indices 23,24,28,29 include the height and width of the two bounding boxes. These are relative to the entire image.

## Loss function

Yolov1 has a bulky loss function, seperable into 5 components:

<img width="710" alt="image" src="https://user-images.githubusercontent.com/98767932/198865341-c8cf92f1-aa79-496b-8df9-a9c5bef35a0c.png">


1. For the cells that have an object in actuality, we select the predicted bounding box with maximum confidence score and calculate:
  1.1 The mean squared difference in predicted and actual object centre coordinates
  1.2 The mean squared difference in square root of predicted and actual height and widths. We take a sqaure root of height and width so that marginal differnces in large bounding boxes are not taxed more than large differences in small bounding boxes
  1.3 The mean squared differnce in predicted and actual confidence scores
2. For the cells that don't have an object, we select both bounding boxes and calculate:
  2.1 The mean squared difference in confidence scores. We may have different weights for losses 1 and loss 2
3. Finally for each class we calculate the mean squared difference in probability scores

## Training 

I've used [pascal_voc dataset](https://www.kaggle.com/datasets/aladdinpersson/pascalvoc-yolo) to train the network. It includes more than 43k images but due
to computational constraints, I trained my network only on 2000 images for 100 epochs. The goal was not to create a flawless object detector but to understand
the math and implementation of yolo network. It's to be noted that the model might predict multiple bounding boxes especially when trained improperly like I've done.
These boxes are cleaned up using [non-max suppression](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/) which selects boxes 
based on their confidence scores, class probabilities (whether multiple boxes predict the same class), and iou scores of boxes predicting same class (Low iou between boxes predicting same class might be because the objects are in different locations).

## Results
 As performance was not the end goal, the results aren't desriable. But they are not bad either. There are several points to improve. Despite non-max suppression,
 the model is predicting multiple boxes with similar iou for the same object. This infers that different cells are classifying the same object under different classes.
 Clearly, the model is under-trained and with more data and iterations should be able to improve. Also, the accuracy of bounding boxes can also be improved.
 However, the results indicate that the implementation works.

<img width="737" alt="image" src="https://user-images.githubusercontent.com/98767932/198865612-89a54ef6-c9e3-4408-814d-4c04b28065a4.png">
<img width="696" alt="image" src="https://user-images.githubusercontent.com/98767932/198865619-71baa7c6-69df-4c95-915c-2b533dd94bbe.png">
<img width="672" alt="image" src="https://user-images.githubusercontent.com/98767932/198865631-40429169-a81d-49a7-b3db-ee639840bb97.png">
<img width="704" alt="image" src="https://user-images.githubusercontent.com/98767932/198865639-7706bb23-4208-4492-8b0f-d5850f1ac435.png">
<img width="675" alt="image" src="https://user-images.githubusercontent.com/98767932/198865642-20bc5f31-e0c6-4098-965c-a3bcb25ae694.png">
<img width="676" alt="image" src="https://user-images.githubusercontent.com/98767932/198865653-11ab3376-818e-4d4e-8eba-19f5b6dacfe6.png">
