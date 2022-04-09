
**** Work in progress
The aim of this project is to use convolutional neural networks (CNNs) for medical image segmentation. Once trained, the CNN will be able to perform image segmentaion 
on MRI scans.

# Data
The training data can be obtained from here (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation). It consists of image and mask pairs from 110 patients. 

Image: MRI scans 
Mask: Binary pixel image (only 0 and 1) separating the tumor region(value=1) from the non-tumor region(value=0).
![image](https://user-images.githubusercontent.com/98767932/161374388-7499b71a-7373-4149-8a02-4be9c18f217b.png)



# Image Segmentation
Image segmentation is the process differentiating different components in an image. This project, for instance, focuses on differentiating between tumor and non-tumor regions in MRI scans.

![image](https://user-images.githubusercontent.com/98767932/161025979-5ffd3ddf-56fe-4898-bc7a-e0dddfdb5da8.png)

# Model Architectures
To carry image segmentation, I have trained a UNet like CNN from scratch. UNet is a CNN used for biomedical image segmentation. You can find more information regarding it [here](https://arxiv.org/abs/1505.04597). In summary, UNet, like any other classification nerual network, first searches for presence of tumor in the passed image. While doing so, it also memorizes information about the tumor's location (if present) to later construct a final image differentiating tumor and non-tumor regions. An improvement over this is Attnetion UNet. The idea with attention, like in NLP, is to emphasise more on important regions in images. Such a network would give more weightage to areas with tumor and consequently output a better representation of the image. The architecture of Attention UNet is quite similar to that of traditional UNet with slight modifications. It includes an attention block. You can read more about it [here](https://arxiv.org/abs/1804.03999)

### UNet
![UNet](https://user-images.githubusercontent.com/98767932/161028294-9be7fd3d-7767-4649-ab97-aeb94c8b1b03.png)

### Attention UNet
![Attention UNet](https://user-images.githubusercontent.com/98767932/161028449-b25db755-50d1-43e7-ae6f-4ff9a2d86f43.png)
#### Attention block
![Attention block](https://user-images.githubusercontent.com/98767932/161028491-3fb670d8-46ec-4585-8096-02c908a0b0cc.png)

# Loss functions and metrics
I have used two loss functions to train the models, weighted cross entropy and dice loss.

### Weighted Cross Entropy
It is similar to binary cross entropy but has weight assigned to each class. The dataset for biomedical tasks is generally highly imbalanced. There are limited images with tumor and even in those images, the majority of the area does not contain tumor. Hence, the model trained on binary cross entropy could end up just giving outputs of black background (0 pixels) as that too would end up in a pretty high score. Hence, I calculated the average proportion of tumor region in training masks and accordingly weighted the two classes (tumor and non tumor)

### Dice loss
The dice coefficient is calculated as 2 * area of overlap between predicted and real image divided by the total number of pixels in both images. It is the harmonic mean of precision and recall. A higher dice coefficient implies better segmentation, and hence its negative can be used as loss function. You can read more about it [here](https://www.jeremyjordan.me/semantic-segmentation/#loss)

![image](https://user-images.githubusercontent.com/98767932/161381182-d31f4909-5817-4215-ba96-ca1444a3e796.png)

### IOU
IOU and dice-loss are correlated. IOU is the overlapping area between two images divided by the union of their areas. I have used IOU only as an evaluation metric. I have compared the models trained on the above loss functions using IOU.

![image](https://user-images.githubusercontent.com/98767932/161381380-2119e500-8e7c-4243-add9-9c65fa755950.png)


# Model Performance

### UNet
I trained the UNet network using both loss functions, and the model trained using dice loss performed better.

#### Dice loss results

IOU score on test set: 0.5160

![image](https://user-images.githubusercontent.com/98767932/161030810-5f6ff91e-deea-4cfb-a3f0-08e7e039a5ff.png)

#### Weighted cross entropy results

IOU score on test set: 0.3414

![image](https://user-images.githubusercontent.com/98767932/161030921-af93b4e2-17e2-46d7-b32b-1ed60e40ac7e.png)

### Attention UNet

I trained the attention UNet network only using Dice loss, and it outperformed the traditional UNet network. Although the difference was minor, I belive the attention UNet will give much better results when trained for more epochs.

IOU score on test set: 0.6002

![image](https://user-images.githubusercontent.com/98767932/161031237-9421f52b-7cf5-4271-a0ff-8bb23f017825.png)

# CONCLUSION

The IOU scores and image plots from the three models show that attention UNet outperformed the other two models. I trained it for 25 epochs and there weren't any signs of overfitting in validation loss curves. Hence, training for more epochs will likely improve the performance. For more detailed information, I urge you to go through my Kaggle notebooks.

###### [UNet](https://www.kaggle.com/code/shashank069/brainmri-image-segmentation-attentionunet/notebook?scriptVersionId=91727533)
###### [Attention UNet](https://www.kaggle.com/code/shashank069/brainmri-image-segmentation-attentionunet/notebook?scriptVersionId=91727533)




