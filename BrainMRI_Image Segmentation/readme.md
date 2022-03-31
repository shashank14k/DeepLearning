The aim of this project is to use convolutional neural networks (CNNs) for medical image segmentation. Once trained, the CNN will be able to perform image segmentaion 
on MRI scans.
# Image Segmentation
Image segmentation entails differentiating different components in an image. For our purpose, for instance, we need to differentiate between tumor and other parts of
the brain. Hence, the model will be able to identify tumor and non-tumor regions in MRI scans.
![image](https://user-images.githubusercontent.com/98767932/161025979-5ffd3ddf-56fe-4898-bc7a-e0dddfdb5da8.png)

#Model Architectures
To carry image segmentation, I have trained a UNet like CNN from scratch. UNet is a CNN used for biomedical image segmentation. You can find more information regarding it ![here](https://arxiv.org/abs/1505.04597). In summary, though, UNet, like any other classification nerual network, first identifies whether a tumor is present in the passed image. While doing so, it also memorizes the tumor locations and uses this information to output a final image differntiating tumor from non-tumor region. An improvement over this is Attnetion UNet. The idea with attention, like in NLP, is to give more emphasis to important regions in the image. Such a network would give more weightage to areas with tumor and consequently output a better representation of the image. The architecture of Attention UNet is quite similar to that of traditional UNet with slight modifications. It includes and attention block. You can read more about it ![here](https://arxiv.org/abs/1804.03999)

![UNet](https://user-images.githubusercontent.com/98767932/161028294-9be7fd3d-7767-4649-ab97-aeb94c8b1b03.png)
![Attention UNet](https://user-images.githubusercontent.com/98767932/161028449-b25db755-50d1-43e7-ae6f-4ff9a2d86f43.png)
![Attention block](https://user-images.githubusercontent.com/98767932/161028491-3fb670d8-46ec-4585-8096-02c908a0b0cc.png)





