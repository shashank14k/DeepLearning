The aim of the project is to build a CNN that can diagnose presence of thorax diseases from chest X-rays.
# DATA
The data used for training can be obtained from [here](https://www.kaggle.com/code/shashank069/chest-xray-classification/data). It inlcudes chest Xrays from 30,805 patients. There are multiple X-rays for some patients which were probably taken at different points of time. These patients might have succumbed to new diseases and hence separate xrays of the same patient might have presence of different diseases.
### Count of each disease in the dataset
![image](https://user-images.githubusercontent.com/98767932/161377893-d84559a2-29f1-4004-8288-8b68b778e126.png)
### Chest Xrays
![image](https://user-images.githubusercontent.com/98767932/161377729-6503f2b4-20c0-4bd1-9f91-6752d43f87ed.png)

# PRE-PROCESSING

### Data Leakage
While preparing the training, testing, and validation sets, I ensured that separte images of the same patient weren't part of multiple sets. Disregarding so would have led to data leakage, causing an inaccurate evaluation of model perfromance.

### Tackling data imbalance
As is evident from the first image, the dataset is extremely imbalanced. Besides 'No disease', all other classes have an extremely small percentage of positive samples. The highest is for Infiltration (around 18%). To account for this imabalance, I used weighted cross entropy as loss function. The idea with it is to ensure that each class has an equal contribution of positive and negative samples.
![image](https://user-images.githubusercontent.com/98767932/161379040-278af470-b165-4f34-9865-a5168855d69f.png)


This can be easily accomplished by:

![image](https://user-images.githubusercontent.com/98767932/161379030-24913683-6f51-42fc-9145-df0755d49f18.png)


Hence, the final loss function is:

![image](https://user-images.githubusercontent.com/98767932/161378966-7793cc4c-00cd-4e7e-9aa7-e93a3434d1a1.png)


# MODEL
For modeling, I have used DenseNet with pre-trained 'imagenet' weights. I stacked a GlobalAveragePooling layer, and a dense layer on top of Densenet. Only the last two blocks of DenseNet and the additional two layers were kept as trainable.

# Evaluation

1. As this is a multi-classification and multi-label problem (a single patient could be suffering from multiple diseases), it's difficult to create a reliable working model. 
2. The training set had more than 78000 images, but I trained the model for only 10 epochs of 400 iterations and 8 batch length. With more training, the model performance should likely improve.
3. I also will experiment with different trainable layers and with different model architectures.

#### ROC curve

![image](https://user-images.githubusercontent.com/98767932/161426212-814e0fd1-fd71-4e20-8115-2701f31cbaac.png)

#### ROC-AUC score

Besides Hernia, the classifier has a poor roc-auc score for all diseases. It should also be noted that Hernia has a very small sample space. For diseases with roc-auc <=0.5, the classifier is useless

![image](https://user-images.githubusercontent.com/98767932/161426496-54af8a92-590c-46bb-9d6f-97fc87560d01.png)



