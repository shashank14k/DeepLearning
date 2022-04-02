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
![image](https://user-images.githubusercontent.com/98767932/161378610-357c8f85-19d2-47e2-b9d0-bad2a7c1d291.png)
This can be accomplished easily by:
![image](https://user-images.githubusercontent.com/98767932/161378646-534015d0-1d8c-419f-b34e-9085ee866f9c.png)
Hence, the final loss function is:
![image](https://user-images.githubusercontent.com/98767932/161378788-10ce51a4-dd0c-4039-8700-de9aa5d095b0.png)

# MODEL
For modeling, I have used DenseNet with pre-trained 'imagenet' weights. I stacked a GlobalAveragePooling layer, and a dense layer on top of Densenet. Only the last block of DenseNet and the additional two layers were kept as trainable.


