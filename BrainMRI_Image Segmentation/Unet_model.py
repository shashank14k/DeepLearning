#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import chain
import random
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose,MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Loading data
patient=[]
for d in os.listdir('../data/'):
    if d!='data.csv' and d!='README.md':
        patient.append(d)

def create_dataset(start,end,dataset_type):
    train_files = []
    mask_files=[]
    c=0
    for i,p in enumerate(patient[start:end]):
        vals=[]
        mask_files.append(glob('../input/lgg-mri-segmentation/kaggle_3m/'+p+'/*_mask*'))
        for m in mask_files[i]:
            vals.append(np.max(cv2.imread(m)))
        if max(vals)==0:
            print(f'patient { p } has no tumor')
            c+=1
    if c==0:
        print(f'Each patient in {dataset_type} dataset has brain tumor')
    mask_files=list(chain.from_iterable(mask_files))
    for m in mask_files:
        train_files.append(m.replace('_mask',''))
    df = pd.DataFrame(data={"filepath": train_files, 'mask' : mask_files})
    return df

#lenghts for training,validation and testing datasets
a=int(0.9*len(patient))
b=int(0.8*a)

#Creating datasets and finding whether there's any patient with no tumor in the provided data
df_train=create_dataset(0,b,'training')
df_val=create_dataset(b,a,'validation')
df_test=create_dataset(a,len(patient),'testing')

#Marking masks in test dataset as 0 and 1. Will be useful while making final plots
for i in range(0,len(df_test)):
    arr=np.where(cv2.imread(df_test['mask'].loc[i])==255,1,0) 
    v=np.max(arr)
    if v==1:
        df_test.loc[i,'res']=1
    else:
        df_test.loc[i,'res']=0

#Preparing data pipeline
#Function to create image datasets using keras flow_from_dataframe
def img_dataset(df_inp,path_img,path_mask,aug_args,batch):
    img_gen=ImageDataGenerator(rescale=1./255.,**aug_args)
    df_img = img_gen.flow_from_dataframe(dataframe=df_inp,
                                     x_col=path_img,
                                     class_mode=None,
                                     batch_size=batch,
                                    color_mode='rgb',
                                         seed=1,
                                     target_size=(256,256))
    df_mask=img_gen.flow_from_dataframe(dataframe=df_inp,
                                     x_col=path_mask,
                                     class_mode=None,
                                     batch_size=batch,
                                    color_mode='grayscale',
                                        seed=1,
                                     target_size=(256,256))
    data_gen = zip(df_img,df_mask)
    return data_gen

#Declaring loss functions
#Calculating weights for weighted cross entropy
means=[]
for p in df_train['mask']:
    arr=np.where(cv2.imread(p)==255,1,0)
    means.append(arr.mean())
non_tumor_wt=sum(means)/len(means)
tumor_wt=1-non_tumor_wt

#Weighted cross entropy
def wt_cross_entropy_loss(y_true,y_pred):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    loss = K.mean(-(tumor_wt*y_true* K.log(y_pred+1e-7) 
                   + non_tumor_wt* (1 - y_true) * K.log( 1 - y_pred + 1e-7)))
    #l1=K.mean(-(tumor_wt*y_true* K.log(y_pred+1e-7)))
    #l2=K.mean(-(non_tumor_wt* (1 - y_true) * K.log( 1 - y_pred + 1e-7)))
    return loss

#Dice loss
def dice_loss(y_true, y_pred):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersec=K.sum(y_true* y_pred)
    return (-((2* intersec + 0.1) / (K.sum(y_true) + K.sum(y_pred) + 0.1)))

#Iou metric
def iou(y_true,y_pred):
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    iou = (intersec + 0.1) / (union- intersec + 0.1)
    return iou

# UNet Model architecture 
def conv_block(inp,filters,drop=False):
    x=Conv2D(filters,(3,3),padding='same',activation='relu')(inp)
    x=Conv2D(filters,(3,3),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    return x
def encoder_block(inp,filters,drop=False):
    x=conv_block(inp,filters,drop)
    p=MaxPooling2D(pool_size=(2,2))(x)
    return x,p
def decoder_block(inp,filters,concat_layer,drop=False):
    x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp)
    x=concatenate([x,concat_layer])
    x=conv_block(x,filters)
    return x

inputs=Input((256,256,3))
d1,p1=encoder_block(inputs,64)
d2,p2=encoder_block(p1,128)
d3,p3=encoder_block(p2,256)
d4,p4=encoder_block(p3,512)
b1=conv_block(p4,1024)
e2=decoder_block(b1,512,d4)
e3=decoder_block(e2,256,d3)
e4=decoder_block(e3,128,d2)
e5=decoder_block(e4,64,d1)
outputs = Conv2D(1, (1,1),activation="sigmoid")(e5)
model=Model(inputs=[inputs], outputs=[outputs],name='Unet')

#Data augmentation arguments for training set
augmentation_args=dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            fill_mode='nearest')
batch=32

#Function to train model on different loss functions
def train_model(model,save_name,loss_func):
    opt = Adam(learning_rate=1e-4, epsilon=None, amsgrad=False,beta_1=0.9,beta_2=0.99)
    model.compile(optimizer=opt, loss=loss_func,metrics=[iou])
    callbacks = [ModelCheckpoint(save_name,verbose=1,save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1,verbose=1,patience=5, min_lr=1e-6),
                EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)]
    train=img_dataset(df_train,'filepath','mask',augmentation_args,batch)
    val=img_dataset(df_val,'filepath','mask',dict(),batch)

    history = model.fit(train, 
                                  validation_data=val,
                                  steps_per_epoch=len(df_train)/batch, 
                                  validation_steps=len(df_val)/batch, 
                                  epochs = 25,
                                 callbacks=callbacks)

#Training UNet using dice loss
train_model(model,'unet_wts1.hdf5',dice_loss)
#Training UNet using weighted cross entropy
train_model(model,'unet_wts2.hdf5',wt_cross_entropy_loss)

#Model Evaluation
#Function to evaluate the models
def eval_model(model_wts,custom_objects):
    model = load_model(model_wts,custom_objects=custom_objects)
    test=img_dataset(df_test[['filepath','mask']],'filepath','mask',dict(),32)
    model.evaluate(test,steps=len(df_test)/32)
    a=np.random.RandomState(seed=42)
    indexes=a.randint(1,len(df_test[df_test['res']==1]),10)
    for i in indexes:
        img = cv2.imread(df_test[df_test['res']==1].reset_index().loc[i,'filepath'])
        img = cv2.resize(img ,(256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred=model.predict(img)

        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(cv2.imread(df_test[df_test['res']==1].reset_index().loc[i,'mask'])))
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.show()
    
#Evaluating performance of UNet trained on dice loss
eval_model('unet_wts1.hdf5',{'dice_loss':dice_loss,'iou':iou})
#Evaluating performance of UNet trained on weighted cross entropy
eval_model('unet_wts2.hdf5',{'wt_cross_entropy_loss':wt_cross_entropy_loss,'iou':iou})