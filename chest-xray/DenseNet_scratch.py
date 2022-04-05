import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Data Pipeline-customize depending on where you store your data

train=pd.read_csv('../input/data/Data_Entry_2017.csv')
img_paths={os.path.basename(x): x for x in glob(os.path.join('..', 'input','data','images*', '*', '*.png'))}
train['img_path']=train['Image Index'].map(img_paths.get)#mapping image ids to all image paths

train=train[['Finding Labels','Patient ID','img_path']]
#Finding unique all labels
labels=train['Finding Labels'].str.split('|',expand=True).stack().unique()
for l in labels:
    train[l]=train['Finding Labels'].map(lambda x:1 if l in x else 0)

#Delcaring random generator for sampling patient IDs

rand_gen=np.random.RandomState(42)

ids=set(train['Patient ID'].unique().flatten())
train_ids=set(rand_gen.choice(train['Patient ID'].unique(),int(len(train['Patient ID'].unique())*0.7),replace=False).flatten())
val_ids=ids-train_ids
test_ids=set(rand_gen.choice(list(val_ids),int(len(val_ids)*0.4),replace=False))
val_ids=val_ids-test_ids
val=train[train['Patient ID'].isin(val_ids)]
test=train[train['Patient ID'].isin(test_ids)]
train=train[train['Patient ID'].isin(train_ids)]
train.reset_index(inplace=True,drop=True)
val.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)


#Image generator function

def img_dataset(df_inp,path,targets,aug_args,shuff=True):
    #base_dir='../input/data/'
    img_gen=ImageDataGenerator(rescale=1./255,**aug_args)
    df_gen = img_gen.flow_from_dataframe(dataframe=df_inp,
                                     x_col=path,
                                     y_col=targets,
                                     class_mode="raw",
                                     batch_size=8,
                                         seed=42,
                                         shuffle=shuff,
                                     target_size=(256,256))
    return df_gen

augm_args=dict(rotation_range=0.15,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            fill_mode='nearest')
batch=8

#Constructing a weighted binary cross entropy function
pos_frq=np.array(train.drop(columns=['Patient ID','img_path']).sum(axis=0))/len(train)
neg_frq=1-pos_frq
pos_weights = neg_frq
neg_weights = pos_frq
pos_contribution = pos_frq * pos_weights 
neg_contribution = neg_frq * neg_weights

def weighted_loss(true,pred):
    loss=0
    true=tf.cast(true, dtype='float32')
    for i in range(len(pos_weights)):
        loss_pos = -1 * K.mean(pos_weights[i] * true[:, i] * K.log(pred[:, i] + 1e-7))
        loss_neg = -1 * K.mean(neg_weights[i] * (1 - true[:, i]) * K.log(1 - pred[:, i] + 1e-7))
        loss += loss_pos + loss_neg
    return loss

#Model-Custom DenseNet class

class DenseNet:
    def __init__(self,input_shape,filters,reps,final_reps,n,activation='softmax'):
        self.filt=filters
        inp=Input(input_shape)
        x=Conv2D(64,7,strides=2,padding='same')(inp)
        x=MaxPooling2D(3,strides=2,padding='same')(x)
        for r in reps:
            x=self.dense_block(x,r)
            x=self.transit_layer(x)
        x=self.dense_block(x,final_reps)
        x=self.classifier(x,n,activation)
        self.model=Model(inp,x)
    #convolutional block
    def conv_block(self,x,filters,kernel=1,strides=1):
        x=BatchNormalization()(x)
        x=ReLU()(x)
        x=Conv2D(filters,kernel,strides,padding='same')(x)
        return x
    #Dense block
    def dense_block(self,x,reps):
        for _ in range(reps):
            y=self.conv_block(x,4*self.filt)
            y=self.conv_block(y,self.filt,3)
            x=Concatenate()([y,x])
return x
    #Transition layer
    def transit_layer(self,x):
        x=self.conv_block(x,K.int_shape(x)[-1]//2)
        x=AveragePooling2D(2,strides=2,padding='same')(x)
        return x
    
    #Classification layer
    def classifier(self,x,n,activation):
        x=GlobalAveragePooling2D()(x)
        x=Dense(units=n,activation=activation)(x)
        return x

model=DenseNet((256,256,3),32,[6,8,12],10,15)

callbacks = [ModelCheckpoint('model_wts',verbose=1,save_best_only=True)]
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss=weighted_loss,metrics=['accuracy'])

train_gen=img_dataset(train,'img_path',train.columns[2:],augm_args,shuff=True)
val_gen=img_dataset(val,'img_path',train.columns[2:],dict(),shuff=False)
history = model.fit_generator(train_gen, 
                              validation_data=val_gen,
                              steps_per_epoch=400, 
                              validation_steps=100, 
                              epochs = 10,
                             callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Evaluation

model = load_model('model_wts',custom_objects={'weighted_loss':weighted_loss})

li=rand_gen.choice(range(len(test)),1000,replace=False)
test.reset_index(inplace=True,drop=True)

def ret_test_imgs(path_indices):
    imgs=[]
    for p in path_indices:
        arr=cv2.imread(test.loc[p]['img_path'])/255
        arr=cv2.resize(arr, dsize=(256,256))
        imgs.append(arr)
    return np.array(imgs)

y_preds=model.predict(ret_test_imgs(li))
y_vals=test[test.index.isin(li)].drop(columns=['Patient ID','img_path']).values

#ROC curve plotting
fpr = {}
tpr = {}
thresh ={}
for i in range(len(test.columns[2:])):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_vals[:,i], y_preds[:,i])
plt.subplots(figsize=(10,8))
for i,c in enumerate(test.columns[2:]):
    plt.plot(fpr[i], tpr[i], linestyle='--', label=f'{c}')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')

#function to check predictions at different tresholds
def ret_tresh(tresh):
    arr=y_preds.copy()
    arr[arr>tresh]=1
    arr[arr<=tresh]=0
    return arr

arr=ret_tresh(0.6)
for i,c in enumerate(test.columns[2:]):
    act=np.sum(y_vals[:,i])
    pr=np.sum(arr[:,i])
    print(f'For {c} actual cases={act},predicted cases={pr},roc-auc-score{roc_auc_score(y_vals[:,i],y_preds[:,i])}')
plt.show()
