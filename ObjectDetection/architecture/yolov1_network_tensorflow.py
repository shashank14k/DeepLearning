import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import LeakyReLU,Reshape,Dense,Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose,MaxPooling2D, concatenate,UpSampling2D,Dropout,Add,Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def conv_block(inp,filters,kernel,stride,padding,pool=True):
    conv=inp
    for f,k,s,p in zip(filters,kernel,stride,padding):
        conv=tf.keras.layers.ZeroPadding2D(padding=(p,p))(conv)
        conv=Conv2D(f,k,strides=s,padding='valid')(conv)
        conv=BatchNormalization()(conv)
        conv=LeakyReLU(alpha=0.1)(conv)
    if pool==True:
        conv=MaxPooling2D(strides=2)(conv)
    return conv

inp=Input((448,448,3))
b1=conv_block(inp,[64],[7],[2],[3])
b2=conv_block(b1,[192],[3],[1],[1])
b3=conv_block(b2,[128,256,256,512],[1,3,1,3],[1,1,1,1],[0,1,0,1])
rb1=b3
for i in range(4):
    rb1=conv_block(rb1,[256,512],[1,3],[1,1],[0,1],pool=False)
b4=conv_block(rb1,[512,1024],[1,3],[1,1],[0,1])
rb2=b4
for i in range(2):
    rb2=conv_block(rb2,[512,1024],[1,3],[1,1],[0,1],pool=False)
b5=conv_block(rb2,[1024,1024,1024,1024],[3,3,3,3],[1,2,1,1],[1,1,1,1],pool=False)
f1=Flatten()(b5)
d1=Dense(4096)(f1)
d1=LeakyReLU(alpha=0.1)(d1)
d2=Dense(1470)(d1)
out=tf.reshape(d2,shape=(7,7,30))
model=Model(inp,out)