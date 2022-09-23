import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose,MaxPooling2D, concatenate,UpSampling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

class network(tf.keras.Model):

    def __init__(self ,model ,loss ,metric ,loss_weights):
        super().__init__()
        self.loss = loss
        self.metric = metric
        self.model = model
        self.loss_weights = loss_weights

    def call(self ,inputs ,training):
        out = self.model(inputs)
        if trainin g= =True:
            return out
        else:
            if type(out) == list:
                return out[0]
            else:
                return out

    def calc_supervision_loss(self ,y_true ,y_preds):
        loss = 0
        for i ,pred in enumerate(y_preds):
            y_resized = tf.image.resize(y_true ,[*pred.shape[1:3]])
            los s+= self.loss_weights[ i +1] * self.loss(y_resized ,pred)
            return loss

    def train_step(self ,data):
        x ,y = data
        with tf.GradientTape() as tape:
            y_preds = self(x ,training=True)
            if type(y_preds) == list:
                loss = self.loss_weights[0] * self.loss(y, y_preds[0])
                acc = self.metric(y, y_preds[0])
                loss += self.calc_supervision_loss(y, y_preds[1:])
            else:
                loss = self.loss(y, y_preds)
                acc = self.metric(y, y_preds)
            trainable_vars = self.trainable_variables  # Network trainable parameters
            gradients = tape.gradient(loss,
                                      trainable_vars)  # Calculating gradients
            # Applying gradients to optimizer
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            return loss, acc

        def test_step(self, data):
            x, y = data
            y_pred = self(x, training=False)
            loss = self.loss(y, y_pred)
            acc = self.metric(y, y_pred)
            return loss, acc

        def train_start(self,train,epochs,optimizer,train_len,loss,batch_size,
                        ,optimizer=Adam,learning_rate=1e-3,val,val_len,metrics=None):

            opt = optimizer(learning_rate=learning_rate, epsilon=None, amsgrad=False,
                      beta_1=0.9, beta_2=0.99)

            self.compile(optimizer=opt, loss=BinaryCrossentropy(),
                          metrics=metrics)

            best_val = np.inf
            for epoch in range(epochs):
                epoch_train_loss = 0.0
                epoch_train_acc = 0.0
                epoch_val_acc = 0.0
                epoch_val_loss = 0.0
                num_batches = 0
                for x in train:
                    if num_batches > (train_len // batch_size ):
                        break
                    a, b = self.train_step(x)
                    epoch_train_loss += a
                    epoch_train_acc += b
                    num_batches += 1
                epoch_train_loss = epoch_train_loss / num_batches
                epoch_train_acc = epoch_train_acc / num_batches
                num_batches_v = 0
                for x in val:
                    if num_batches_v > (val_len // batch_size):
                        break
                    a, b = self.test_step(x)
                    epoch_val_loss += a
                    epoch_val_acc += b
                    num_batches_v += 1
                epoch_val_loss = epoch_val_loss / num_batches_v
                # if epoch_val_loss < best_val:
                #     best_val = epoch_val_loss
                #     print('---Validation Loss improved,saving model---')
                #     model.model.save('./weights', save_format='tf')
                epoch_val_acc = epoch_val_acc / num_batches_v
                template = (
                    "Epoch: {}, TrainLoss: {}, TainAcc: {}, ValLoss: {}, ValAcc {}")
                print(template.format(epoch, epoch_train_loss, epoch_train_acc,
                                      epoch_val_loss, epoch_val_acc))