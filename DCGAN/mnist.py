import numpy as np 
import time 
import os
#dataset
 
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

import json

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import initializers

import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 22})


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    '''
    Main gan class
    '''
    def __init__(self, img_rows=28, img_cols=28, channel=1,directory =None):
        self.dir = directory
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel

        self.a_loss =[]
        self.d_loss =[]

        # self.G =None
        # self.D =None
        self.AM =None
        self.DM =None
        self.GM = None

        self.initialize()

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train[:, np.newaxis, :, :]
        # self.x_train = X_train
        # self.x_train = input_data.read_data_sets("mnist",\
        # 	one_hot=True).train.images
        self.x_train = X_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)


    def initialize(self):
        optimizer = Adam(lr=0.00014, beta_1=0.6)
        optimizer2 = Adam(lr=0.0002, beta_1=0.6)
        # self.G = self.generator()
        # self.D = self.discriminator()
        self.GM = self.generator(optimizer)
        self.DM =  self.discriminator(optimizer2)
        self.AM= self.adversarial_model(optimizer2)

    # def discriminator(self):
    #     if self.D:
    #         return self.D
    #     self.D = Sequential()
    #     depth = 64
    #     dropout = 0.4


    def generator(self, optimizer = None):
        if self.GM:
            return self.GM
        GM = Sequential()
        dropout = 0.5
        depth = 224 
        input_dim = 100

        moment = 0.9
        inits = initializers.RandomNormal(stddev=0.02)

        # In: 100
        # Out: 3x3x224
        GM.add(Dense(5*5*depth, input_dim=input_dim,kernel_initializer= inits))
        GM.add(BatchNormalization(momentum=0.9))
        GM.add(Activation('relu'))
        GM.add(Reshape((5, 5, depth)))
        GM.add(Dropout(dropout))
        # In: 4x4x224 
        # Out: 7x7x112
        # GM.add(UpSampling2D())
        # GM.add(Cropping2D(cropping=((0, 1), (0, 1))))
        # GM.add(ZeroPadding2D(padding=(2, 2)))
        GM.add(Conv2DTranspose(int(depth/2), 3,strides =(1,1), padding='valid'))
        GM.add(BatchNormalization(momentum=0.9))
        GM.add(Activation('relu'))

        # In: 7x7x112 
        # Out: 14x14x56
        GM.add(UpSampling2D())
        GM.add(Conv2DTranspose(int(depth/4), 5,strides=(1,1), padding='same'))
        GM.add(BatchNormalization(momentum=0.9))
        GM.add(Activation('relu'))

        # In: 14x14x56 
        # Out: 28x28x1
        GM.add(UpSampling2D())
        GM.add(Conv2DTranspose(1, 5,strides=(1,1), padding='same'))
        GM.add(Activation('tanh'))


        GM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])

        GM.summary()
        self.GM = GM
        return self.GM


    def discriminator(self, optimizer = None):
        if self.DM:
            return self.DM
        DM = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        print('input shape',input_shape)
        DM.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, \
                          padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        DM.add(LeakyReLU(alpha=0.2))
        DM.add(Dropout(dropout))

        DM.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        DM.add(LeakyReLU(alpha=0.2))
        DM.add(Dropout(dropout))

        DM.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        DM.add(LeakyReLU(alpha=0.2))
        DM.add(Dropout(dropout))

        # DM.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        # DM.add(LeakyReLU(alpha=0.2))
        # DM.add(Dropout(dropout))

        # Out: 1-dim probability
        DM.add(Flatten())
        DM.add(Dense(1))
        DM.add(Activation('sigmoid'))

        if not optimizer:
            optimizer = Adam(lr=0.0001, beta_1=0.5)

        DM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        DM.summary()
        self.DM =DM
        return self.DM
    


    # def generator_model(self,optimizer=None):
    #     if self.GM:
    #         return self.GM
    #     # optimizer = RMSprop(lr=0.0002)
    #     # optimizer = RMSprop(lr=0.0002, decay=6e-8)
    #     if not optimizer:
    #         optimizer = Adam(lr=0.0001, beta_1=0.5)
    #     self.GM = Sequential()
    #     self.GM.add(self.generator())
    #     self.GM.compile(loss='binary_crossentropy', optimizer=optimizer,\
    #         metrics=['accuracy'])
    #     return self.GM


    # def discriminator_model(self,optimizer=None):
    #     if self.DM:
    #         return self.DM
    #     # optimizer = RMSprop(lr=0.0002)
    #     # optimizer = RMSprop(lr=0.0002, decay=6e-8)
    #     if not optimizer:
    #         optimizer = Adam(lr=0.0001, beta_1=0.5)
    #     # optimizer = Adam(lr=0.0001, beta_1=0.5)
    #     self.DM = Sequential()
    #     self.DM.add(self.discriminator())
    #     self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
    #         metrics=['accuracy'])
    #     return self.DM

    def adversarial_model(self,optimizer=None):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.0001)
        # optimizer = Adam(lr=0.0001, beta_1=0.5)
        if not optimizer:
            optimizer = Adam(lr=0.0001, beta_1=0.5)
        # inp = Input(shape=(100))
        # x =self.GM(inp) 
        # ganout = self.DM(x)
        # self.AM = Model(inputs=inp,outputs=ganout)
        # self.AM.compile(loss='binary_crossentropy',optimizer=optimizer)
        
        
        self.AM = Sequential()
        self.AM.add(self.GM)
        self.AM.add(self.DM)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def train(self, batch_size=32, save_interval=0, epochs=100):
        noise_input = None
        noise_input = np.random.uniform(-1.0, 1.0, size=[60, 100])
        # self.x_train = self.x_train[:5000]

        for e in range(epochs):
            train_size = self.x_train.shape[0]
            index_array = np.arange(train_size)
            np.random.shuffle(index_array)
            nr_batches = int(train_size/batch_size)

            start_time = time.time()
            for i in range(nr_batches):
                ind = index_array[i*batch_size:(i+1)*batch_size]
                images_train = self.x_train[ind]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                images_fake = self.GM.predict(noise)
                x = np.concatenate((images_train, images_fake))
                y = np.ones([2*batch_size, 1])
                y[batch_size:, :] = 0

                self.d_loss.append(self.DM.train_on_batch(x, y))

                y = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                # self.DM.trainable = False
                self.a_loss.append(self.AM.train_on_batch(noise, y))
                # self.DM.trainable = True

                log_mesg = "%d/%d  epoch:%d : [D loss: %f, acc: %f]" % (
                    i,nr_batches,e, self.d_loss[-1][0], self.d_loss[-1][1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (
                    log_mesg, self.a_loss[-1][0], self.a_loss[-1][1])

                print(log_mesg)
                if (i+1)%save_interval==0:
                    self.plot_images(noise=noise_input, step=(e*nr_batches+i+1))
                    self.plot_learning(step=(e*nr_batches+i+1))
            print('epoch took {}s'.format(time.time()-start_time))
            self.save_models(e)

    def save_models(self, step):
        self.GM.save(os.path.join(self.dir[1],str(step)+'gen.h5'))
        self.DM.save(os.path.join(self.dir[1], str(step) + 'dis.h5'))
        losses = {'losses':{'aloss':[float(i[0]) for i in self.a_loss],'dloss':[float(i[0]) for i in self.d_loss]}}
        with open(os.path.join(self.dir[1],str(step)+'losses.json'),'w') as fid:
            json.dump(losses,fid)

    def plot_images(self, noise=None, step=0):
            filename = "mnist_%d.png" % step
            images = self.GM.predict(noise)

            plt.figure(figsize=(16,10))
            for i in range(images.shape[0]):
                plt.subplot(6, 10, i+1)
                image = images[i, :, :, :]
                image = np.reshape(image, [self.img_rows, self.img_cols])
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(self.dir[0],filename))
            plt.close('all')

    def plot_learning(self, step=0):
            filename = "learning_%d.png" % step

            plt.figure(figsize=(16,10))
            x = np.arange(len(self.a_loss))
            plt.plot(x,np.array(self.a_loss)[:,0],label='adversary')
            plt.plot(x,run_mean(np.array(self.a_loss)[:,0],10),label='AM')
            
            plt.plot(x,np.array(self.d_loss)[:,0],label='discriminator')
            plt.plot(x,run_mean(np.array(self.d_loss)[:,0],10),label='DM')
            plt.legend()
            plt.savefig(os.path.join(self.dir[0],filename))
            plt.close('all')


def run_mean(x,n):
    out = x.copy()
    for i in range(n):
        # print(i)
        out += np.r_[x[:i+1],x[:-1-i]] 
    return out/n 


def get_current_dir():
    image_path = 'images'
    model_path = 'models'
    paths = (image_path,model_path)
    out = []
    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)
        if p is image_path:
            nr_of_folders = len(os.listdir(p))
        current_folder = os.path.join(p,str(nr_of_folders))
        os.mkdir(current_folder)
        out.append(current_folder)
    return out

if __name__=='__main__':
    np.random.seed(1) 
    foo = DCGAN(directory=get_current_dir())
    foo.train( batch_size=32, save_interval=10)

