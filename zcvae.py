import os
import argparse
import time 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Activation, Dense, 
                          Conv2D, Conv2DTranspose,
                          Lambda, Input, BatchNormalization,
                          Dropout, Flatten, Reshape, 
                          LeakyReLU, ELU,
                          MaxPooling2D, UpSampling2D,
                          Concatenate, RepeatVector)
from keras.losses import mse, binary_crossentropy
from keras.metrics import KLDivergence
from keras.initializers import glorot_normal, glorot_uniform
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
import gpu_mem
from zdata import z_data, test_data

def sampling(args):
    z_mean, z_log_var = args
    batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim)) # default: mu=0.0, std=1.0
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

K.clear_session()

parser = argparse.ArgumentParser()

parser.add_argument('root',
                    help='Root directory (remote machine).')
parser.add_argument('data_path',
                    help='Directory of redshift data.')
parser.add_argument('figure_path',
                    help='Directory to save figures in.')

parser.add_argument('Npix',
                    help='Image resolution. Suggested Npix=32.')
parser.add_argument('z_dim', 
                    help='Dimension of latent space.')
parser.add_argument('lr', 
                    help='Learning rate for training.')
parser.add_argument('Nbatch', 
                    help='Number of images in batch.')
parser.add_argument('Nepochs', 
                    help='Number of epochs for training.')

parser.add_argument('load',
                    help='Load pre-trained zCVAE.')
args = parser.parse_args()

root =        args.root
data_path =   args.data_path
figure_path = args.figure_path

Npix = args.Npix
lr = args.lr
epochs = args.Nepochs
batch_size = args.Nbatch
latent_dim = args.z_dim

load = args.load

image_shape = (Npix, Npix, 1)
input_shape = (np.prod(image_shape),)
redshifts = [0.0, 1.07, 2.07, 3.06, 4.17, 5.28]
n_classes = len(redshifts)


(x_train, y_train) = z_data(two_dim=True) 
x_test, y_test = test_data(x_train, y_train)

ix = np.random.randint(0, x_train.shape[0], size=64)
x_train = x_train.reshape(-1, Npix * Npix).astype('float32')
x_test = x_test.reshape(-1, Npix * Npix).astype('float32')

y_train /= max(redshifts)
y_test  /= max(redshifts)


f, k = 64, 5
act = 'relu' 
k_init = glorot_normal()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ encoder
inputs = Input(shape=input_shape, name='image_input')
x = Reshape(image_shape)(inputs)

x = Conv2D(f, k, activation=act, padding='same', 
           kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)           
x = Conv2D(f, k, strides=2, activation=act, padding='same',
           kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)           
x = Conv2D(f, k, activation=act, padding='same', 
           kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)           
x = Conv2D(f, k, strides=2, activation=act, padding='same',
           kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)
x = Conv2D(f, k, activation=act, padding='same', 
           kernel_initializer=k_init)(x)           
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(4 * f, 
          activation=act, kernel_initializer=k_init)(x)

z_mean = Dense(latent_dim, kernel_initializer=k_init, 
               name='z_mean')(x)
z_log_var = Dense(latent_dim, kernel_initializer=k_init, 
                  name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
label_inputs = Input(shape=(1,), dtype='float32', name='label')
label_tile = RepeatVector(4)(label_inputs)
label_tile = Reshape((4,))(label_tile)

x = Concatenate()([latent_inputs, label_tile]) # label_inputs
x = Dense(4 * f, 
          kernel_initializer=k_init, activation=act)(x)
x = Dense(Npix // 2 * Npix // 2 * f, 
          kernel_initializer=k_init, activation=act)(x)
x = Reshape((Npix // 2, Npix // 2, f))(x)
x = Dropout(0.25)(x)

x = Conv2D(f, k, activation=act, padding='same', 
           kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)           
x = Conv2D(f, k, activation=act, padding='same', 
           kernel_initializer=k_init)(x)           
x = Dropout(0.25)(x)           
x = Conv2DTranspose(f, k, strides=2, padding='same', activation=act,
                    kernel_initializer=k_init)(x)
x = Dropout(0.25)(x)                    
x = Conv2D(1, k, activation='sigmoid', padding='same', 
           kernel_initializer=k_init)(x)
outputs = Reshape(input_shape)(x)

decoder = Model([latent_inputs, label_inputs], outputs, name='decoder')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ variational autoencoder
outputs = decoder([encoder(inputs)[2], label_inputs])
vae = Model([inputs, label_inputs], outputs, name='vae_mlp')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

reconstruction_loss = np.prod(image_shape) * mse(inputs, outputs)
kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), 
                       axis=-1) 
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.add_metric(kl_loss, name='kl_loss') # callback ?
vae.compile(optimizer=Adam(lr=lr))

weights_file = os.path.join(root, 'vae_cnn_mnist.h5')

class PlotCall(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            Nplot = 10
            fig, axes = plt.subplots(2, Nplot, figsize=(40, 8))

            def show(ax, im):
                ax.imshow(im.reshape((Npix, Npix)), cmap='gray_r')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            sample = np.random.randint(len(x_train), size=Nplot)
            reconst = vae.predict([x_train[sample], y_train[sample]])                    

            for i in range(Nplot):
                axes[0, i].set_title('%s' % y_train[sample[i]])
                plt.sca(axes[0, i])
                show(axes[0, i], x_train[sample[i]])
                plt.sca(axes[1, i])
                show(axes[1, i], reconst[i])

            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            plt.savefig(os.path.join(figure_path, 'z_cvae_g_plot.png'))
            plt.close()

            print("STATS:", x_train.min(), x_train.max())

class SaveCall(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 250 == 0:
            decoder_path = os.path.join(root, 'decoder_w_2d.h5')
            decoder.save_weights(decoder_path)
            print("saved at location:", decoder_path)
            
plot_call, save_call = PlotCall(), SaveCall()

if load and os.path.exists(weights_file):
    vae.load_weights(weights_file)
else:
    t0 = time.time()
    history = vae.fit([x_train, y_train],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=([x_test, y_test], None),
                      callbacks=[plot_call,
                                 save_call],
                      verbose=1)

    kl = history.history['kl_loss']

    print("training time: %.0f s" % (time.time() - t0))
    vae.save_weights(os.path.join(root,'vae_cnn_mnist.h5'))
    decoder.save_weights(os.path.join(root, 'decoder_w_2d.h5'))


