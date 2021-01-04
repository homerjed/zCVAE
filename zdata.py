import numpy as np 
import os 
import time
import argparse

redshifts = [0.0, 1.07, 2.07, 3.06, 4.17, 5.28]
n_classes = len(redshifts)

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

def z_data(Nclass=64, redshifts=redshifts, two_dim=True):
    """ load cubes for all redshifts """    

    cube_size = 100   
    a, b = 0.0, 1.0

    def norm(x):
        norm_min, norm_max = a, b
        nom = (x - x.min()) * (norm_max - norm_min)
        denom = x.max() - x.min()
        return norm_min + nom / denom

    def scale(x, a=10):
        return (2.0 * x) / (x + a) - 1.0 
    
    def rotate(X):
        Y = np.rot90(X, 1, (1,2))
        Z = np.rot90(Y, 1, (1,2))
        return np.concatenate((X, Y, Z))

    def flip(X, flips=2):
        Y = np.flipud(X)
        Z = np.fliplr(Y)
        return np.concatenate((X, Y, Z))

    """
        real_c = np.load('/home/jed/ml_project/data/cubes_data.npy')
        Ntot = real_c.shape[0]
        real_im = np.zeros((Ntot,Npix,Npix))
        # make into an image
        for i,d in enumerate(real_c):
            h,_,_,_ = plt.hist2d(d.reshape(-1,3)[:,0],d.reshape(-1,3)[:,1],bins=[Npix,Npix],range=[[-1,1],[-1,1]])
            h /= np.max(h.flatten()) - np.min(h.flatten())
            h = 2.0 * h - 1.0 
            real_im[i,:,:] = h
    """

    def sort_data(z, N=Nclass):
        # normalising each class separately here... 
        real_c = np.load(os.path.join(data_path, 
                                      'z_data_cubes',
                                      'cubes_data_%d_%d.npy' % (cube_size, z)),
                         allow_pickle=True)
    
        real_im = np.zeros((Nclass, Npix, Npix))
        for i,d in enumerate(real_c):
            coords = d.reshape(-1,3)

            cube = np.array([coords[:,0],
                             coords[:,1],
                             coords[:,2]]).transpose()
            cube = norm(cube)
            h, _ = np.histogramdd(cube, 
                                  bins=(Npix,Npix,Npix), 
                                  range=[[a,b],[a,b],[a,b]])
            img = norm(h)
            real_im[i,:,:] = h[0,:,:].squeeze() #img # if img scale_a=1.25 
        del real_c 
        return real_im, np.ones((Nclass, 1), dtype="float32") * z

    def sort_data_2d(z, N=Nclass):
        # normalising each class separately here... 
        real_c = np.load(os.path.join(data_path, 
                                      'z_data_cubes',
                                      'cubes_data_%d_%d.npy' % (cube_size, z)),
                         allow_pickle=True)
    
        real_im = np.zeros((Nclass, Npix, Npix))
        for i,d in enumerate(real_c):
            cube = norm(cube)
            h, _, _, _ = plt.hist2d(d.reshape(-1,3)[:,0],
                                    d.reshape(-1,3)[:,1],
                                    bins=[Npix,Npix],
                                    range=[[-1,1],[-1,1]])
            real_im[i,:,:] = norm(h)
        del real_c 
        return real_im, np.ones((Nclass, 1), dtype="float32") * z


    NTOT = n_classes * Nclass
    # redshift images and  labels
    X_train = np.zeros((NTOT, *image_shape))
    y_train = np.zeros((NTOT, 1))
    print("Loading redshift data ...")
    for i,z in enumerate(redshifts):
        # redshift data, labels
        print("z = %.2f" % z)
        X, y = sort_data(z=z)
        X = np.expand_dims(X, axis=-1)
        X = scale(X, a=1.25)
        X_train[i * Nclass : (i + 1) * Nclass] = X
        y_train[i * Nclass : (i + 1) * Nclass] = y
    X_train = norm(X_train)
    print("... loaded %d cubes for redshifts %.1f - %.1f" % (X_train.shape[0], 
                                                             min(redshifts), 
                                                             max(redshifts)))
    if two_dim:
      # return [X_train[:,0,:,:,:].reshape(-1,Npix,Npix,1), y_train]
      return [X_train.reshape(-1,Npix,Npix,1), y_train]

def test_data(xt, yt, Nz=16):
    x_test = np.zeros((Nz * n_classes, *image_shape))
    y_test = np.zeros((Nz * n_classes, 1))

    for z, _ in enumerate(redshifts):
        x_test[z * Nz : (z + 1) * Nz] = np.flipud(
                xt[z * Nz : (z + 1) * Nz]
            )
        y_test[z * Nz : (z + 1) * Nz] = yt[z * Nz : (z + 1) * Nz]
    return x_test, y_test
