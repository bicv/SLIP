# -*- coding: utf8 -*-
"""
SLIP: a Simple Library for Image Processing.

See http://pythonhosted.org/SLIP


"""
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import os
PID, HOST = os.getpid(), os.uname()[1]
TAG = 'host-' + HOST + '_pid-' + str(PID)
# -------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

import logging
logging.basicConfig(filename='debug.log', format='%(asctime)s@[' + TAG + '] %(message)s', datefmt='%Y%m%d-%H:%M:%S')
log = logging.getLogger("SLIP")
#log.setLevel(level=logging.WARN)
log.setLevel(level=logging.INFO)
# log.setLevel(logging.DEBUG) #set verbosity to show all messages of severity >= DEBUG
import pickle
import matplotlib.pyplot as plt

class Image:
    """
    Collects image processing routines for one given image size:
     - Some classical related to pure number crunching:
        - creating masks
        - normalize,
        - fourier_grid : defines a useful grid for generating filters in FFT
        - show_FT : displays the envelope and impulse response of a filter
        - convert / invert : go from one side to the other of the fourier transform
        - trans : translation filter in Fourier space
        - whitening procedures
     - Some related to handling experiments:
        - load_in_database : loads a random image in a folder and
        - patch : takes a random patch of the correct size
    """
    def __init__(self, pe):
        """
        initializes the Image class

        """
        self.pe = pe
        self.N_X = pe.N_X # n_x
        self.N_Y = pe.N_Y # n_y

        self.f_x, self.f_y = self.fourier_grid()
        self.f = self.frequency_radius()
        self.f_theta = self.frequency_angle()

        self.x, self.y = np.mgrid[-1:1:1j*self.N_X, -1:1:1j*self.N_Y]
        self.R = np.sqrt(self.x**2 + self.y**2)
        self.mask = (np.cos(np.pi*self.R)+1)/2 *(self.R < 1.)
        self.X, self.Y  = np.meshgrid(np.arange(pe.N_X), np.arange(pe.N_Y))

    def mkdir(self):
        for path in self.pe.figpath, self.pe.matpath:
            if not(os.path.isdir(path)): os.mkdir(path)

    def full_url(self, name_database):
        import os
        return os.path.join(self.pe.datapath, name_database)

    def list_database(self, name_database):
        import os
        try:
            filelist = os.listdir(self.full_url(name_database))
            for garbage in ['.AppleDouble', '.DS_Store']:
                if garbage in filelist:
                    filelist.remove(garbage)
            return filelist
        except:
            print('failed opening database ', name_database)
            print('failed opening database ',  self.full_url(name_database))
            return 'Failed to laod directory'

    def load_in_database(self, name_database, i_image=None, filename=None, verbose=True):
        """
        Loads a random image from directory name_database

        """
        filelist = self.list_database(name_database=name_database)
        np.random.seed(seed=self.pe.seed)

        if filename is None:
            if i_image is None:
                i_image = np.random.randint(0, len(filelist))
            else:
                i_image = i_image % len(filelist)

            if verbose: print 'Using image ', filelist[i_image]
            filename = filelist[i_image]

        import os
        image = plt.imread(os.path.join(self.full_url(name_database), filename)) * 1.
        if image.ndim == 3:
            image = image.sum(axis=2)
        return image, filename

    def make_imagelist(self, name_database, verbose=False):
        """
        makes a list of images with no repetition.
        """

        filelist = self.list_database(name_database)
        N_image_db = len(filelist)
        if self.pe.N_image==None:
            N_image = len(filelist)
        else:
            N_image = self.pe.N_image

        np.random.seed(seed=self.pe.seed)
        shuffling = np.random.permutation(np.arange(len(filelist)))[:N_image]

        imagelist = []
        for i_image in range(N_image):
            image_, filename, croparea = self.patch(name_database, i_image=shuffling[i_image % N_image_db], verbose=verbose)
            imagelist.append([filename, croparea])

        return imagelist

    def get_imagelist(self, exp, name_database='natural'):
        self.mkdir()
        matname = os.path.join(self.pe.matpath, exp + '_' + name_database)
        try:
            imagelist = pickle.load( open(matname + '_images.pickle', "rb" ) )
        except Exception, e:
            # todo : allow to make a bigger batch from a previous run - needs us to parse imagelist... or just concatenate old data...
            log.info('There is no imagelist, creating one: %s ', e)
            if not(os.path.isfile(matname + '_images_lock')):
                log.info(' > setting up image list for %s ', name_database)
                file(matname + '_images_lock', 'w').close()
                imagelist = self.make_imagelist(name_database=name_database)
                pickle.dump(imagelist, open( matname + '_images.pickle', "wb" ) )
                try:
                    os.remove(matname + '_images_lock')
                except Exception, e:
                    log.error('Failed to remove lock file %s_images_lock, error : %s ', matname, e)
            else:
                log.warn(' Some process is building the imagelist %s_images.pickle', matname)
                imagelist = 'locked'

        return imagelist

    def patch(self, name_database, i_image=None, filename=None, croparea=None, threshold=0.2, verbose=True, center=True, use_max=True):
        """
        takes a subimage of size s (a tuple)

        does not accept if energy is relatively below a threshold (flat image)

        """
#         if not(filename is None):
#             image, filename = self.load_in_database(name_database, filename=filename, verbose=verbose)
#         else:
        image, filename = self.load_in_database(name_database, i_image=i_image, filename=filename, verbose=verbose)
        if not i_image==None and not self.pe.seed==None: np.random.seed(seed=self.pe.seed + i_image)

        if (croparea is None):
            image_size_h, image_size_v = image.shape
            if self.N_X > image_size_h or self.N_Y > image_size_v:
                raise Exception('Patch size too big for the image in your DB')
            elif self.N_X == image_size_h or self.N_Y == image_size_v:
                return image, filename, [0, self.N_X, 0, self.N_Y]
            else:
                energy = image.std()
                energy_ = 0

                while energy_ < threshold*energy:
                    #if energy_ > 0: print 'dropped patch'
                    x_rand = int(np.ceil((image_size_h-self.N_X)*np.random.rand()))
                    y_rand = int(np.ceil((image_size_v-self.N_Y)*np.random.rand()))
                    image_ = image[(x_rand):(x_rand+self.N_X), (y_rand):(y_rand+self.N_Y)]
                    energy_ = image_[:].std()

                if verbose: print 'Cropping @ [l,r,b,t]: ', [x_rand, x_rand+self.N_X, y_rand, y_rand+self.N_Y]

                croparea = [x_rand, x_rand+self.N_X, y_rand, y_rand+self.N_Y]
        image_ = image[croparea[0]:croparea[1], croparea[2]:croparea[3]]
        if self.pe.do_mask: image_ *= self.mask
        image_ = self.normalize(image_, center=center, use_max=use_max)
        return image_, filename, croparea

    def normalize(self, image, center=True, use_max=True):
        image_ = image.copy()*1.
        if center: image_ -= image_.mean()
        if use_max:
            if np.max(np.abs(image_.ravel()))>0: image_ /= np.max(np.abs(image_.ravel()))
        else:
            if image_.std()>0: image_ /= image_.std() # self.energy(image_)**.5
        return image_

    #### filter definition
    def fourier_grid(self):
        """
            use that function to define a reference frame for envelopes in Fourier space.
            In general, it is more efficient to define dimensions as powers of 2.

        """

        # From the numpy doc:
        # (see http://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft )
        # The values in the result follow so-called “standard” order: If A = fft(a, n),
        # then A[0] contains the zero-frequency term (the mean of the signal), which
        # is always purely real for real inputs. Then A[1:n/2] contains the positive-frequency
        # terms, and A[n/2+1:] contains the negative-frequency terms, in order of
        # decreasingly negative frequency. For an even number of input points, A[n/2]
        # represents both positive and negative Nyquist frequency, and is also purely
        # real for real input. For an odd number of input points, A[(n-1)/2] contains
        # the largest positive frequency, while A[(n+1)/2] contains the largest negative
        # frequency. The routine np.fft.fftfreq(A) returns an array giving the frequencies
        # of corresponding elements in the output. The routine np.fft.fftshift(A) shifts
        # transforms and their frequencies to put the zero-frequency components in the
        # middle, and np.fft.ifftshift(A) undoes that shift.
        #
        fx, fy = np.mgrid[(-self.N_X//2):((self.N_X-1)//2 + 1), (-self.N_Y//2):((self.N_Y-1)//2 + 1)]
        fx, fy = fx*1./self.N_X, fy*1./self.N_Y
        return fx, fy

#     def expand_complex(self, FT, hue=False):
#         if hue:
#             image_temp = np.zeros((FT.shape[0], FT.shape[1], 4))
#             import matplotlib.cm as cm
#             angle = np.angle(FT)/2./np.pi+.5
#             print 'angle ', angle.min(), angle.max()
#             alpha = np.abs(FT)
#             alpha /= alpha.max()
#             print 'alpha ', alpha.min(), alpha.max()
#             image_temp = cm.hsv(angle)#, alpha=alpha)
#             print image_temp.shape, image_temp.min(), image_temp.max()
#         else:
#             image_temp = 0.5 * np.ones((FT.shape[0], FT.shape[1], 3))
#             FT_ = self.normalize(FT)
#             print 'real ', FT_.real.min(), FT_.real.max()
#             print 'imag ', FT_.imag.min(), FT_.imag.max()
#             image_temp[:,:,0] = 0.5 + 0.5 * FT_.real # * (FT_.real>0) #np.angle(FT)/2./np.pi+.5 #
# #            alpha = np.abs(FT)
# #            alpha /= alpha.max()
#             image_temp[:,:,1] = 0.5
#             image_temp[:,:,2] = 0.5 + 0.5 * FT_.imag #  * (FT_.imag>0)  #alpha
#         return image_temp
    def frequency_radius(self):
#         N_X, N_Y = self.f_x.shape[0], self.f_y.shape[1]
        R2 = self.f_x**2 + self.f_y**2
        R2[self.N_X//2 , self.N_Y//2] = 1e-12 # to avoid errors when dividing by frequency
        return np.sqrt(R2)

    def frequency_angle(self):
        return np.arctan2(self.f_y, self.f_x)

    def show_FT(self, FT, axis=False):#,, phase=0. do_complex=False
        N_X, N_Y = FT.shape
        image_temp = self.invert(FT)#, phase=phase)
        import matplotlib.pyplot as plt
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        fig = plt.figure(figsize=(12,6))
        a1 = fig.add_subplot(121)
        a2 = fig.add_subplot(122)
        a1.imshow(np.absolute(FT), cmap=plt.cm.hsv, origin='upper')
        a2.imshow(image_temp/np.abs(image_temp).max(), vmin=-1, vmax=1, cmap=plt.cm.gray, origin='upper')
        if not(axis):
            plt.setp(a1, xticks=[], yticks=[])
            plt.setp(a2, xticks=[], yticks=[])
        a1.axis([0, N_X, N_Y, 0])
        a2.axis([0, N_X, N_Y, 0])
        return fig, a1, a2

    def invert(self, FT_image, full=False):
        if full:
            return ifft2(ifftshift(FT_image))
        else:
            return ifft2(ifftshift(FT_image)).real

    def FTfilter(self, image, FT_filter, full=False):
        FT_image = fftshift(fft2(image)) * FT_filter
        return self.invert(FT_image, full=full)

    def trans(self, u, v):
        return np.exp(-1j*2*np.pi*(u*self.f_x + v*self.f_y))
#         return np.exp(-1j*2*np.pi*(u/self.N_X*self.f_x + v/self.N_Y*self.f_y))

    def translate(self, image, vec, preshift=False):
        """
        Translate image by vec (in pixels)

        """
        u, v = vec
        u, v = u * 1., v * 1.

        if preshift:
            # first translate by the integer value
            image = np.roll(np.roll(image, np.int(u), axis=0), np.int(v), axis=1)
            u -= np.int(u)
            v -= np.int(v)

        # sub-pixel translation
        return self.FTfilter(image, self.trans(u, v))

    def olshausen_whitening_filt(self):
        """
        Returns the whitening filter used by (Olshausen, 98)

        /!\ you will have some problems at dewhitening without a low-pass

        """
        power_spectrum =  self.f**(-self.pe.white_alpha*2)
        power_spectrum /= np.mean(power_spectrum)
        K_ols = (self.pe.white_N_0**2 + power_spectrum)**-.5
        K_ols *= self.low_pass(f_0=self.pe.white_f_0, steepness=self.pe.white_steepness)
        K_ols /= np.mean(K_ols)
        return  K_ols

    def low_pass(self, f_0, steepness):
        """
        Returns the low_pass filter used by (Olshausen, 98)

        parameters from Atick (p.240)
        f_0 = 22 c/deg in primates: the full image is approx 45 deg
        alpha makes the aspect change (1=diamond on the vert and hor, 2 = anisotropic)

        from Olshausen 98 (p.11):
        f_0  = 200 cycles / image (512 x 512 images)
        in absolute coordiantes : f_0 = 200 / 512 / 2 (?)

        steepness is to produce a "fairly sharp cutoff"

        """
        return np.exp(-(self.f/f_0)**steepness)

    def whitening_filt(self):
        """
        Returns the average correlation filter in FT space.

        Computes the average power spectrum = FT of cross-correlation, the mean decorrelation
        is given for instance by (Attick, 92).

        """
        if self.pe.white_n_learning>0:
            try:
                K = np.load('white'+ str(self.N_X) + '-' + str(self.N_Y) + '.npy')
                if self.pe.recompute:
                    raise('Recomputing the whitening filter')
            except:
                print ' Learning the whitening filter'
                power_spectrum = 0. # power spectrum
                for i_learning in range(self.pe.white_n_learning):
                    image, filename, croparea = self.patch(self.pe.name_database, verbose=False)
                    #image = self.normalize(image) #TODO : is this fine?
                    power_spectrum += np.abs(fft2(image))**2

                power_spectrum = fftshift(power_spectrum)
                power_spectrum /= np.mean(power_spectrum)

                # formula from Atick:
#                 M = np.sqrt(power_spectrum / (self.pe.white_N**2 + power_spectrum))# * self.low_pass(f_0=self.pe.white_f_0, alpha=self.pe.white_alpha)
#                 K = M / np.sqrt(M**2 * (self.pe.white_N**2 + power_spectrum) + self.pe.white_N_0**2)
                K = (self.pe.white_N**2 + power_spectrum)**-.5
                K *= self.low_pass(f_0 = self.pe.white_f_0, steepness = self.pe.steepness)
                K /= np.mean(K) # normalize energy :  DC is one <=> xcorr(0) = 1

                np.save('white'+ str(self.N_X) + '-' + str(self.N_Y) + '.npy', K)
        else:
            K = self.olshausen_whitening_filt()
        return K

    def whitening(self, image):
        """
        Returns the whitened image
        """
        K = self.whitening_filt()
        return self.FTfilter(image, K)

    def dewhitening(self, white, threshold=0.001):
        """
        Returns the dewhitened image

        """
        K = self.whitening_filt()
        K[K==0] = 1.e12 # avoid DC component + corners for which gain is almost null
        FT_image = fftshift(fft2(white)) / K
        FT_image[K<threshold*K.max()] = 0.
        return self.invert(FT_image, full=False)

    def retina(self, image):
        """
        A dummy retina processsing

        """

#        TODO: log-polar transform with openCV
        white = self.whitening(image)
        white = self.normalize(white) # mean = 0, std = 1
        return white

def _test():
    import doctest
    doctest.testmod()
#####################################
#
if __name__ == '__main__':
    _test()

    #### Main
    """
    Some examples of use for the class

    """
    from plt import imread
    # whitening
    image = imread('database/gris512.png')[:,:,0]


