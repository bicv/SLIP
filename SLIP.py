# -*- coding: utf8 -*-
"""
SLIP: a Simple Library for Image Processing.

See http://pythonhosted.org/SLIP


"""
import numpy as np
import scipy.ndimage as nd
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

class Image:
    """
    Collects image processing routines for one given image size:
        - load_in_database : loads a random image in a folder and
        - patch : takes a random patch of the correct size
        - energy, normalize,
        - fourier_grid : defines a useful grid for generating filters in FFT
        - show_FT : displays the envelope and impulse response of a filter
        - convert / invert : go from one side to the other of the fourier transform
        - trans : translation filter in Fourier space

    """
    def __init__(self, pe):
        """
        initializes the Image class

        """
        self.pe = pe
        self.n_x = pe.N_X # n_x
        self.n_y = pe.N_X # n_y
#        self.url_database = url_database
        if self.n_x%2 or self.n_y%2: print('having images of uneven dimensions will fail')

        self.f_x, self.f_y = self.fourier_grid()
        self.f = np.sqrt(self.f_x**2 + self.f_y**2)
#        print ' hello ' , self.f[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1], self.f[(self.n_x-1)//2 , (self.n_y-1)//2 ]
        self.f[(self.n_x-1)//2 + 1, (self.n_y-1)//2 + 1] = 1e-12 # np.inf #
#        self.f[(self.n_x-1)//2 , (self.n_y-1)//2 ] = 1e-12 # np.inf

        # TODO: the max is here 1. / in MotionClouds we use Nyquist
        self.f_norm = self.f / np.max((self.n_x, self.n_y))

    def full_url(self, url_database):
        import os
        return os.path.join(self.pe.datapath, url_database)

    def list_database(self, url_database):
        import os
        try:
            filelist = os.listdir(self.full_url(url_database))
        except:
            print('failed opening database %s' % url_database)
        #TODO
        for garbage in ['.AppleDouble', '.DS_Store']:
            if garbage in filelist:
                filelist.remove(garbage)
        return filelist

    def load_in_database(self, url_database, i_image=None, filename=None, verbose=True):
        """
        Loads a random image from directory url_database

        """
        filelist = self.list_database(url_database=url_database)

        if filename is None:
            if i_image is None:
                i_image = np.random.randint(0, len(filelist))
            else:
                i_image = i_image % len(filelist)

            if verbose: print 'Using image ', filelist[i_image]
            filename = filelist[i_image]

        from pylab import imread
        import os
        image = imread(os.path.join(self.full_url(url_database), filename)) * 1.
        if image.ndim == 3:
            image = image.sum(axis=2)
        return image, filename

    def patch(self, url_database, i_image=None, filename=None, croparea=None, threshold=0.2, verbose=True):
        """
        takes a subimage of size s (a tuple)

        does not accept if energy is relatively below a threshold (flat image)

        """
#         if not(filename is None):
#             image, filename = self.load_in_database(url_database, filename=filename, verbose=verbose)
#         else:
        image, filename = self.load_in_database(url_database, i_image=i_image, filename=filename, verbose=verbose)

        if (croparea is None):
            image_size_h, image_size_v = image.shape
            if self.n_x > image_size_h or self.n_y > image_size_v:
                raise Exception('Patch size too big for the image in your DB')
            elif self.n_x == image_size_h or self.n_y == image_size_v:
                return image, filename, [0, self.n_x, 0, self.n_y]
            else:
                energy = image[:].std()
                energy_ = 0

                while energy_ < threshold*energy:
                    #if energy_ > 0: print 'dropped patch'
                    x_rand = int(np.ceil((image_size_h-self.n_x)*np.random.rand()))
                    y_rand = int(np.ceil((image_size_v-self.n_y)*np.random.rand()))
                    image_ = image[(x_rand):(x_rand+self.n_x), (y_rand):(y_rand+self.n_y)]
                    energy_ = image_[:].std()

                if verbose: print 'Cropping @ [l,r,b,t]: ', [x_rand, x_rand+self.n_x, y_rand, y_rand+self.n_y]

                croparea = [x_rand, x_rand+self.n_x, y_rand, y_rand+self.n_y]
        image_ = image[croparea[0]:croparea[1], croparea[2]:croparea[3]]
        image_ -= image_.mean()
        return image_, filename, croparea

    def energy(self, image):
        #       see http://fseoane.net/blog/2011/computing-the-vector-norm/
        return  np.mean(image.ravel()**2)

    def normalize(self, image, center=True, use_max=True):
        image_ = image.copy()
        if center: image_ -= np.mean(image_.ravel())
        if use_max:
            if np.max(np.abs(image_.ravel()))>0: image_ /= np.max(np.abs(image_.ravel()))
        else:
            if self.energy(image_)>0: image_ /= self.energy(image_)**.5
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
        return np.mgrid[(-self.n_x//2):((self.n_x-1)//2 + 1), (-self.n_y//2):((self.n_y-1)//2 + 1)]

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

    def show_FT(self, FT, axis=False):#,, phase=0. do_complex=False
        N_X, N_Y = FT.shape
        image_temp = self.invert(FT)#, phase=phase)
        import pylab
#         origin : [‘upper’ | ‘lower’], optional, default: None
#         Place the [0,0] index of the array in the upper left or lower left corner of the axes. If None, default to rc image.origin.
#         extent : scalars (left, right, bottom, top), optional, default: None
#         Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        fig = pylab.figure(figsize=(12,6))
        a1 = fig.add_subplot(121)
        a2 = fig.add_subplot(122)
        a1.imshow(np.absolute(FT), cmap=pylab.cm.hsv, origin='upper')
        a2.imshow(image_temp/np.abs(image_temp).max(), vmin=-1, vmax=1, cmap=pylab.cm.gray, origin='upper')
        if not(axis):
            pylab.setp(a1, xticks=[], yticks=[])
            pylab.setp(a2, xticks=[], yticks=[])
        a1.axis([0, N_X, N_Y, 0])
        a2.axis([0, N_X, N_Y, 0])
        return fig, a1, a2

    def convert(self, image):
        return fftshift(fft2(image))

    def invert(self, FT_image, full=False):#, phase=np.pi/2):
        if full:
            return ifft2(ifftshift(FT_image)) # *np.exp(1j*phase)
        else:
            return ifft2(ifftshift(FT_image)).real

    def FTfilter(self, image, FT_filter, full=False):
        FT_image = self.convert(image) * FT_filter
        return self.invert(FT_image, full=full)

    def trans(self, u, v):
#        return np.exp(-1j*2*np.pi*(u*self.f_x + v*self.f_y))
        return np.exp(-1j*2*np.pi*(u/self.n_x*self.f_x + v/self.n_y*self.f_y))

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
#
    def coco(self, image, filter_, normalize=True):
        """
        Returns the correlation coefficient
#
        """
        from scipy.signal import correlate2d
        coco = correlate2d(image, filter_, mode='same')
        if normalize:
            coco /= np.sqrt(self.energy(image)*self.energy(filter_))
#
        return coco

    def olshausen_whitening_filt(self, f_0=.16, alpha=1.4, N=0.01):
        """
        Returns the whitening filter used by (Olshausen, 98)

        f_0 = 200 / 512

        /!\ you will have some problems at dewhitening without a low-pass

        """

        K_ols = (N**2 + self.f_norm**2)**.5 * self.low_pass(f_0=f_0, alpha=alpha)
        K_ols /= np.max(K_ols)

        return  K_ols

    def low_pass(self, f_0, alpha):
        """
        Returns the low_pass filter used by (Olshausen, 98)

        parameters from Atick (p.240)
        f_0 = 22 c/deg in primates: the full image is approx 45 deg
        alpha makes the aspect change (1=diamond on the vert and hor, 2 = anisotropic)

        """
        return np.exp(-(self.f_norm/f_0)**alpha)

    def whitening_filt(self, size=(512, 512),
                             url_database='Yelmo',
                             n_learning=400,
                             N=1.,
                             f_0=.8, alpha=1.4,
                             N_0=.5,
                             recompute=False):
        """
        Returns the average correlation filter in FT space.

        Computes the average power spectrum = FT of cross-correlation, the mean decorrelation
        is given for instance by (Attick, 92).

        """
        #import shelve # http://docs.python.org/library/shelve.html
        #results = shelve.open('white'+ str(size[0]) + '-' + str(size[1]) + '.mat')
        try:
            K = np.load('white'+ str(size[0]) + '-' + str(size[1]) + '.npy')
            if recompute:
                print('Recomputing the whitening filter')
#             boingk
            #    results.remove('K')
            #K = results['K']

        except:
            print ' Learning the whitening filter'
            power_spectrum = 0. # power spectrum
            for i_learning in range(n_learning):
                image, filename, croparea = self.patch(url_database, verbose=False)
                image = self.normalize(image) #TODO : is this fine?
                power_spectrum += np.abs(fft2(image))**2

            power_spectrum = fftshift(power_spectrum)

    ##        from scipy import mgrid
    ##        fx, fy = mgrid[-1:1:1j*size[0],-1:1:1j*size[1]]
    ##        rho = np.sqrt(fx**2+fy**2)
    ##        power_spectrum = N**2 / (rho**2 + N**2)

            power_spectrum /= np.mean(power_spectrum)
            # formula from Attick:
            M = np.sqrt(power_spectrum / (N**2 + power_spectrum)) * self.low_pass(f_0=f_0, alpha=alpha)
            K = M / np.sqrt(M**2 * (N**2 + power_spectrum) + N_0**2)
     #       K = 1 / np.sqrt( N**2 + power_spectrum)  * low_pass(f_0 = f_0, alpha = alpha)
            K /= np.max(K) # normalize energy :  DC is one <=> xcorr(0) = 1

            np.save('white'+ str(size[0]) + '-' + str(size[1]) + '.npy', K)

            #results['K'] = K
            #results['power_spectrum'] = power_spectrum
        #results.close()

        return K

    def whitening(self, image):
        """
        Returns the whitened image
        """
        K = self.whitening_filt(size=image.shape, recompute=False)

        if not(K.shape == image.shape):
            K = self.whitening_filt(size=image.shape, recompute=True)

        return self.FTfilter(image, K)

    def dewhitening(self, white):
        """
        Returns the dewhitened image

        """
        K = self.whitening_filt(white.shape)

        return self.FTfilter(white, 1./K)


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
    from pylab import imread
    # whitening
    image = imread('database/gris512.png')[:,:,0]


