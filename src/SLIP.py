# -*- coding: utf8 -*-
from __future__ import (absolute_import, division, print_function, unicode_literals)
"""
SLIP: a Simple Library for Image Processing.

See http://pythonhosted.org/SLIP

"""
import numpy as np

def imread(URL, grayscale=True, rgb2gray=[0.2989, 0.5870, 0.1140]):
    """
    Loads whatever image. Returns a grayscale (2D) image.

    Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

    These scalar values correspond to the grayscale luminance: "The intensity of a pixel is expressed within a given range between a minimum and a maximum, inclusive. This range is represented in an abstract way as a range from 0 (total absence, black) and 1 (total presence, white), with any fractional values in between." This range is here between 0 and 1.

    If ``grayscale`` is True, a grayscale image is obtained by summing over channels following the formula:

    Y  = 0.2989 * R + 0.5870 * G + 0.1140 * B

    http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    which corresponds to the definition of luma:
    http://www.poynton.com/notes/colour_and_gamma/ColorFAQ.html#RTFToC11

    This function tries to guess at best the range and format.
    If that fails, returns a string with the error message.

    TODO: the above formula is an approximation of the official conversion:

        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    in the linear RGB space.
    (see https://en.wikipedia.org/wiki/Grayscale#Colorimetric_.28luminance-preserving.29_conversion_to_grayscale)


    """
    try:
        import imageio
        image = imageio.imread(URL)
        if image.dtype == np.uint8: image = np.array(image, dtype=np.float) / 255.
        image = np.array(image, dtype=np.float)
        if image.ndim > 3:
            return 'dimension higher than 3'
        if image.ndim == 3:
            if image.shape[2]==4: # discard alpha channel
                image = image[:, :, :3] * image[:, :, -1, np.newaxis]
            if image.shape[2] > 4:
                return 'imread : more than 4 channels, have you imported a video?'
            if grayscale is True:
                image *= np.array(rgb2gray)[np.newaxis, np.newaxis, :]
                image = image.sum(axis=-1) # convert to grayscale

        return image
    except:
        return 'could not return an image'

from numpy.fft import fft2, fftshift, ifft2, ifftshift
import os
# -------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

import pickle
import matplotlib.pyplot as plt
import matplotlib
from NeuroTools.parameters import ParameterSet
import logging

class Image:
    """
    This library collects different Image Processing tools.

    Fork me on https://github.com/meduz/SLIP !

    This library is used in other projects, in particular  for use with the ``LogGabor`` and ``SparseEdges`` libraries
    For more information check respective pages @
        - http://pythonhosted.org/LogGabor and
        - http://pythonhosted.org/SparseEdges

    Collects image processing routines for one given image size:
     - Some classical related to pure Fourier number crunching:
        - creating masks
        - normalize,
        - fourier_grid : defines a useful grid for generating filters in FFT
        - show_FT : displays the envelope and impulse response of a filter
        - invert : go to the other of the fourier transform
    - Some usual application of Fourier filtering:
        - trans : translation filter in Fourier space
        - whitening procedures
     - Some related to handling experiments:
        - load_in_database : loads a random image in a folder and
        - patch : takes a random patch of the correct size
    """
    def __init__(self, pe={'N_X':128, 'N_Y':128}):
        """
        Initializes the Image class

        May take as input:

        - a dictionary containing parameters
        - a ``ndarray`` (dimensions ``N_X`` and ``N_Y`` are guessed from this array)
        - a string representing a file or URL pointing to an image file
        - a string pointing to  a file or URL containing a dictionary of parameters
        - a ``NeuroTools.parameters.ParameterSet`` object containing parameters

        Parameters are

        - N_X and N_Y which are respectively the number of pixels in the vertical and horizontal dimensions respectively (MANDATORY)
        - optional parameters which are used in the various functions such as N_image when handling a database or the whitening parameters.

        """
        self.pe = self.get_pe(pe)
        self.init_logging()
        self.init()

    def get_pe(self, pe):
        """ guesses parameters from the init variable
        outputs a ParameterSet
        """
        if type(pe) is tuple:
            return ParameterSet({'N_X':pe[0], 'N_Y':pe[1]})
        elif type(pe) is ParameterSet:
            return pe
        elif type(pe) is dict:
            return ParameterSet(pe)
        elif type(pe) is np.ndarray:
            return ParameterSet({'N_X':pe.shape[0], 'N_Y':pe.shape[1]})
        elif type(pe) is str:
            im = imread(pe)
            if not type(im) is np.ndarray: #  loading an image failed
               return ParameterSet(pe)
            else:
               return ParameterSet({'N_X':im.shape[0], 'N_Y':im.shape[1]})
        else:
            print('error finding parameters')
            return ParameterSet({'N_X':0, 'N_Y':0})

    def init_logging(self, filename='debug.log', name="SLIP"):
        try:
            PID = os.getpid()
        except:
            PID = 'N/A'
        try:
            HOST = os.uname()[1]
        except:
            HOST = 'N/A'
        self.TAG = 'host-' + HOST + '_pid-' + str(PID)
        logging.basicConfig(filename=filename, format='%(asctime)s@[' + self.TAG + '] %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')
        self.log = logging.getLogger(name)
        try:
            self.log.setLevel(level=self.pe.verbose) #set verbosity to show all messages of severity >= DEBUG
        except:
            self.pe.verbose = logging.WARN
            self.log.setLevel(level=self.pe.verbose) #set verbosity to show all messages of severity >= DEBUG

    def get_size(self, im):
        if type(im) is tuple:
            return im
        elif type(im) is str:
            im =  self.imread(im)
        return im.shape[0], im.shape[1]

    def set_size(self, im):
        """
        Re-initializes the Image class with  the size given in ``im``

        May take as input:

        - a numpy array,
        - a string representing a file or URL pointing to an image file
        - a tuple

        Updated parameters are

        - N_X and N_Y which are respectively the number of pixels in the vertical and horizontal dimensions respectively (MANDATORY)

        """
        try: # to read pe as a tuple
            self.pe.N_X, self.pe.N_Y = self.get_size(im)
        except:
            self.log.error('Could not set the size of the SLIP object')
        self.pe.N_X = self.pe.N_X # n_x
        self.pe.N_Y = self.pe.N_Y # n_y
        self.init()

    def init(self):
        """
        Initializes different convenient matrices for image processing.

        To be called when keeping the same Image object but changing the size of the image.

        """
        self.f_x, self.f_y = self.fourier_grid()
        self.f = self.frequency_radius()
        self.f_theta = self.frequency_angle()

        self.x, self.y = np.mgrid[-1:1:1j*self.pe.N_X, -1:1:1j*self.pe.N_Y]
        self.R = np.sqrt(self.x**2 + self.y**2)
        self.mask = ((np.cos(np.pi*self.R)+1)/2 *(self.R < 1.))**(1./self.pe.mask_exponent)
        self.f_mask = self.retina()
        self.X, self.Y  = np.meshgrid(np.arange(self.pe.N_X), np.arange(self.pe.N_Y))

    def mkdir(self):
        """
        Initializes two folders for storing intermediate matrices and images.

        To be called before any operation to store or retrieve a result or figure.

        """
        for path in self.pe.figpath, self.pe.matpath:
            if not(os.path.isdir(path)): os.mkdir(path)

    def full_url(self, name_database):
        import os
        return os.path.join(self.pe.datapath, name_database)

    def list_database(self, name_database):
        """
        Returns a list of the files in a folder

        """
        import os
        try:
            # TODO: use a list of authorized file types
            GARBAGE = ['.AppleDouble', '.DS_Store'] # MacOsX stuff
            filelist = os.listdir(self.full_url(name_database))
            for garbage in GARBAGE:
                if garbage in filelist: filelist.remove(garbage)
            return filelist
        except:
            self.log.error('XX failed opening database ',  self.full_url(name_database))
            return 'Failed to load directory'

    def imread(self, URL, resize=True):
        image = imread(URL)
        if type(image) is str: self.log.error(image)
        elif resize and (self.pe.N_X is not image.shape[0] or self.pe.N_Y is not image.shape[1]):
            self.set_size(image)
        return image

    def load_in_database(self, name_database, i_image=None, filename=None, verbose=True):
        """
        Loads a random image from the database ``name_database``.

        The strategy is to pick one image in the folder using the list provided by the ``list_database``function.


        """
        filelist = self.list_database(name_database=name_database)
        np.random.seed(seed=self.pe.seed)

        if filename is None:
            if i_image is None:
                i_image = np.random.randint(0, len(filelist))
            else:
                i_image = i_image % len(filelist)

            if verbose: print('Using image ', filelist[i_image])
            filename = filelist[i_image]

        import os
        image = self.imread(os.path.join(self.full_url(name_database), filename), resize=False)
        return image, filename

    def make_imagelist(self, name_database, verbose=False):
        """
        Makes a list of images with no repetition.

        Takes as an input the name of a database (the name of a folder in the ``datapath``),
        returns a list of the filenames along with the crop area.

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
        """
        returns an imagelist from a pickled database.

        If the stored imagelist does not exist, creates it.
        The ``exp`` string allows to tag the list for a particular experiment.

        """
        self.mkdir()
        matname = os.path.join(self.pe.matpath, exp + '_' + name_database)
        try:
            imagelist = pickle.load( open(matname + '_images.pickle', "rb" ) )
        except Exception as e:
            # todo : allow to make a bigger batch from a previous run - needs us to parse imagelist... or just concatenate old data...
            self.log.info('There is no imagelist, creating one: %s ', e)
            if not(os.path.isfile(matname + '_images_lock')):
                self.log.info(' > setting up image list for %s ', name_database)
                open(matname + '_images_lock', 'w').close()
                imagelist = self.make_imagelist(name_database=name_database)
                pickle.dump(imagelist, open( matname + '_images.pickle', "wb" ) )
                try:
                    os.remove(matname + '_images_lock')
                except Exception as e:
                    self.log.error('Failed to remove lock file %s_images_lock, error : %s ', matname, e)
            else:
                self.log.warn(' Some process is building the imagelist %s_images.pickle', matname)
                imagelist = 'locked'

        return imagelist

    def patch(self, name_database, i_image=None, filename=None, croparea=None, threshold=0.2, verbose=True,
            preprocess=True, center=True, use_max=True):
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
            if self.pe.N_X > image_size_h or self.pe.N_Y > image_size_v:
                print('N_X patch_v patch_h  ', self.pe.N_X, image_size_h, image_size_v)
                raise Exception('Patch size too big for the image in your DB')
            elif self.pe.N_X == image_size_h or self.pe.N_Y == image_size_v:
                return image, filename, [0, self.pe.N_X, 0, self.pe.N_Y]
            else:
                energy = image.std()
                energy_ = 0

                while energy_ < threshold*energy:
                    #if energy_ > 0: print 'dropped patch'
                    x_rand = int(np.ceil((image_size_h-self.pe.N_X)*np.random.rand()))
                    y_rand = int(np.ceil((image_size_v-self.pe.N_Y)*np.random.rand()))
                    image_ = image[(x_rand):(x_rand+self.pe.N_X), (y_rand):(y_rand+self.pe.N_Y)]
                    energy_ = image_[:].std()

                if verbose: print('Cropping @ [top, bottom, left, right]: ', [x_rand, x_rand+self.pe.N_X, y_rand, y_rand+self.pe.N_Y])

                croparea = [x_rand, x_rand+self.pe.N_X, y_rand, y_rand+self.pe.N_Y]
        image_ = image[croparea[0]:croparea[1], croparea[2]:croparea[3]]
        if self.pe.do_mask: image_ *= self.mask
        image_ = self.normalize(image_, preprocess=preprocess, center=center, use_max=use_max)
        return image_, filename, croparea

    def normalize(self, image, preprocess=True, center=True, use_max=True):
        if preprocess: image_ = self.preprocess(image)
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
        fx, fy = np.mgrid[(-self.pe.N_X//2):((self.pe.N_X-1)//2 + 1), (-self.pe.N_Y//2):((self.pe.N_Y-1)//2 + 1)]
        fx, fy = fx*1./self.pe.N_X, fy*1./self.pe.N_Y
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
        R2[self.pe.N_X//2 , self.pe.N_Y//2] = 1e-12 # to avoid errors when dividing by frequency
        return np.sqrt(R2)

    def frequency_angle(self):
        return np.arctan2(self.f_y, self.f_x)

    def enveloppe_color(self, alpha):
        # 0.0, 1.0, 2.0 are resp. white, pink, red/brownian envelope
        # (see http://en.wikipedia.org/wiki/1/f_noise )
        if alpha == 0:
            return 1.
        else:
            f_radius = np.zeros(self.f.shape)
            f_radius = self.f**alpha
            f_radius[(self.pe.N_X-1)//2 + 1 , (self.pe.N_Y-1)//2 + 1 ] = np.inf
            return 1. / f_radius

    # Fourier number crunching
    def invert(self, FT_image, full=False):
        if full:
            return ifft2(ifftshift(FT_image))
        else:
            return ifft2(ifftshift(FT_image)).real

    def fourier(self, image, full=True):
        """
        Using the ``fourierr`` function, it is easy to retieve its Fourier transformation.

        """
        FT = fftshift(fft2(image))
        if full:
            return FT
        else:
            return np.absolute(FT)

    def FTfilter(self, image, FT_filter, full=False):
        """
        Using the ``FTfilter`` function, it is easy to filter an image with a filter defined in Fourier space.

        """
        FT_image = self.fourier(image, full=True) * FT_filter
        return self.invert(FT_image, full=full)

    def trans(self, u, v):
        return np.exp(-1j*2*np.pi*(u*self.f_x + v*self.f_y))
#         return np.exp(-1j*2*np.pi*(u/self.pe.N_X*self.f_x + v/self.pe.N_Y*self.f_y))

    def translate(self, image, vec, preshift=True):
        """
        Translate image by vec (in pixels)

        Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

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

    def retina(self, df=.07, sigma=.5):
        """
        A parametric description of the envelope of retinal processsing.
        See http://blog.invibe.net/posts/2015-05-21-a-simple-pre-processing-filter-for-image-processing.html
        for more information.

        In digital images, some of the energy in Fourier space is concentrated outside the
        disk corresponding to the Nyquist frequency. Let's design a filter with:

            - a sharp cut-off for radial frequencies higher than the Nyquist frequency,
            - times a smooth but sharp transition (implemented with a decaying exponential),
            - times a high-pass filter designed by one minus a gaussian blur.

        This filter is rotation invariant.

        Note that this filter is defined by two parameters:
            - one for scaling the smoothness of the transition in the high-frequency range,
            - one for the characteristic length of the high-pass filter.

        The first is defined relative to the Nyquist frequency (in absolute values) while the second
        is relative to the size of the image in pixels and is given in number of pixels.
        """
        # removing high frequencies in the corners
        env = (1-np.exp((self.f-.5)/(.5*df)))*(self.f<.5)
        # removing low frequencies
        env *= 1-np.exp(-.5*(self.f**2)/((sigma/self.pe.N_X)**2))
        return env

    def olshausen_whitening_filt(self):
        """
        Returns the whitening filter used by (Olshausen, 98)

        """
        power_spectrum =  self.f**(-self.pe.white_alpha*2)
        power_spectrum /= np.mean(power_spectrum)
        K_ols = (power_spectrum)**-.5
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
        in absolute coordinates : f_0 = 200 / 512 / 2

        steepness is to produce a "fairly sharp cutoff"

        """
        if f_0==0:
            return 1
        else:
            return np.exp(-(self.f/f_0)**steepness)

    def power_spectrum(self, image):
        return fftshift(np.absolute(fft2(image))**2)

    def whitening_filt(self, recompute=False):
        """
        Returns the envelope of the whitening filter.

        if we chose one based on structural assumptions (``struct=True``)
            then we return a 1/f spectrum based on the assumption that the structure of images
            is self-similar and thus that the Fourier spectrum scales a priori in 1/f.

        elif we chose to learn,
            returns theaverage correlation filter in FT space.

            Computes the average power spectrum = FT of cross-correlation, the mean decorrelation
            is given for instance by (Attick, 92).

        else
            we return the parametrization based on Olshausen, 1996

        """
        if self.pe.white_n_learning>0:
            try:
                K = np.load(os.path.join(self.pe.matpath, 'white'+ str(self.pe.N_X) + '-' + str(self.pe.N_Y) + '.npy'))
                if recompute:
                    raise('Recomputing the whitening filter')
            except:
                print(' Learning the whitening filter')
                power_spectrum = 0. # power spectrum
                for i_learning in range(self.pe.white_n_learning):
                    image, filename, croparea = self.patch(self.pe.white_name_database, verbose=False)
                    power_spectrum += self.power_spectrum(image)
                power_spectrum /= np.mean(power_spectrum)

                # formula from Atick:
#                 M = np.sqrt(power_spectrum / (self.pe.white_N**2 + power_spectrum))# * self.low_pass(f_0=self.pe.white_f_0, alpha=self.pe.white_alpha)
#                 K = M / np.sqrt(M**2 * (self.pe.white_N**2 + power_spectrum) + self.pe.white_N_0**2)
                K = (self.pe.white_N**2 + power_spectrum)**-.5
                K *= self.low_pass(f_0 = self.pe.white_f_0, steepness = self.pe.white_steepness)
                K /= np.mean(K) # normalize energy :  DC is one <=> xcorr(0) = 1
                self.mkdir()
                np.save(os.path.join(self.pe.matpath, 'white'+ str(self.pe.N_X) + '-' + str(self.pe.N_Y) + '.npy'), K)
        else:
            K = self.olshausen_whitening_filt()
        return K

    def preprocess(self, image):
        """
        Returns the pre-processed image

        From raw pixelized images, we want to keep information that is relevent to the content of
        the objects in the image. In particular, we want to avoid:

            - information that would not be uniformly distributed when rotating the image. In
            particular, we discard information outside the unit disk in Fourier space, in particular
            above the Nyquist frequency,
            - information that relates to information of the order the size of the image. This
            involves discarding information at low-level frequencies.

        See http://blog.invibe.net/posts/2015-05-21-a-simple-pre-processing-filter-for-image-processing.html
        for more information.
        """
        return self.FTfilter(image, self.f_mask)


    def whitening(self, image):
        """
        Returns the whitened image
        """
        K = self.whitening_filt()
        return self.FTfilter(image, K)

    def dewhitening(self, white):
        """
        Returns the dewhitened image

        """
        K = self.whitening_filt()
        return self.FTfilter(white, 1./K)

    def hist_radial_frequency(self, FT, N_f = 20):
        """
        A simple function to compute a radial histogram in different spatial frequency bands.

        """
         #F.shape[0]/2 # making an histogram with N_f bins
        f_bins = np.linspace(0., 0.5, N_f+1)
        f_bins = np.logspace(-2., 0, N_f+1, base=10)*0.5

        N_orientations = 24 # making an histogram with N_f bins
        theta_bins = np.linspace(0, np.pi, N_orientations, endpoint=False)

        F_rot = np.zeros((N_f, N_orientations))
        for i_theta in range(N_orientations):
            for i_f in range(N_f):
                f_slice = (f_bins[i_f] < self.f) *  ( self.f < f_bins[i_f+1])
                theta_slice = np.exp(np.cos(self.f_theta - theta_bins[i_theta])/(1.5*2*np.pi/N_orientations)**2)
                F_rot[i_f, i_theta] = (f_slice * theta_slice * FT).sum()
                F_rot[i_f, i_theta] /= (f_slice * theta_slice).sum() # normalize by the integration area (numeric)
        if np.isnan(F_rot).any(): print('Beware of the NaNs!')
        F_rot /= F_rot.max()
        return f_bins, theta_bins, F_rot

    def imshow(self, image, fig=None, ax=None, cmap=plt.cm.gray, axis=False, norm=True, center=True,
            xlabel='Y axis', ylabel='X axis', figsize=(8, 8), mask=False, vmin=-1, vmax=1):
        """
        Plotting routine to show an image

        Place the [0,0] index of the array in the upper left  corner of the axes. Data limits for the axes. The default assigns zero-based row, column indices to the x, y centers of the pixels.
        Note that the convention for coordinates follows that of matrices: the origin is at the top left of the image, and coordinates are first the rows (vertical axis, going down) then the columns (horizontal axis, going right).

        """

        if fig is None: fig = plt.figure(figsize=(self.pe.figsize_edges*self.pe.N_Y/self.pe.N_X, self.pe.figsize_edges))
        if ax is None: ax = fig.add_subplot(111)
        if norm: image = self.normalize(image, center=True, use_max=True)
        ax.pcolor(image, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        if not(axis):
            plt.setp(ax, xticks=[], yticks=[])
        else:
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
        ax.axis([0, self.pe.N_Y-1, self.pe.N_X-1, 0])
        if mask:
            linewidth_mask = 1 # HACK
            circ = plt.Circle((.5*self.pe.N_Y, .5*self.pe.N_Y), radius=0.5*self.pe.N_Y-linewidth_mask/2., fill=False, facecolor='none', edgecolor = 'black', alpha = 0.5, ls='dashed', lw=linewidth_mask)
            ax.add_patch(circ)
        return fig, ax

    def show_image_FT(self, image, FT_image, fig=None, figsize=(14, 14/2.), a1=None, a2=None, axis=False,
            title=True, FT_title='Spectrum', im_title='Image', norm=True,
            vmin=-1., vmax=1.):
        if fig is None: fig = plt.figure(figsize=figsize)
        if a1 is None: a1 = fig.add_subplot(121)
        if a2 is None: a2 = fig.add_subplot(122)
        fig, a1 = self.imshow(np.absolute(FT_image)/np.absolute(FT_image).max()*2-1, fig=fig, ax=a1, cmap=plt.cm.hot, norm=norm, axis=axis, vmin=vmin, vmax=vmax)
        fig, a2 = self.imshow(image, fig=fig, ax=a2, cmap=plt.cm.gray, norm=norm, axis=axis, vmin=vmin, vmax=vmax)
        if title:
            plt.setp(a1, title='Spectrum')
            plt.setp(a2, title='Image')
        if not(axis):
            plt.setp(a1, xticks=[self.pe.N_X/2], yticks=[self.pe.N_Y/2], xticklabels=[''], yticklabels=[''])
            plt.setp(a2, xticks=[], yticks=[])
        else:
            plt.setp(a1, xticks=[self.pe.N_X/2], yticks=[self.pe.N_Y/2], xticklabels=['0.'], yticklabels=['0.'])
            plt.setp(a2, xticks=np.linspace(0, self.pe.N_X, 5), yticks=np.linspace(0, self.pe.N_Y, 5))
            plt.setp(a1, xlabel=r'$f_x$', ylabel=r'$f_y$')
            plt.setp(a2, xlabel=r'$f_x$', ylabel=r'$f_y$')

        a1.axis('equal')#[0, self.pe.N_X-1, self.pe.N_Y-1, 0])
        a2.axis('equal')#[0, self.pe.N_X-1, self.pe.N_Y-1, 0])
        return fig, a1, a2

    def show_FT(self, FT_image, fig=None, figsize=(14, 14/2), a1=None, a2=None, axis=False,
            title=True, FT_title='Spectrum', im_title='Image', norm=True, vmin=-1., vmax=1.):
        image = self.invert(FT_image)#, phase=phase)
        fig, a1, a2 = self.show_image_FT(image, FT_image, fig=fig, figsize=figsize, a1=a1, a2=a2, axis=axis,
                                    title=title, FT_title=FT_title, im_title=im_title, norm=norm, vmin=vmin, vmax=vmax)
        return fig, a1, a2

    def show_spectrum(self, image, fig=None, figsize=(14, 14/2), a1=None, a2=None, axis=False,
            title=True, FT_title='Spectrum', im_title='Image', norm=True, vmin=-1., vmax=1.):
        FT_image = np.absolute(self.fourier(image, full=False))
        fig, a1, a2 = self.show_image_FT(image, FT_image , fig=fig, figsize=figsize, a1=a1, a2=a2, axis=axis,
                                    title=title, FT_title=FT_title, im_title=im_title, norm=norm, vmin=vmin, vmax=vmax)
        return fig, a1, a2

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
    im = Image('database/gris512.png')
