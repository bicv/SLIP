{
# Image
'N_X' : 256, # size of images
'N_Y' : 256, # size of images
'do_mask'  : False, # used forinstance in the Matching Pursuit algorithm self.pe.do_mask
'figpath' : 'figures/',
'matpath' : 'mat/',
'datapath' : '../AssoField/database/',
'ext' : '.pdf',
'seed': None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None or a given number to freeze the RNG
'N_image': None, # number of images to pick in a database, set to None if you want to pick all in the database
# whitening parameters:
'name_database' : 'serre07_distractors',
'n_learning' : 400,
'N' : .6,
'N_0' : .0, # olshausen = 0.
'f_0' : .4, # olshausen = 0.2
'alpha' : 1.15,
'steepness' : 4.,
'recompute' : False,
'learn' : False,
}
