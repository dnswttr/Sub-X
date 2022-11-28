#########################################################################
#
#	Sub-X Main
#
#
#	INPUT (required):
#	
#	 - fname: name of the input fits file, without ".fits"
#			  eg: file.fits -> fname = "file"
#
#	 - path: path to the input file
#
#	 - opath: path to the output directory
#
#	INPUT (optional):
#
#	 - rmask: name of the file that contains the regions to which the 
#			  computation is restricted (must have been produced with ds9)
#			  without ".reg", eg: region.reg -> rmask = "region"
#			  default: "none"
#
#	 - mlim: set a minimum threshold by hand instead of computing
#		     it from the data
#			 default: 0
#
#	 - layer: if multiple maps are stored within one fits-file, use layer
#			  to access a specific one, e.g. if file.fits contains data of
#			  size x*y*z = 200*200*4, here 200 are the image sizes 
#			  and 4 are four different frequencies, then set layer to 
#			  0 to access the first map, 1 to access the second map etc
#			  default: -1
#
#	 - sigin: set the standard deviation of the Gaussian smoothing by hand
#			  default: -1
#
#
#	OUTPUT: Sub-X has different output for 2D and 3D data:
#
#	2D: for an input file called "input.fits" the outputs are:
#
#		- input_*_contour_values.dat: contains the contour values
#
#		- input_*.fits: contains fits file with 8 slices:
#						1: original image
#						2: image with values below lim removed
#						3: euclidean distance transform
#						4: smoothed image
#						5: laplace transform
#						6: only pixels above cmax are selected
#						7: only pixels above cmean are selected
#						8: final output with all extracted substructures
#
#	3D:	for an input file called "input.fits" the outputs are:
#       - input_*_contour_values.dat: contains contour values
#
#       - input_*_distance.fits: contains euclidean distance transform
#
#       - input_*_gauss.fits: contains smooth image
#
#       - input_*_laplace.fits: contains laplace transform
#
#       - input_*_tsad.fits: only pixels above cmean are selected
#
#       - input_*_tags.fits:  final output with all extracted substructures
#
#	For both the output files contain the "*" in their name, "*" encodes some of the input
#    parameters and parameters used during the computation, such as the standard deviation 
#	 of the Gaussian etc. In total a file name can look like this:
#
#			input_layer_1_reg_masked_pmin_10_sig_2.fits
#
#	 This file was generated from input.fits, used the second slice of the data cube,
#	  restricted the extraction to a specific region, to compute the gaussian std all 
#	  cells below value 10 have been removed, and the standard deviation of the Gaussian is 2
#
#
#########################################################################

import subX


fname = 'MK_5'
path = './'
opath = './'
subX.subX(fname, path, opath)


