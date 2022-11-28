############################################################
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation 
from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure
from quantimpy import minkowski as mk
from astropy.io import fits

import numpy as np

###########################################################

def my_write_numpy_arr(data, fname):
	np.save(fname, data)

###########################################################

def my_readfits(path):
	HDU = fits.open(path)
	im = HDU[0].data
	header = HDU[0].header
	naxis = header[2]
	sd = np.zeros(naxis)
	for i in range(naxis):
		sd[i] = np.size(im,i)
	if naxis == 2: x = np.arange(sd[0])
	if naxis == 2: y = np.arange(sd[1])
	if naxis == 3: x = np.arange(sd[1])
	if naxis == 3: y = np.arange(sd[2])
	return header, im, sd, x, y

###################################################################################################

def mymink(im, res):
	ndim = im.ndim
	if ndim == 2: mink = mk.functionals(im, [res,res])
	if ndim == 3: mink = mk.functionals(im, [res,res,res])
	return mink

###################################################################################################

def minkM_to_pf(M):
	M0 = M[0]
	M1 = M[1]
	M2 = M[2]
	M3 = M[3]
	r1 = 3*M0 / (8*M1)
	r2 = 4*M1 / (np.pi**2*M2)
	r3 = 3*M2 / (8*M3)
	r = np.sort([r1,r2,r3])
	t = r[0]
	w = r[1]
	l = r[2]
	p = (w-t) / (w+t)
	f = (l-w) / (l+w)
	return p, f

###################################################################################################

def minkM_to_pf_sheth(M):
	M0 = M[0]
	M1 = M[1]
	M2 = M[2]
	M3 = M[3]

	v = M0
	s = 8 * M1
	c = 2 * np.pi**2 * M2
	e = 4*np.pi * M3 / 3
	g = 1 - e/2

	r1 = 3  * v  / s
	r2 = s/c 
	r3 = c / (4*np.pi * ( 1 + g))
	r = np.sort([r1,r2,r3])

	t = r[0]
	w = r[1]
	l = r[2]

	p = (w-t) / (w+t)
	f = (l-w) / (l+w)

	return p, f

###################################################################################################

def minkM_to_pf_bag(M):
	M0 = M[0]
	M1 = M[1]
	M2 = M[2]
	M3 = M[3]

	v = M0
	s = 8 * M1
	c = 2 * np.pi**2 * M2
	e = 4*np.pi * M3 / 3
	g = 1 - e

	r1 = 3  * v  / s
	r2 = s/c 
	r3 = c / (4*np.pi) 
	r = np.sort([r1,r2,r3])

	t = r[0]
	w = r[1]
	l = r[2]

	p = (w-t) / (w+t)
	f = (l-w) / (l+w)

	return p, f


###################################################################################################

def minkM_to_f_2D(M):
	M0 = M[0]
	M1 = M[1]
	M2 = M[2]
	A = M0
	P = 2*np.pi*M1
	f = (P**2-4*np.pi*A)/(P**2+4*np.pi*A)
	return  f


###################################################################################################

def loop_mink(arr, res):
	print("Starting to compute Minkowski functions")
	ndim = arr.ndim
	max_arr = int(np.max(arr))
	print(max_arr)

	M0 = np.zeros(max_arr)
	M1 = np.zeros(max_arr)
	M2 = np.zeros(max_arr)
	M3 = np.zeros(max_arr)

	for i in range(1,max_arr+1):
		print("region " + str(i) + " of " + str(max_arr))
		if ndim == 3:
			im = np.zeros([np.size(arr, 0)+2,np.size(arr, 1)+2,np.size(arr, 2)+2], dtype = bool)
		elif ndim == 2:
			im = np.zeros([np.size(arr, 0)+2,np.size(arr, 1)+2], dtype = bool)

		ids = np.where(arr == i)
		im[ids] = True

		if ndim == 3:
			mink = mymink(im, res)
			M0[i-1] = mink[0]
			M1[i-1] = mink[1]
			M2[i-1] = mink[2]
			M3[i-1] = mink[3]
		elif ndim == 2:
			mink = mymink(im, res)
			M0[i-1] = mink[0]
			M1[i-1] = mink[1]
			M2[i-1] = mink[2]


	print("Ending to compute Minkowski functions \n")
	if ndim == 2: return M0, M1, M2
	if ndim == 3: return M0, M1, M2, M3


###################################################################################################

def loop_mink_fill(arr, res, rad, mag, mach):
	print("Starting to compute Minkowski functions with filled holes")
	ndim = arr.ndim
	max_arr = int(np.max(arr))

	M0 = np.zeros(max_arr)
	M1 = np.zeros(max_arr)
	M2 = np.zeros(max_arr)
	M3 = np.zeros(max_arr)
	p14 = np.zeros(max_arr)
	size = np.zeros(max_arr)
	mmag = np.zeros(max_arr)
	smag = np.zeros(max_arr)
	mmach = np.zeros(max_arr)
	smach = np.zeros(max_arr)

	if ndim == 3:
		dims = [np.size(arr, 0)+2,np.size(arr, 1)+2,np.size(arr, 2)+2]
		arr2 = np.zeros(dims)
		arr2[1:-1,1:-1,1:-1] = arr

		rad2 = np.zeros(dims)
		rad2[1:-1,1:-1,1:-1] = rad

		mag2 = np.zeros(dims)
		mag2[1:-1,1:-1,1:-1] = mag

		mach2 = np.zeros(dims)
		mach2[1:-1,1:-1,1:-1] = mach

	elif ndim == 2:
		dims = [np.size(arr, 0)+2,np.size(arr, 1)+2]

	for i in range(1,max_arr+1):
		print("region " + str(i) + " of " + str(max_arr))

		if ndim == 3:
			im = np.zeros(dims, dtype = bool)
			im[:,:,:] = False
			ims = np.zeros(dims)

		elif ndim == 2:
			ims = np.zeros(dims)

		ids = np.where(arr2 == i)
		im[ids] = True


		if ndim == 2:
			im = binary_fill_holes(ims)
			mink = mymink(im, res)
			M0[i-1] = mink[0]
			M1[i-1] = mink[1]
			M2[i-1] = mink[2]

		elif ndim == 3: 
			mink = mymink(im, res)
			M0[i-1] = mink[0]
			M1[i-1] = mink[1]
			M2[i-1] = mink[2]
			M3[i-1] = mink[3]
			p14[i-1] = np.sum(rad2[im == True])
			size[i-1] = mink[0]
			mmag[i-1] = np.mean(mag2[im == True])
			smag[i-1] = np.std(mag2[im == True])
			mmach[i-1] = np.mean(mach2[im == True])
			smach[i-1] = np.std(mach2[im == True])

			if M3[i-1] != 0.238732414637843:
				print('ERR: ', p14 [i-1])
				im = binary_fill_holes(im)
				mink = mymink(im, res)
				M0[i-1] = mink[0]
				M1[i-1] = mink[1]
				M2[i-1] = mink[2]
				M3[i-1] = mink[3]
				p14[i-1] = np.sum(rad2[im == True])
				size[i-1] = mink[0]
				mmag[i-1] = np.mean(mag2[im == True])
				smag[i-1] = np.std(mag2[im == True])
				mmach[i-1] = np.mean(mach2[im == True])
				smach[i-1] = np.std(mach2[im == True])
				print('ERR: ', p14 [i-1])

			st1 = generate_binary_structure(3, 2)
			while M3[i-1] != 0.238732414637843:
				im = binary_erosion(binary_dilation(im, structure=st1).astype(bool)).astype(bool)
				mink = mymink(im, res)
				M0[i-1] = mink[0]
				M1[i-1] = mink[1]
				M2[i-1] = mink[2]
				M3[i-1] = mink[3]
				p14[i-1] = np.sum(rad2[im == True])
				size[i-1] = mink[0]
				mmag[i-1] = np.mean(mag2[im == True])
				smag[i-1] = np.std(mag2[im == True])
				mmach[i-1] = np.mean(mach2[im == True])
				smach[i-1] = np.std(mach2[im == True])
				print('ERR: ', p14 [i-1])

####
				if mink[3] != 0.238732414637843:
					print('ERR: ', mink[3], i)
###

	print("Ending to compute Minkowski functions \n")
	if ndim == 2: return M0, M1, M2
	if ndim == 3: return M0, M1, M2, M3, p14, size, mmag, smag, mmach, smach




###################################################################################################

def loop_pf(M0,M1,M2,M3):
	print("Starting to compute P and F")
	sd = np.size(M0,0)
	p = np.zeros(sd)
	f = np.zeros(sd)

	psheth = np.zeros(sd)
	fsheth = np.zeros(sd)

	pbag = np.zeros(sd)
	fbag = np.zeros(sd)

	for i in range(0,sd):
		print("region " + str(i+1) + " of " + str(sd))
		mtemp = [M0[i], M1[i], M2[i], M3[i]]
		ptemp, ftemp = minkM_to_pf(mtemp)
		pstemp, fstemp = minkM_to_pf_sheth(mtemp)
		pbtemp, fbtemp = minkM_to_pf_bag(mtemp)
		p[i] = ptemp
		f[i] = ftemp
		psheth[i] = pstemp
		fsheth[i] = fstemp
		pbag[i] = pbtemp
		fbag[i] = fbtemp

	print("Ending to compute P and F \n")
	return p, f, psheth, fsheth, pbag, fbag

###################################################################################################

def loop_f_2D(M0,M1,M2):
	print("Starting to compute F")
	sd = np.size(M0,0)
	f = np.zeros(sd)

	for i in range(0,sd):
		print("region " + str(i+1) + " of " + str(sd))
		mtemp = [M0[i], M1[i], M2[i]]
		ftemp = minkM_to_f_2D(mtemp)
		f[i] = ftemp

	print("Ending to compute F \n")
	return f

###################################################################################################

def get_radio_power_and_size(arr, tags):
	max_arr = int(np.max(tags))
	p14 = np.zeros(max_arr)
	size = np.zeros(max_arr)

	for i in range(1,max_arr+1):
		print(i)
		ids = np.where(tags == i)
		p14[i-1] = np.sum(arr[ids])
		size[i-1] = np.size(ids)

	return p14, size



###################################################################################################

def mink2D(path_tags, path_maps, opath, file1, label, layer = -1):
	print("Reading input \n")
	head1, im1, sd1, x1, y1 = my_readfits(path_maps + file1 + '.fits')
	if (layer > -1) & (np.size(sd1) > 2):
		im1 = im1[layer,:,:]
		sd1 = sd1[1:3]

    # tag regions
	headt, tags, sdt, xt, yt = my_readfits(path_tags+label + '.fits')
	tags = tags[-1,:,:]
	sdt = sdt[1:3]

	print("We have " + str(np.max(tags)) + " independent regions \n")

    # get radio power and size
	p14, size = get_radio_power_and_size(im1, tags)
	my_write_numpy_arr(p14, opath+'mink_p14_' + label)
	my_write_numpy_arr(size, opath+'mink_size_' + label)

    # compute MINK
	M0,M1,M2 = loop_mink(tags,  1)
	my_write_numpy_arr(M0, opath+'mink_M0_' + label)
	my_write_numpy_arr(M1, opath+'mink_M1_' + label)
	my_write_numpy_arr(M2, opath+'mink_M2_' + label)

    # compute F and T 
	f = loop_f_2D(M0,M1,M2)

    # save data
	my_write_numpy_arr(f,opath + 'mink_filamentary_' + label)

###################################################################################################


def mink3D(files, sig = 0):

	bpath = files[0] 
	bname = files[1] 
	mpath = files[2] 
	mname = files[3] 
	tpath = files[4] 
	tname = files[5] 
	ppath = files[6] 
	pname = files[7] 
	opath = files[8]

	label = tname

	print("Reading input \n")
	headp, rad, sdr, xr, yr = my_readfits(ppath+pname+'.fits')
	headp, tags, sdt, xt, yt = my_readfits(tpath+tname+'.fits')
	headb, mag, sdb, xb, yb = my_readfits(bpath+bname+'.fits')
	headm, mach, sdm, xm, ym = my_readfits(mpath+mname+'.fits')

	print("We have " + str(np.max(tags)) + " independent regions \n")

	if sig == 0:
		# compute MINK
		M0,M1,M2,M3 = loop_mink(tags,  1)
		my_write_numpy_arr(M0, opath + 'mink_M0_' + label)
		my_write_numpy_arr(M1, opath + 'mink_M1_' + label)
		my_write_numpy_arr(M2, opath + 'mink_M2_' + label)
		my_write_numpy_arr(M3, opath + 'mink_M3_' + label)

		# compute F and T 
		p_seta, f_seta, p_sheth, f_sheth, p_bag, f_bag = loop_pf(M0,M1,M2,M3)
		my_write_numpy_arr(f_seta, opath + 'mink_filamentary_seta_' + label)
		my_write_numpy_arr(p_seta, opath + 'mink_planarity_seta_' + label)
		my_write_numpy_arr(f_bag, opath + 'mink_filamentary_bag_' + label)
		my_write_numpy_arr(p_bag, opath + 'mink_planarity_bag_' + label)


	elif sig < 0:
		# compute MINK
		M0, M1, M2, M3, p14, size, mmag, smag, mmach, smach = loop_mink_fill(tags,  1, rad, mag, mach)
		my_write_numpy_arr(M0, opath + 'mink_M0_fill_' + label)
		my_write_numpy_arr(M1, opath + 'mink_M1_fill_' + label)
		my_write_numpy_arr(M2, opath + 'mink_M2_fill_' + label)
		my_write_numpy_arr(M3, opath + 'mink_M3_fill_' + label)
		my_write_numpy_arr(p14, opath + 'mink_p14_fill_' + label)
		my_write_numpy_arr(size, opath + 'mink_size_fill_' + label)
		my_write_numpy_arr(mmach, opath + 'mink_mach_avg_fill_' + label)
		my_write_numpy_arr(smach, opath + 'mink_mach_std_fill_' + label)
		my_write_numpy_arr(mmag, opath + 'mink_mag_avg_fill_' + label)
		my_write_numpy_arr(smag, opath + 'mink_mag_std_fill_' + label)

		# compute F and T 
		p_seta, f_seta, p_sheth, f_sheth, p_bag, f_bag = loop_pf(M0,M1,M2,M3)
		my_write_numpy_arr(f_seta, opath + 'mink_filamentary_seta_fill_' + label)
		my_write_numpy_arr(p_seta, opath + 'mink_planarity_seta_fill_' + label)
		my_write_numpy_arr(f_bag, opath + 'mink_filamentary_bag_fill_' + label)
		my_write_numpy_arr(p_bag, opath + 'mink_planarity_bag_fill_' + label)


	
