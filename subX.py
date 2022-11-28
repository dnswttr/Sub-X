import numpy as np
import pyregion as pyreg
from scipy import ndimage 
from os.path import exists
from astropy.io import fits



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


def my_writefits(fname, data):
	hdu = fits.PrimaryHDU(data)
	hdul = fits.HDUList([hdu])
	hdul.writeto(fname + '.fits', overwrite = True)

def my_write_dat_arr(data, fname):
	np.savetxt(fname + '.dat', data)




def detect_lim(arr, lim):
	print('Start the detection limit cut')
	arr[arr <= lim] = 0
	print('Done')
	return arr


def single_cells(arr):
	print('Start to remove all cells without a direct neighbour')

	ndim = arr.ndim

	if ndim == 2:
		ids = np.where(
			((arr > 0) & (np.roll(arr, 1, axis = 0) == 0)) &
			((arr > 0) & (np.roll(arr, -1, axis = 0) == 0)) &
			((arr > 0) & (np.roll(arr, 1, axis = 1) == 0)) &
			((arr > 0) & (np.roll(arr, -1, axis = 1) == 0))
		)
		arr[ids] = 0

	elif ndim == 3:
		ids = np.where(
			((arr > 0) & (np.roll(arr, 1, axis = 0) == 0)) &
			((arr > 0) & (np.roll(arr, -1, axis = 0) == 0)) &
			((arr > 0) & (np.roll(arr, 1, axis = 1) == 0)) &
			((arr > 0) & (np.roll(arr, -1, axis = 1) == 0)) &
			((arr > 0) & (np.roll(arr, 1, axis = 2) == 0)) &
			((arr > 0) & (np.roll(arr, -1, axis = 2) == 0))
		)
		arr[ids] = 0

	print('Done')
	return arr

def reg_mask(head1,path, rmask, dims):
	print('Start to create the region mask from')
	print(rmask)

	if np.size(rmask) == 1:
		if exists(path + rmask + '.reg') == True:
			r = pyreg.open(path + rmask + '.reg').as_imagecoord(head1)
			mask = r.get_mask(shape=(dims[0], dims[1]))
		else: 
			print('No region file was found \n now you have two options:')
			print('1) provide the file of the region mask')
			print('2) do not pass rmask to the subX function')
			print('Good Bye')
			exit()

	else:
		if (np.size(rmask) == 3) & (np.size(dims) == 3) :
			mask = np.zeros(dims)+1

			if ( exists(path + rmask[0] + '.reg') == True) & ( exists(path + rmask[1] + '.reg') == True) & ( exists(path + rmask[2] + '.reg') == True):
				for i in [0,1,2]:
					if exists(path + rmask[i] + '.reg') == True:
						r = pyreg.open(path + rmask[i] + '.reg').as_imagecoord(head1)
						xd, yd = 0,0

						if i == 0:
							xd = dims[1]
							yd = dims[2]
						elif i == 1:
							xd = dims[0]
							yd = dims[2]
						else:
							xd = dims[0]
							yd = dims[1]

						j = dims[i]
						mask2 = r.get_mask(shape=(xd,yd))
						mask2[mask2 == False] = 0
						mask2[mask2 == True] = 1

						if i == 0:
							for k in range(j):
								mask[:,:,k] *= mask2
	
						elif i == 1:
							for k in range(j):
								mask[:,k,:] *= mask2

						elif i == 2:
							for k in range(j):
								mask[k,:,:] *= mask2

				mask = np.array(mask,dtype = bool)
				
			else:
				print('No region file was found \n continue withour region mask')

		else:
			print('The number of dimensions and the number of regions mask are not equal \n (and at least one is not 3) \n continue without region mask')

	print('Done')
	return mask

###################################################################################################

def tag_regions_2D(arr, dims):
	print('Start to assign tags to maximum regions 2D')

	tags = np.zeros([dims[0]+2,dims[1]+2])
	tags[1:-1,1:-1] = arr

	tags[tags > 0] = -1

	found = True
	reg = 1
	while found == True:
		ids = np.where(tags == -1)
		tags[ids[0][0],ids[1][0]] = reg

		found2 = True
		while found2 == True:
			ids2 = np.where(
				(tags == -1) & 
				((np.roll(tags, 1, axis = 0) == reg) |
				(np.roll(tags, -1, axis = 0) == reg) |
				(np.roll(tags, 1, axis = 1) == reg) |
				(np.roll(tags, -1, axis = 1) == reg)) 
			)
			# | (np.roll(tags, 1, axis = 1) == reg) | (np.roll(tags, -1, axis = 1) == reg))
			if np.size(ids2) == 0: found2 = False
			else: tags[ids2] = reg
		ids = np.where(tags == -1)
		reg = reg + 1
		if np.size(ids) == 0: found = False
	tags = tags[1:-1,1:-1]

	print('Done')
	return tags


###################################################################################################

def tag_regions_3D(arr, dims):
	print('Start to assign tags to maximum regions 3D')

	numx = int(dims[0]/50)+1
	numy = int(dims[1]/50)+1
	numz = int(dims[2]/50)+1

	ranx = np.arange(numx)*50
	ranx[-1] = np.min([ranx[-1],dims[0]])
	ranx[0] = 1
	dx = np.arange(numx-1)
	dx[:] = ranx[1:] - ranx[0:-1] + 3

	rany = np.arange(numy)*50
	rany[-1] = np.min([rany[-1],dims[1]])
	rany[0] = 1
	dy = np.arange(numy-1)
	dy[:] = rany[1:] - rany[0:-1] + 3

	ranz = np.arange(numz)*50
	ranz[-1] = np.min([ranz[-1],dims[2]])
	ranz[0] = 1
	dz = np.arange(numz-1)
	dz[:] = ranz[1:] - ranz[0:-1] + 3

	xdim = int(dims[0])
	ydim = int(dims[1])
	zdim = int(dims[2])

	tags = np.zeros([xdim,ydim,zdim])

	tags[:,:,:] = arr
	tags[tags > 0] = -1

	found = True
	reg = 0

	while found == True:
		reg += 1
		if np.mod(reg, 100) == 0: print('REG: ', reg)

		ids = np.where(tags == -1)
		tags[ids[0][0],ids[1][0],ids[2][0]] = reg

		found4 = True
	
		while found4 == True:
			found4 = False


			for i in range(numx-1):
				xl = ranx[i]-1
				xr = ranx[i+1]
				ddx = dx[i]
				for j in range(numy-1):
					yl = rany[j]-1
					yr = rany[j+1]
					ddy = dy[j]
					for k in range(numz-1):
						zl = ranz[k]-1
						zr = ranz[k+1]
						ddz = dz[k]

						temp = np.zeros([ddx,ddy,ddz])
						temp[1:-1,1:-1,1:-1] = tags[xl:xr,yl:yr,zl:zr]

						found2 = False
						if reg in temp: found2 = True

						while found2 == True:
							found2 = False

							if True in (temp == -1) & (np.roll(temp, -1, axis = 0) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 0) == reg) ] = reg
								found2 = True
								found4 = True

							if True in (temp == -1) & (np.roll(temp, -1, axis = 1) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 1) == reg) ] = reg
								found2 = True
								found4 = True

							if True in (temp == -1) & (np.roll(temp, -1, axis = 2) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 2) == reg) ] = reg
								found2 = True
								found4 = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 0) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 0) == reg) ] = reg
								found2 = True
								found4 = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 1) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 1) == reg) ] = reg
								found2 = True
								found4 = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 2) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 2) == reg) ] = reg
								found2 = True
								found4 = True

							if found2 == False:
								tags[xl:xr,yl:yr,zl:zr] = temp[1:-1,1:-1,1:-1]
		
		if -1 not in tags: found = False

	print('Done')
	return tags
		


######################################################


def get_saddle_regions_3D(dsad,dims,tmax):
	print('Start to assign tags to saddle regions 3D')

	tsad = np.zeros([dims[0],dims[1],dims[2]])
	tsad[dsad > 0] = -1
	tsad[tmax > 0] = tmax[tmax > 0]

	imax = int(np.max(tmax))

	numx = int(dims[0]/50)+1
	numy = int(dims[1]/50)+1
	numz = int(dims[2]/50)+1

	ranx = np.arange(numx)*50
	ranx[-1] = np.min([ranx[-1],dims[0]])
	ranx[0] = 1
	dx = np.arange(numx-1)
	dx[:] = ranx[1:] - ranx[0:-1] + 3

	rany = np.arange(numy)*50
	rany[-1] = np.min([rany[-1],dims[1]])
	rany[0] = 1
	dy = np.arange(numy-1)
	dy[:] = rany[1:] - rany[0:-1] + 3

	ranz = np.arange(numz)*50
	ranz[-1] = np.min([ranz[-1],dims[2]])
	ranz[0] = 1
	dz = np.arange(numz-1)
	dz[:] = ranz[1:] - ranz[0:-1] + 3

	xdim = int(dims[0])
	ydim = int(dims[1])
	zdim = int(dims[2])

	for reg in range(1,imax+1):
		found = True
		if np.mod(reg, 100) == 0: print('REG: ', reg)

		while found == True:
			found = False

			for i in range(numx-1):
				xl = ranx[i]-1
				xr = ranx[i+1]
				ddx = dx[i]
				for j in range(numy-1):
					yl = rany[j]-1
					yr = rany[j+1]
					ddy = dy[j]
					for k in range(numz-1):
						zl = ranz[k]-1
						zr = ranz[k+1]
						ddz = dz[k]

						temp = np.zeros([ddx,ddy,ddz])
						temp[1:-1,1:-1,1:-1] = tsad[xl:xr,yl:yr,zl:zr]

						found2 = False
						if reg in temp: found2 = True

						while found2 == True:
							found2 = False
							if True in (temp == -1) & (np.roll(temp, -1, axis = 0) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 0) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp == -1) & (np.roll(temp, -1, axis = 1) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 1) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp == -1) & (np.roll(temp, -1, axis = 2) == reg):
								temp[(temp == -1) & (np.roll(temp, -1, axis = 2) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 0) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 0) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 1) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 1) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp == -1) & (np.roll(temp, 1, axis = 2) == reg):
								temp[(temp == -1) & (np.roll(temp, 1, axis = 2) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, -1, axis = 0) == reg):
								temp[(temp > reg) & (np.roll(temp, -1, axis = 0) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, -1, axis = 1) == reg):
								temp[(temp > reg) & (np.roll(temp, -1, axis = 1) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, -1, axis = 2) == reg):
								temp[(temp > reg) & (np.roll(temp, -1, axis = 2) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, 1, axis = 0) == reg):
								temp[(temp > reg) & (np.roll(temp, 1, axis = 0) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, 1, axis = 1) == reg):
								temp[(temp > reg) & (np.roll(temp, 1, axis = 1) == reg) ] = reg
								found2 = True
								found = True

							if True in (temp > reg) & (np.roll(temp, 1, axis = 2) == reg):
								temp[(temp > reg) & (np.roll(temp, 1, axis = 2) == reg) ] = reg
								found2 = True
								found = True

							if found2 == False:
								tsad[xl:xr,yl:yr,zl:zr] = temp[1:-1,1:-1,1:-1]



	ids = np.where(tsad == -1)

	if np.size(ids) > 0:
		tsad2 = np.zeros([dims[0],dims[1],dims[2]])
		dmax = np.zeros([dims[0],dims[1],dims[2]])

		dmax[ids] = -1

		tsad2 = tag_regions_3D(dmax, dims)

		tsad2[tsad2 > 0] = tsad2[tsad2 > 0] + np.max(tsad[tsad > 0])
		tsad[ids] = tsad2[ids] # cleaning saddle points that do not share a face with an other saddle point

	tsad[ids] = 0 # cleaning saddle points that do not share a face with an other saddle point
	print('Done')
	return tsad




######################################################

def get_saddle_regions_2D(dsad,dims,tmax):
	print('Start to assign tags to saddle regions 2D')
	tsad = np.zeros([dims[0],dims[1]])

	tsad[dsad > 0] = -1
	tsad[tmax > 0] = tmax[tmax > 0]
	
	imax = int(np.max(tmax))
	for i in range(1,imax+1):
		found = True
		while found == True:
			found = False

			ids2 = np.where(
				((tsad == -1) & (np.roll(tsad, 1, axis = 0) == i)) |
				((tsad == -1) & (np.roll(tsad, -1, axis = 0) == i)) |
				((tsad == -1) & (np.roll(tsad, 1, axis = 1) == i)) |
				((tsad == -1) & (np.roll(tsad, -1, axis = 1) == i)) 
			)
			if np.size(ids2) > 0:
				tsad[ids2] = i
				found = True

			ids2 = np.where(
				((tsad > i) & (np.roll(tsad, 1, axis = 0) == i)) |
				((tsad > i) & (np.roll(tsad, -1, axis = 0) == i)) |
				((tsad > i) & (np.roll(tsad, 1, axis = 1) == i)) |
				((tsad > i) & (np.roll(tsad, -1, axis = 1) == i)) 
			)
			if np.size(ids2) > 0:
				tsad[ids2] = i
				found = True



	ids = np.where(tsad == -1)

	if np.size(ids) > 0:
		tsad2 = np.zeros([dims[0],dims[1]])
		dmax = np.zeros([dims[0],dims[1]])

		dmax[ids] = -1

		tsad2 = tag_regions_2D(dmax, dims)

		tsad2[tsad2 > 0] = tsad2[tsad2 > 0] + np.max(tsad[tsad > 0])
		tsad[ids] = tsad2[ids] # cleaning saddle points that do not share a face with an other saddle point

	print('Done')
	return tsad

def find_nearest_saddle_point(sval, lim, dims, im1, tsad, name):
	print('Start to determine nearest saddle point')
	ndim = im1.ndim
	if ndim == 2:
		mask = np.zeros([dims[0],dims[1]])
		gradx = np.roll(im1, -1, axis = 0)-np.roll(im1, 1, axis = 0)
		grady = np.roll(im1, -1, axis = 1)-np.roll(im1, 1, axis = 1)
		mask[(im1 > lim ) & (im1 <= sval) & (tsad < 1)] = 1
		ids = np.where(mask == 1)
		nps = np.size(ids,1)

		for i in range(nps):

			found = False
			x = ids[0][i]
			y = ids[1][i]
			xn = x
			yn = y

			visited = np.array([x,y])
			if tsad[x,y] > 0:
				found = True
		
			while found == False:
				gx = gradx[xn,yn]
				gy = grady[xn,yn]
				if np.abs(gx) > np.abs(gy):
					if gx < 0:
						xn = xn-1
					else:
						xn = xn+1
				else:
					if gy < 0:
						yn = yn-1
					else:
						yn = yn+1

				if (xn < dims[0]) & (x >= 0):
					if (yn < dims[1]) & (y >= 0):
						if tsad[xn,yn] > 0:
							tsad[x,y] = tsad[xn,yn]
							mask[x,y] = 0
							found = True
					elif  (yn >= dims[1]) | (y < 0):
						mask[x,y] = 0
						found = True
				elif (xn >= dims[0]) | (x < 0):
					mask[x,y] = 0
					found = True

				if [xn,yn] in visited.tolist():
					found = True
				else:
					visited = np.vstack([visited, [xn,yn]])


    ###### END 2D, START 3D
	elif ndim == 3:
		mask = np.zeros([dims[0],dims[1],dims[2]])
		gradx = np.roll(im1, -1, axis = 0)-np.roll(im1, 1, axis = 0)
		grady = np.roll(im1, -1, axis = 1)-np.roll(im1, 1, axis = 1)
		gradz = np.roll(im1, -1, axis = 2)-np.roll(im1, 1, axis = 2)

		mask[(im1 > lim ) & (im1 <= sval)] = 1
		ids = np.where(mask == 1)
		nps = np.size(ids,1)

		for i in range(nps):

			found = False
			x = ids[0][i]
			y = ids[1][i]
			z = ids[2][i]
			xn = x
			yn = y
			zn = z

			visited = np.array([x,y,z])
			if tsad[x,y,z] > 0:
				found = True
		
			while found == False:
				gx = gradx[xn,yn,zn]
				gy = grady[xn,yn,zn]
				gz = gradz[xn,yn,zn]
				if (np.abs(gx) > np.abs(gy)) & (np.abs(gx) > np.abs(gz)):
					if gx < 0:
						xn = xn-1
					else:
						xn = xn+1
				elif (np.abs(gy) > np.abs(gx)) & (np.abs(gy) > np.abs(gz)):
					if gy < 0:
						yn = yn-1
					else:
						yn = yn+1
				else:
					if gz < 0:
						zn = zn-1
					else:
						zn = zn+1

				if (xn < dims[0]) & (x >= 0):
					if (yn < dims[1]) & (y >= 0):
						if (zn < dims[2]) & (z >= 0):
							if tsad[xn,yn,zn] > 0:
								tsad[x,y,z] = tsad[xn,yn,zn]
								mask[x,y,z] = 0
								found = True
						elif  (zn >= dims[2]) | (z < 0):
							mask[x,y,z] = 0
							found = True
					elif  (yn >= dims[1]) | (y < 0):
						mask[x,y,z] = 0
						found = True
				elif (xn >= dims[0]) | (x < 0):
					mask[x,y,z] = 0
					found = True

				if [xn,yn,zn] in visited.tolist():
					found = True
				else:
					visited = np.vstack([visited, [xn,yn,zn]])


	print('Done')
	return tsad

def reorder_ids(arr):
	print('Start to reorder ids')
	mval = int(np.max(arr))
	ids = np.zeros(mval)
	for i in range(1,mval+1):
		ids2 = np.where(arr == i)
		sd = np.size(ids2)
		if sd > 0:
			ids[i-1] = i
	ids2 = np.where(ids > 0)
	ids = ids[ids2]
	sd = np.size(ids)
	for i in  range(0,sd):
		#print(ids[i], ' -> ', i+1)
		ids2 = np.where(arr == ids[i])
		arr[ids2] = i+1
	print('Done')
	return arr

def get_contours(laplace, imax):
	if imax > 1e20:
		lap = np.array([np.mean(laplace[laplace > 0]), np.median(laplace[laplace > 0]), np.max(laplace[laplace > 0]), np.std(laplace[laplace > 0]/1e26)*1e26])
	else:
		lap = np.array([np.mean(laplace[laplace > 0]), np.median(laplace[laplace > 0]), np.max(laplace[laplace > 0]), np.std(laplace[laplace > 0])])
	con = np.zeros(3)
	con[0] =  np.max([lap[0], lap[1]])
	con[1] =  np.mean([lap[0], lap[1]])
	con[2] =  np.min([lap[0], lap[1]])
	print('cmax  = ', con[0])
	print('cmean = ', con[1])
	print('cmin  = ', con[2])
	return con


def subX(fname, path, opath, rmask = 'none', mlim = 0, layer = -1, sigin = -1):
	print('\n Starting to analyse file ' + fname)

	# read input file
	print('Reading file')
	head, im1, sd1, x, y = my_readfits(path+fname+'.fits')
	if (layer > -1) & (np.size(sd1) > 2):
		im1 = im1[layer,:,:]
		sd1 = sd1[1:3]  
	print('Done')

	# get dimensions
	ndim = im1.ndim
	if ndim == 2:
		dims = np.array([int(sd1[0]),int(sd1[1])])
		im2 = np.zeros([dims[0],dims[1]])
	elif ndim == 3:
		dims = np.array([int(sd1[0]),int(sd1[1]),int(sd1[2])])
		im2 = np.zeros([dims[0],dims[1],dims[2]])


	# generate name of output file

	name = fname
	if layer != -1:
		name = name + '_layer_' + str(layer)

	if rmask != 'none':
		name = name + '_reg_masked'

	# init output array
	idout = 0

	if ndim == 2:
		dout = np.zeros([8,dims[0],dims[1]])
		dout[idout,:,:] = im1 # 1
		idout += 1

	# prep data, i.e. create mask from contour file 
	if rmask != 'none':
		reg = reg_mask(head, path, rmask, dims)
	
	# get limit for MAT
	if mlim != 0:
		lim = mlim
	elif rmask != 'none':
		lim = np.mean(im1[(im1 > 0) & (reg == True)])
	else:
		lim = np.mean(im1[(im1 > 0)])

	nlim = str(lim)
	print('The detection limit is ' + nlim)
	name = name + '_pmin_' + nlim 


	im2[:,:] = im1[:,:]
	if rmask != 'none':
		im2[reg == False] = 0

	im2 = detect_lim(im2, lim)
	im2 = single_cells(im2)

	if ndim == 2:
		dout[idout,:,:] = im2 # 2
		idout += 1
	
	# medial axis transform (MAT)
	mat = ndimage.distance_transform_edt(im2)                                  

	# gauss smoothing
	im1[im1 == 0] = np.min(im1[im1 != 0])

	gauss = im1
	if sigin == -1:
		sig = int(np.max([np.ceil(np.median(mat[mat > 0]))/2.,2]))
	else:
		sig = sigin

	name = name + '_sig_' + str(sig) 
	gauss = ndimage.gaussian_filter(gauss, sigma = sig )
	print('The Gaussian std is ', sig )

	# output mat and gauss
	if ndim == 3:
		my_writefits(opath + name + '_distance' , mat)
		my_writefits(opath + name + '_gauss' , gauss)

	if ndim == 2:
		dout[idout,:,:] = mat # 3
		idout += 1
		dout[idout,:,:] = gauss # 4
		idout += 1

	# Laplace transform
	laplace = -1*ndimage.laplace(gauss)

	if rmask != 'none':
		laplace[reg == False] = 0


	if ndim == 3:
		my_writefits(opath + name + '_laplace' , laplace)

	if ndim == 2:
		dout[idout,:,:] = laplace # 5
		idout += 1

	if ndim == 2:
		dmax = np.zeros([dims[0],dims[1]])
	elif ndim == 3:
		dmax = np.zeros([dims[0],dims[1],dims[2]])

	con = get_contours(laplace, np.max(im1))
	my_write_dat_arr(con, opath + name + '_contour_values')

	ids = np.where(laplace > con[0])
	dmax[ids] = 1


	if ndim == 2:
		tmax = tag_regions_2D(dmax, dims)
		dout[idout,:,:] = tmax
		idout += 1
	elif ndim == 3:
		tmax = tag_regions_3D(dmax, dims)

	print(np.size(tmax > 0))

	if ndim == 2:
		dsad = np.zeros([dims[0],dims[1]])
	elif ndim == 3:
		dsad = np.zeros([dims[0],dims[1],dims[2]])

	ids = np.where(laplace > con[1])
	dsad[ids] = 1

	if ndim == 2:
		tsad = get_saddle_regions_2D(dsad,dims,tmax)
	elif ndim == 3:
		tsad = get_saddle_regions_3D(dsad,dims,tmax)

	if ndim == 2:
		dout[idout,:,:] = tsad # 6
		final1 = dout[idout,:,:]
		final1 = reorder_ids(final1)
		idout += 1
	elif ndim == 3:
		final1 = tsad
		final1 = reorder_ids(final1)
 
	tsadg =  find_nearest_saddle_point(con[1], con[2], dims, laplace, tsad, name)
 
	final2 = tsadg
	final2 = reorder_ids(final2)

	if ndim == 2:
		dout[idout,:,:] = final2 # 7
		idout += 1

	print('Writing file with tags to:')
	if ndim == 2:
		my_writefits(opath + name , dout)
		print(opath + name +'.fits')
	elif ndim == 3:
		my_writefits(opath + name + '_tags' , final2)
		print(opath + name + '_tags.fits')
	print('Good Bye!')

##################


