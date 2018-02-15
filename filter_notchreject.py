import ipcv
import numpy as np

def filter_notchreject(im, notchCenter, notchRadius, order=1, filterShape=ipcv.IPCV_IDEAL):

	"""
	Title:
		filter_notchreject
	Description:
		Creates a notch filter for given image and parameters
	Attributes:
		im - 2D ndarray signal to provide shape of filter
		notchCenter - center of the frequency to be blocked
		notchRadius - width of notch
		order - order of the filter (applicable only to Butterworth)
		filterShape - shape of frequency filter:
			- 0 = Ideal
			- 1 = Butterworth
			- 2 = Gaussian
	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(im) != np.ndarray:
		raise TypeError("Source image must be ndarray.")
	if type(notchCenter) != tuple or notchCenter[0] < 0 or type(notchCenter[0]) != int \
	or notchCenter[1] <0 or type(notchCenter[1]) != int:
		raise ValueError("Given notchCenter must be tuple of positive integers.")
	if type(notchRadius) != int or notchRadius < 0:
		raise ValueError("Given notch radius must be positive integer.")
	if type(order) != int or order < 0:
		raise ValueError("Given order must be positive integer.")
	if type(filterShape) != int or filterShape < 0 or filterShape >2:
		raise ValueError("Filter shape option limited to 0, 1, or 2.")

	src = im[:,:,0]
	M = src.shape[0]
	N = src.shape[1]
	D1 = np.empty(src.shape)
	D2 = np.copy(D1)
	H = np.ones(src.shape)

	for u in range(M): #rows
		for v in range(N): #columns
			D1[u,v] = (u-(M/2)-notchCenter[0])**2 + (v-(N/2)-notchCenter[1])**2
			D2[u,v] = (u-(M/2)+notchCenter[0])**2 + (v-(N/2)+notchCenter[1])**2
	D1 = np.sqrt(D1)
	D2 = np.sqrt(D2)

	if filterShape == 0: #ideal
		H[np.where(D1<=notchRadius)] = 0
		H[np.where(D2<=notchRadius)] = 0
		
	elif filterShape == 1: #butterworth
		H = 1/(1+(notchRadius**2/(D1*D2))**order)
	
	elif filterShape == 2: #gaussian
		H = 1-np.exp(-0.5*((D1*D2)/notchRadius**2))
	
	return(H)


if __name__ == '__main__':

	import cv2
	import ipcv
	import numpy
	import matplotlib.pyplot
	import matplotlib.cm
	import mpl_toolkits.mplot3d
	import os.path

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
	im = cv2.imread(filename)

	#frequencyFilter = ipcv.filter_notchreject(im,
						#(32,32),
						#10,
						#filterShape=ipcv.IPCV_IDEAL)
	frequencyFilter = ipcv.filter_notchreject(im,
						(32,32),
						10,
						order=2,
						filterShape=ipcv.IPCV_BUTTERWORTH)
	#frequencyFilter = ipcv.filter_notchreject(im,
						#(32,32),
						#10,
						#filterShape=ipcv.IPCV_GAUSSIAN)

	# Create a 3D plot and image visualization of the frequency domain filter
	rows = im.shape[0]
	columns = im.shape[1]
	u = numpy.arange(-columns/2, columns/2, 1)
	v = numpy.arange(-rows/2, rows/2, 1)
	u, v = numpy.meshgrid(u, v)

	figure = matplotlib.pyplot.figure('Frequency Domain Filter', (14, 6))
	p = figure.add_subplot(1, 2, 1, projection='3d')
	p.set_xlabel('u')
	p.set_xlim3d(-columns/2, columns/2)
	p.set_ylabel('v')
	p.set_ylim3d(-rows/2, rows/2)
	p.set_zlabel('Weight')
	p.set_zlim3d(0, 1)
	p.plot_surface(u, v, frequencyFilter)
	i = figure.add_subplot(1, 2, 2)
	i.imshow(frequencyFilter, cmap=matplotlib.cm.Greys_r)
	matplotlib.pyplot.show()
