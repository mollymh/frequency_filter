import ipcv
import numpy as np

def filter_lowpass(im, cutoffFrequency, order=1, filterShape=ipcv.IPCV_IDEAL):

	"""
	Title:
		filter_lowpass
	Description:
		Creates a lowpass filter for given image and parameters
	Attributes:
		im - 2D ndarray signal to provide shape of filter
		cutoffFrequency - allow frequencies below this threshold
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
	if type(cutoffFrequency) != int or cutoffFrequency < 0:
		raise ValueError("Given cutoff frequency must be positive integer.")
	if type(order) != int or order < 0:
		raise ValueError("Given order must be positive integer.")
	if type(filterShape) != int or filterShape < 0 or filterShape >2:
		raise ValueError("Filter shape option limited to 0, 1, or 2.")	
	
	#initialize variables
	src = im[:,:,0]
	M = src.shape[0]
	N = src.shape[1]
	D = np.empty(src.shape)
	H = np.zeros(src.shape)

	for u in range(M): #rows
		for v in range(N): #columns
			D[u,v] = (u-M/2)**2 + (v-N/2)**2 #distance filter
	D = np.sqrt(D)

	if filterShape == 0: #ideal
		H[np.where(D<cutoffFrequency)] = 1
		
	elif filterShape == 1: #butterworth
		H = 1/(1+(D/cutoffFrequency)**(2*order))
	
	elif filterShape == 2: #gaussian
		H = np.exp(-((D**2)/(2*cutoffFrequency))**2)
	
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

	#frequencyFilter = ipcv.filter_lowpass(im,
						#16,
						#filterShape=ipcv.IPCV_IDEAL)
	#frequencyFilter = ipcv.filter_lowpass(im,
						#16,
						#order=1,
						#filterShape=ipcv.IPCV_BUTTERWORTH)
	frequencyFilter = ipcv.filter_lowpass(im,
						16,
						filterShape=ipcv.IPCV_GAUSSIAN)

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
