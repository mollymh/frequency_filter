import cv2
import numpy as np

def frequency_filter(im, frequencyFilter, delta=0):
	"""
	Title:
		frequency_filter
	Description:
		Applies given frequencyFilter to given image
	Attributes:
		im - 2D ndarray signal to be filtered
		frequencyFilter - filter to be applied
		delta - offset to be added to final image
	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(im) != np.ndarray:
		raise TypeError("Source image must be ndarray.")
	if type(frequencyFilter) != np.ndarray:
		raise TypeError("Source image must be ndarray.")
	if im[:,:,0].size != frequencyFilter.size:
		raise ValueError("frequencyFilter and im must be same size")
	if type(delta) != int or delta < 0:
		raise ValueError("Given delta must be positive integer.")

	#deal with color
	if len(im.shape) > 2:
		channels = im.shape[2]
	else:
		channels = 1
	
	#create checkerboard
	check = np.empty(frequencyFilter.shape)
	for u in range(check.shape[0]): #rows
		for v in range(check.shape[1]): #columns
			check[u,v] = (-1)**(u+v)
			
	dst = np.empty(im.shape)

	#go through each channel
	for c in range(channels):
		src = im[:,:,c]*check #center image
		Fc = np.fft.fft2(src) #to freq space
		Fc = np.multiply(Fc,frequencyFilter) #apply filter
		dst[:,:,c] = np.fft.ifft2(Fc)*check #back to spatial and un-center

	dst = np.absolute(dst) #obtain magnitude
	
	dst = (dst+delta).astype(im.dtype) #apply offset and fix dtype
	
	return dst
	

if __name__ == '__main__':

	import cv2
	import ipcv
	import numpy
	import os.path
	import time

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
	#filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'

	im = cv2.imread(filename)

	frequencyFilter = ipcv.filter_lowpass(im, 
						16, 
						filterShape=ipcv.IPCV_GAUSSIAN)

	startTime = time.clock()
	offset = 0
	filteredImage = ipcv.frequency_filter(im, frequencyFilter, delta=offset)
	filteredImage = numpy.abs(filteredImage)
	filteredImage = filteredImage.astype(dtype=numpy.uint8)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (frequency_filter)= {0} [s]'.format(elapsedTime))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, im)
	cv2.imshow(filename, ipcv.histogram_enhancement(im))

	filterName = 'Filtered (' + filename + ')'
	cv2.namedWindow(filterName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filterName, filteredImage)
	cv2.imshow(filterName, ipcv.histogram_enhancement(filteredImage))

	ipcv.flush()
