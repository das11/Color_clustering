import cv2
import numpy as np 

maxc = list()

def centroid_histogram(clt) :

	clusters = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = clusters)

	hist = hist.astype("float")
	hist /= hist.sum()

	return hist

def plot_colors(hist, centroids) : 

	bar = np.zeros((50,300, 3), dtype = "uint8")
	startx = 0
	
	for (percent, color) in zip(hist, centroids) :
		print(percent)
		maxc.append(percent)
		endx = startx + (percent * 300)
		cv2.rectangle(bar, (int(startx), 0), (int(endx), 50), color.astype("uint8").tolist(), -1)
		startx = endx

	plot_max()

	return bar

def plot_max() : 

	maxbar = np.zeros((50,50,3), dtype = "uint8")

	x = np.amax(maxc)
	cv2.rectangle(maxbar, (0,0), (50,50), x, -1)

	return maxbar
