#Name: Jeff Stokes
#Date: 10/01/2020
#Midterm - Computer Vision
#Instructor: Gil Gallegos, gil.gallegos@gmail.com
#TA: Rahim Ullah, rullah@live.nmhu.edu

#Import necessary libaries
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Initialize variables and array for section D
N = 100
edge_count = 0
pixel_count = 0
density_array = np.zeros([N,1])

for i in range(N):
	
	#Section B

	#read in images and convert to grayscale using openCV
	img = cv2.imread("output/output0_"+str(i)+".jpg")
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#find edges of images with laplacian method
	lap = cv2.Laplacian(imgGray, cv2.CV_64F, ksize=3)
	lap = np.uint8(np.absolute(lap))
	
	#find threshold of images
	retval, thresh = cv2.threshold(lap, 25, 255, cv2.THRESH_BINARY)

	#invert with bitwise_not
	invert = cv2.bitwise_not(thresh)
	
	#write out one of every 20 frames to file (5 total per segment)
	if i % 20 == 0:
		cv2.imwrite("results/grayscale/grayimg_"+str(i)+".jpg", imgGray)
		cv2.imwrite("results/laplacian/lap_"+str(i)+".jpg", lap)
		cv2.imwrite("results/threshold/thresh_"+str(i)+".jpg", thresh)
		cv2.imwrite("results/inverted/inverted_"+str(i)+".jpg", invert)

	#Section C
	#analyze all inverted frames to count black and white pixels as histograms, saving every 20th to file
	hist = cv2.calcHist([invert],[0],None,[256],[0,256])
	if i % 5 == 0:
		plt.plot(hist)
		plt.title("Histogram of inverted frame " + str(i))
		plt.ylabel("Number of pixels")
		plt.xlabel("Pixel intensity")
		plt.savefig("results/histograms/histogram_frame_"+str(i)+".png")
		plt.close()

	#Section D
	#calculate density via dividing black pixels (edges) from histogram by all in histogram
	edge_count += hist[0]
	pixel_count += (edge_count + hist[-1]) 
	density = edge_count/pixel_count
	density_array[i] = density

plt.title("Pixel Density of images")
plt.ylabel("Density")
plt.xlabel("Frame Number")
plt.plot(density_array)
plt.savefig("results/Density_array.png")
plt.close()