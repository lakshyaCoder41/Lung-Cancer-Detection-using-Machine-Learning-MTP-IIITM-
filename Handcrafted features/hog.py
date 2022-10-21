import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def image_array(image_path,with_plot=False):
    img = cv.imread(image_path)
    print('Image Shape: ',img.shape)
    img = img.astype(np.uint8)
    img = img / 255
    plt.imshow(img,cmap='gray')

    # Show the new shape of the image
    image_sum = img.sum(axis=2)
    print(image_sum.shape)

    # Show the max value at any point.  1.0 = Black, 0.0 = White
    image_bw = image_sum/image_sum.max()
    print(image_bw.max())
    #print(image_bw)

    img = resize(color.rgb2gray(io.imread(image_path)), (128, 64))

    #visualization
    if with_plot:
        plt.figure(figsize=(15, 8))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()

    #image array
    img = np.array(img)
    return img

def mag_theta_calculation(img):
	mag = []
	theta = []
	for i in range(128):
		magnitudeArray = []
		angleArray = []
		for j in range(64):
			# Condition for axis 0
			if j-1 <= 0 or j+1 >= 64:
				if j-1 <= 0:
					# Condition if first element
					Gx = img[i][j+1] - 0
				elif j + 1 >= len(img[0]):
					Gx = 0 - img[i][j-1]
			# Condition for first element
			else:
				Gx = img[i][j+1] - img[i][j-1]
			
			# Condition for axis 1
			if i-1 <= 0 or i+1 >= 128:
				if i-1 <= 0:
					Gy = 0 - img[i+1][j]
				elif i +1 >= 128:
					Gy = img[i-1][j] - 0
			else:
				Gy = img[i-1][j] - img[i+1][j]

			# Calculating magnitude
			magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
			magnitudeArray.append(round(magnitude, 9))

			# Calculating angle
			if Gx == 0:
				angle = math.degrees(0.0)
			else:
				angle = math.degrees(abs(math.atan(Gy / Gx)))
			angleArray.append(round(angle, 9))
		mag.append(magnitudeArray)
		theta.append(angleArray)   
	return mag,theta

def visualize_magnitude(mag):
# visualization of magnitudeArray
    plt.figure(figsize=(15, 8))
    plt.imshow(mag, cmap="gray")
    plt.axis("off")
    plt.show()

def visualize_theta(theta):
# visualization of angleArray
    plt.figure(figsize=(15, 8))
    plt.imshow(theta, cmap="gray")
    plt.axis("off")
    plt.show()

def hog_vector(magnitute,angle):
    number_of_bins = 9
    step_size = 180 / number_of_bins

    def calculate_j(angle):
        temp = (angle / step_size) - 0.5
        j = math.floor(temp)
        return j


    def calculate_Cj(j):
        Cj = step_size * (j + 0.5)
        return round(Cj, 9)


    def calculate_value_j(magnitude, angle, j):
        Cj = calculate_Cj(j+1)
        Vj = magnitude * ((Cj - angle) / step_size)
        return round(Vj, 9)


    histogram_points_nine = []
    for i in range(0, 128, 8):
        temp = []
        for j in range(0, 64, 8):
            magnitude_values = [[mag[i][x] for x in range(j, j+8)] for i in range(i,i+8)]
            angle_values = [[theta[i][x] for x in range(j, j+8)] for i in range(i, i+8)]
            for k in range(len(magnitude_values)):
                for l in range(len(magnitude_values[0])):
                    bins = [0.0 for _ in range(number_of_bins)]
                    value_j = calculate_j(angle_values[k][l])
                    Vj = calculate_value_j(magnitude_values[k][l], angle_values[k][l], value_j)
                    Vj_1 = magnitude_values[k][l] - Vj
                    bins[value_j]+=Vj
                    bins[value_j+1]+=Vj_1
                    bins = [round(x, 9) for x in bins]
            temp.append(bins)
        histogram_points_nine.append(temp)


    # print(len(histogram_points_nine))
    # print(len(histogram_points_nine[0]))
    # print(len(histogram_points_nine[0][0]))

    epsilon = 1e-05

    feature_vectors = []
    for i in range(0, len(histogram_points_nine) - 1, 1):
        temp = []
        for j in range(0, len(histogram_points_nine[0]) - 1, 1):
            values = [[histogram_points_nine[i][x] for x in range(j, j+2)] for i in range(i, i+2)]
            final_vector = []
            for k in values:
                for l in k:
                    for m in l:
                        final_vector.append(m)
            k = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
            final_vector = [round(x/(k + epsilon), 9) for x in final_vector]
            temp.append(final_vector)
        feature_vectors.append(temp)


    # print(len(feature_vectors))
    # print(len(feature_vectors[0]))
    # print(len(feature_vectors[0][0]))

    print("\n HOG Feature Vector: ", feature_vectors)