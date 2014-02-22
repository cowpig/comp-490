import csv
import cPickle
import numpy as np
from random import randint
from scipy import ndimage
from scipy.misc import imresize


# literals that are specific to our dataset
BAD_IMAGES = [1747, 1908]

# maps the keypoints to their indices in the dataset
KEYPOINT_INDEX = {
	"left_eye_center"      	  : (1,0),
	"right_eye_center"     	  : (3,2),
	"left_eye_inner"       	  : (5,4),
	"left_eye_outer"       	  : (7,6),
	"right_eye_inner"      	  : (9,8),
	"right_eye_outer"      	  : (11,10),
	"left_eyebrow_inner"   	  : (13,12),
	"left_eyebrow_outer"   	  : (15,14),
	"right_eyebrow_inner"  	  : (17,16),
	"right_eyebrow_outer"  	  : (19,18),
	"nose_tip"             	  : (21,20),
	"left_mouth_corner"    	  : (23,22),
	"right_mouth_corner"   	  : (25,24),
	"center_mouth_top_lip" 	  : (27,26),
	"center_mouth_bottom_lip" : (29,28)
}


# don't want to bother catching exceptions on empty strings
def str_to_float(string):
	if string == '':
		return None
	return float(string)

# loads the training set and processes it. Outputs two 2d arrays: 
#   the raw image data, and the facial feature coordinates. the third
#   array it outputs is the list of the names of the facial features
#   (e.g. 'left_eye_outer_corner_x') I generally call the values 'labels'
#   and that refers to the number, not the string
def load_train_set(filename):
	train_set = []
	labels = []

	with open(filename, 'rb') as f:
		r = csv.reader(f)
		label_names = r.next()[:-1]

		for i, line in enumerate(r):
			try:
				if i not in BAD_IMAGES:
					labels.append([str_to_float(s) for s in line[:-1]])
					train_set.append([float(s) for s in line[-1].split(' ')])
			except:
				import pdb; pdb.set_trace() # loads up python debugger
			# if i > 50:
			#     break

	return (train_set, labels, label_names)


# takes a line containing raw image data, and reshapes it into a 96 row,
#   96-column matrix
def to_matrix(line):
	assert(len(line) == 96 * 96)
	return np.reshape(line, (96, 96))

# takes an image and displays it
def display_image(img):
	if len(img) == 96*96:
		plt.imshow(to_matrix(img))
	else:
		plt.imshow(img)
	plt.gray()
	plt.show()

# resizes an image
def resize(img, size):
	return ndimage.interpolation.zoom(img, size)

# gets the subimage from top-right to bottom-left, INCLUSIVE
def get_subimage(img, top_left, bot_right):
	if len(img) == 96*96:
		img = to_matrix(img)
	top, left = top_left
	bot, right = bot_right
	return img[top:bot+1, left:right+1]

# simple euclidean distance calculation between two points
def euclidean_distance(a, b):
	if type(a) == tuple or type(a) == list:
		a = np.array(a)
	if type(b) == tuple or type(b) == list:
		b = np.array(b)
	return np.linalg.norm(a - b)

# finds the distance between two points in the dataset, useful for normalization
def label_distance(label, indices_a, indices_b):
	point_a = [label[indices_a[0]], label[indices_a[1]]]
	if point_a[0] == '' or point_a[1] == '' or point_a[0] == None or point_a[1] == None:
		return None
	point_b = [label[indices_b[0]], label[indices_b[1]]]
	if point_b[0] == '' or point_b[1] == '' or point_b[0] == None or point_b[1] == None:
		return None

	try:
		point_a = np.array([float(x) for x in point_a])
		point_b = np.array([float(x) for x in point_b])
	except:
		import pdb;pdb.set_trace()

	return euclidean_distance(point_a, point_b)