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

# flips an image horizontally
def flip_horizontal(matrix):
	if type(matrix) == list:
		return [row[::-1] for row in matrix]

	return matrix[...,::-1]


# builds a training set for just eyes
def build_eye_trainset(train_set, labels):
	# to_shuffle = zip(train_set, labels)
	# np.random.shuffle(to_shuffle)
	# train_set, labels = zip(*to_shuffle)

	eyes = []

	for i, label in enumerate(labels):
		dist_h_left_eye = label_distance(label, left_eye_inner, left_eye_outer)
		dist_h_right_eye = label_distance(label, right_eye_inner, right_eye_outer)

		# add each eye image with a positive label
		if dist_h_left_eye != 0 and dist_h_left_eye != None:
			left = label[4]
			right = label[6]
			middle = np.average([label[5], label[7]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			if (top - bot) < 24:
				bot += 1

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_l = (top, left)
			br_l = (bot, right)
			eyes.append((subimg, 1, i))

		if dist_h_right_eye != 0 and dist_h_right_eye != None:
			left = label[10]
			right = label[8]
			middle = np.average([label[9], label[11]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_r = (top, left)
			br_r = (bot, right)
			eyes.append((flip_horizontal(subimg), 1, i))

		def random(x):
			return int(np.random.random() * x)

		def too_close(new, *others):
			for other in others:
				if euclidean_distance(new, other) < TOO_CLOSE_VALUE:
					return True
			return False

		for _ in xrange(2):
			tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
			br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

			while too_close(tl, tl_l, tl_r) or too_close(br, br_l, br_r):
				tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
				br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

			eyes.append((get_subimage(train_set[i], tl, br), 0))

	return eyes