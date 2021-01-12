import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

model = keras.models.load_model('sign_lang.h5')

# open camera and take a get the video feed from camera

cam = cv2.VideoCapture(0)
cv2.namedWindow('video')

labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g',
    17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o',
    25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w',
    33: 'x', 34: 'y', 35: 'z'
}

IMG_SIZE = 200
font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
	ret, frame = cam.read()
	
	if not ret:
		print('failed to grab frame')
		break 
	
	# get the mirror image
	frame = cv2.flip(frame, 1)

	# convert to grayscale
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )

	# defining the ROI
	x1 = int(0.5 * frame.shape[1])
	y1 = 5
	x2 = frame.shape[1] - 5
	y2 = int(0.7 * frame.shape[0])
	cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

	# extracting the ROI
	roi = image[y1:y2, x1:x2]

    # blur the mask to help remove noise, then apply the
    # mask to the frame
	blur = cv2.GaussianBlur(roi, (5, 5), 5)

    # apply opening transformation to the mask
    # using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	image = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    # use thresholding technique for segmenting the hand from the frame
	thr = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	ret, image = cv2.threshold(thr, 150, 255, cv2.THRESH_BINARY_INV)

	cv2.imshow('hand', image)

	resized_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
	x_test = np.array(resized_img)
	x_test = x_test.reshape(1, IMG_SIZE, IMG_SIZE, 1)

	# get the predicted output
	y_pred = np.argmax(model.predict(x_test))

	cv2.putText(frame, 'Character: ' + labels[y_pred], (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
	cv2.imshow('video', frame)

	k = cv2.waitKey(1)
	if k % 256 == 27:
		# Esc pressed
		print('Escape pressed, closing...')
		break

cam.release()
cv2.destroyAllWindows()