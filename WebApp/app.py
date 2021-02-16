import numpy as np
from cv2 import cv2
import sys
from time import sleep
import json
from tensorflow import keras
from string import ascii_lowercase
from flask import Flask, render_template, Response
sys.path.append("..")
from Preprocess import preprocess

# loading the weights for the model
model = keras.models.load_model('../sign_lang.h5')

# open and get the constants from the json file
f = open('../Constants.json')
data = json.load(f)

labels = data['labels']

# size of the images to be generated
IMG_SIZE = data['TRAIN_IMG_SIZE']

classes = ascii_lowercase + '0123456789'

# open camera and take a get the video feed from camera
cam = cv2.VideoCapture(0)

# dictionary to store the results
results = {'character': '', 'word': '', 'sentence': ''}

def generateFrames():

	# define the requrired variables
	character = ''
	current_word = ''
	sentence = ''

	count = {}
	blank_count = 0

	for i in classes:
		count[i] = 0

	while True:
		ret, frame = cam.read()

		if not ret:
			print('failed to grab frame')
			break 

		# get the mirror image
		frame = cv2.flip(frame, 1)

		image = preprocess(frame)

		img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
		x_test = np.array(img)
		x_test = x_test.reshape(1, IMG_SIZE, IMG_SIZE, 1)

		# get the predicted output
		y_pred = np.argmax(model.predict(x_test))

		character = labels[str(y_pred)]

		if len(current_word) != 0 and character == 'blank':
			blank_count += 1
			if blank_count > 80:
				sentence += ' '  + current_word
				current_word = ''
				for i in classes:
					count[i] = 0
				blank_count = 0

		elif character != 'blank':
			count[character] += 1
			if count[character] > 50:
				current_word += character
				for i in classes:
					count[i] = 0
				blank_count = 0

		results['character'] = character
		results['word'] = current_word
		results['sentence'] = sentence

		ret, buffer = cv2.imencode('.jpg', frame)
		frame_bytes = buffer.tobytes()
		# concat frame one by one and show result
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

	cam.release()

app = Flask(__name__)

# so that css dosen't get cached just refresh the page to see css changes
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/results_feed")
def results_feed():
	def generate():
		while True:
			cur = ''
			for key,value in results.items():
				cur += value + '#'
			yield "{}\n".format(cur)
			sleep(1)

	return app.response_class(generate(), mimetype="text/plain")

if __name__ == '__main__':
	app.run(debug=False)