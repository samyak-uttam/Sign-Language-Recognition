# Sign Language Recognizer
Sign language is a visual means of communicating through hand signals, gestures, facial expressions, and body language used by/for people with hearing impairment.
<br/>
In this project, we are trying to build a sign language recognizer to help deaf people communicate through sign language.
We've built our model and implemented using OpenCV the classification of Sign Language using CNN from scratch.
We've built our dataset using OpenCV which includes images of Sign Language Digits from 0 to 9 and characters from a to z.

<img src = 'WebApp/static/images/signs.png'>

### Details of dataset
We have generated images with following specifications:
- Image size: 400 * 400
- Color space: Grayscale
- Numbers of labels(classes): 37 (0 - 9 digits, a - z characters and blank)
- Number of images per label: 150

The dataset can be found [here](https://drive.google.com/drive/folders/1fJ0dQYaLPSlmh0JPRXaq0p1npojAqKIJ?usp=sharing).

### CNN model
We have build a CNN model with 5 Convolution and MaxPooling layers followed by 2 Dense layer, which provide the output in range 0 - 36 (corresponding to our 36 labels (0 - 9, a - z and blank)).

Our model provides 100.00% accuracy on training set and 99.93% accuracy on test set.

### Prerequisites and Installing
Follow the below steps to set up the virtual environment locally
1. First of all, dowload or clone the repo to use it locally, type the following command in cmd
	```
	git clone git@github.com:samyak-uttam/Sign-Language-Recognition.git
	```
1. Pip is installed by default on many newer python builds, to check if it is installed, type
	```
	pip help
	```

2. Install virtualenv using
	```
	pip install virtualenv
	```

3. Inside the project directory run the following command
	```
	python -m venv myenv
	```

4. To activate the created environment :
	
	#### For Windows :
	```
	myenv\Scripts\activate
	```

	#### For Mac/Linux:
	```
	source myenv/bin/activate
	```

5. Now install the packages from the 'requirements.txt' file
	```
	pip install -r requirements.txt
	```

**Note** - If you are using Mac/Linux then use pip3 instead of pip and python3 instead of python.

After this, you are ready to run the code in your machine!

### Output
We have implemented the live classification of sign language through our model using Web App and OpenCV.

## Contributers
[Samyak Uttam](https://github.com/samyak-uttam) and [Ankit Kumar Mishra](https://github.com/anky008).