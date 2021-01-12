# Sign Language Recognizer
Sign language is a visual means of communicating through hand signals, gestures, facial expressions, and body language used by/for people with hearing impairment.
<br/>
In this project, we are trying to build a sign language recognizer to help deaf people communicate through sign language.
We've built our model and implemented using OpenCV the classification of Sign Language using CNN from scratch.
We've built our dataset using OpenCV which includes images of Sign Language Digits from 0 to 9 and characters from a to z.

<img src = 'images/signs.png'>

### Details of dataset
We have generated images with following specifications:
- Image size: 320 * 320
- Color space: Grayscale
- Numbers of labels(classes): 36 (0 - 9 digits and a - z characters)

The dataset and the model file can be found [here](https://drive.google.com/drive/folders/1fJ0dQYaLPSlmh0JPRXaq0p1npojAqKIJ?usp=sharing).

### CNN model
We have build a CNN model with 5 Convolution and MaxPooling layers followed by 2 Dense layer, which provide the output in range 0 - 35 (corresponding to our 36 labels (0 - 9 and a - z)).

Our model provides 99.92% accuracy on training set and 99.67% accuracy on test set.

### Output
Following is the live classification output of our model seen using openCV.

## Contributers
[Samyak Uttam](https://github.com/samyak-uttam) and [Ankit Kumar Mishra](https://github.com/anky008).