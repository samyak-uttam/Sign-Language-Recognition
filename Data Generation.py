# import all the required libraries

from Preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json

# open and get the constants from the json file
f = open('Constants.json')
data = json.load(f)

labels = data['labels']

# number of classes
NUM_CLASSES = data['NUM_CLASSES']

# number of samples per class
SAMPLES_PER_CLASS = data['SAMPLES_PER_CLASS']

# size of the images to be generated
IMG_SIZE = data['IMG_SIZE']

# dictionary to hold the generated data
samples = {}

for key in range(NUM_CLASSES):
    samples[key] = {}
    samples[key]['name'] = key
    samples[key]['images'] = []

def gatherData():
    
    # open camera and take a get the video feed from camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('video')
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # count of images captured
    count = 0
    
    # if all the images of the current label are captured then it will be false
    canPressSpace = True
    
    # enter is pressed or not
    enterPressed = True
    
    while True:
        ret, frame = cam.read()

        if count == NUM_CLASSES * SAMPLES_PER_CLASS:
            print('All images captured, closing.')
            break

        if not ret:
            print('failed to grab frame')
            break 

        # get the mirror image
        frame = cv2.flip(frame, 1)
        
        image = preprocess(frame)

        cv2.imshow('hand', image)

        resized_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_arr = np.array(resized_img)
        img_arr = img_arr.reshape(IMG_SIZE, IMG_SIZE, 1)

        cv2.putText(frame, 'Press Space for capturing images of ' + labels[str(int(count / SAMPLES_PER_CLASS))], (3, 365), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        cv2.putText(frame, 'Captured ' + str(count % SAMPLES_PER_CLASS) + ' images of ' + labels[str(int(count / SAMPLES_PER_CLASS))], (3, 415), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        if not canPressSpace:
            cv2.putText(frame, 'Captured all images of ' + labels[str(int((count - 1) / SAMPLES_PER_CLASS))] + '. Press enter for next label.', (3, 465), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        cv2.imshow('video', frame)
        
        if not enterPressed and count % SAMPLES_PER_CLASS == 0:
            canPressSpace = False

        k = cv2.waitKey(1)
        if k % 256 == 13:
            canPressSpace = True
            enterPressed = True
        elif k % 256 == 32:
            # Space pressed
            if not canPressSpace:
                continue
            samples[int(count / SAMPLES_PER_CLASS)]['images'].append(img_arr)
            print('Taken ' + str(count % SAMPLES_PER_CLASS) + ' samples of: ' + labels[str(int(count / SAMPLES_PER_CLASS))])
            count = count + 1
            enterPressed = False

        elif k % 256 == 27:
            # Esc pressed
            print('Escape pressed, closing...')
            break

    cam.release()
    cv2.destroyAllWindows()
    return samples

def saveData(samples):
    # store the images in the dataset folder
    if('dataset' not in os.listdir()):
        os.mkdir('dataset')
    
    # store images of each labels in their respective folder
    for key, value in samples.items():
        class_name = labels[str(value['name'])]
        dir_name = os.path.join('dataset', class_name)
        if class_name not in os.listdir('dataset'):
            os.mkdir(dir_name)
        
        imgs = np.array(value['images'])
        
        for index in range(len(imgs)):
            img_name = str(index) + '.jpg'
            img_path = os.path.join(dir_name, img_name)
            cv2.imwrite(img_path, imgs[index])


# gather and save the data
if __name__ == '__main__':
    samples = gatherData()
    saveData(samples)