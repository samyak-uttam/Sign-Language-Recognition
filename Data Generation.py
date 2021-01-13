# import all the required libraries
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
        img_arr = np.array(resized_img)
        img_arr = img_arr.reshape(IMG_SIZE, IMG_SIZE, 1)

        cv2.putText(frame, 'Press Space for capturing images of ' + labels[str(int(count / 100))], (3, 365), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        cv2.putText(frame, 'Captured ' + str(count % 100) + ' images of ' + labels[str(int(count / 100))], (3, 415), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        if not canPressSpace:
            cv2.putText(frame, 'Captured all images of ' + labels[str(int((count - 1) / 100))] + '. Press enter for next label.', (3, 465), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
        cv2.imshow('video', frame)
        
        if not enterPressed and count % 100 == 0:
            canPressSpace = False

        k = cv2.waitKey(1)
        if k % 256 == 13:
            canPressSpace = True
            enterPressed = True
        elif k % 256 == 32:
            # Space pressed
            if not canPressSpace:
                continue
            samples[int(count / 100)]['images'].append(img_arr)
            print('Taken ' + str(count % 100) + ' samples of: ' + labels[str(int(count / 100))])
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