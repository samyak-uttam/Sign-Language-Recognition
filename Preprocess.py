import cv2

def preprocess(frame):
    
    # convert to grayscale
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    ret, hand_img = cv2.threshold(thr, 150, 255, cv2.THRESH_BINARY_INV)
    
    return hand_img