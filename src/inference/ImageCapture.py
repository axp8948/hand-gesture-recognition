import cv2
from cvzone.HandTrackingModule import HandDetector # For the hand detection
import numpy as np
import math


cap = cv2.VideoCapture(1) # capture object #0 id number for the webcam
detector = HandDetector(maxHands=1) # For detecting hand


# Set offset for how much padding to put
offset = 40

imgSize = 600


def captureImage():
    # read the image form cap object
    success, img = cap.read()

    if not success:
        return None, None

    # detect the hand from image object
    hands, img = detector.findHands(img)

    # CROPPING THE IMAGE TO EXTRACT THE HAND

    imgWhite = None

    if hands:
        hand = hands[0] # since we only have one hand

        x, y, w, h = hand['bbox']  # the values of bounding box from the dictionary


        # TO MATCH THE DIMENSION OF ALL THE GESTURES
        # Create a base image and fill up the empty spaces
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # 300 x 300 image with color channels # unit8 values will from 0 to 255, multiply by 255 to make the pixels white

        # imgCrop = img[y: y + h, x: x + h] # cropping the image from the matrix of main image => this will crop hand too closely

        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset] # cropping the image from the matrix of main image

        # PLACE THE IMAGE CROP MATRIX INSIDE THE IMAGE WHITE MATRIX
        
        # extract the dimension fo cropped image
        imgCropShape = imgCrop.shape



        # TO MINIMIZE THE WHITE SPACES IN THE PICTURE
        aspectRatio = h / w # if greater than 1, height is greater than the width

        # adjustment for the height
        if aspectRatio > 1:
            k = imgSize / h # constant of stretch
            wCal = math.ceil(k*w)  # stretch the width by proportional scale

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            # Center the image
            wGap = math.ceil((imgSize - wCal) / 2)


            imgWhite[:, wGap:wCal + wGap] = imgResize
        
        # adjustment for the width
        else:
            k = imgSize / w # constant of stretch
            hCal = math.ceil(k*h)  # stretch the width by proportional scale

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # Center the image
            hGap = math.ceil((imgSize - hCal) / 2)


            imgWhite[hGap:hCal + hGap, : ] = imgResize

            


        # # Displaying the cropped image
        # cv2.imshow("Cropped Image", imgCrop)
        # cv2.imshow("White Image", imgWhite)


    return img, imgWhite






