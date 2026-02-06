import cv2
import numpy as np
import tensorflow as tf
from ImageCapture import captureImage
import time

# our training image was scaled to this size
IMG_SIZE = 128

# Prediction period interval
PRED_INTERVAL = 0.5


# Load the trained model
model = tf.keras.models.load_model("../models/hand_gesture_cnn_old.h5")


# Class name adjustment
classNames = ["Thumb", "Fist", "Palm", "Peace", "Call"]


# prediction state -> no frame by frame predictin -> stabilization
lastPredTime = 0
lastClssId = None
lastConfidence = 0.0



while True:
    img, imgWhite = captureImage()
        
    if imgWhite is not None:
        # Currently our image is 600 x 600 with each pixel ranging from 0 to 255
        # Preprocess image to match training input

        imgInput = cv2.resize(imgWhite, (IMG_SIZE, IMG_SIZE)) # resize
        imgInput = imgInput / 255.0 # scale

        # THIS LINE IS CRITICAL
        # During Training we passed the batches of images, so the model expects a batch, something like (1, 128, 128, 3)
        # the following line converts (128, 128, 3)   -->   (1, 128, 128, 3)
        imgInput = np.expand_dims(imgInput, axis=0) 



        # Delaying the prediction
        currentTime = time.time()

        if currentTime - lastPredTime >= PRED_INTERVAL:
            # Prediction
            preds = model.predict(imgInput, verbose=0)
            # this returns a NumPy array of class probabilities

            # verbose = 0  -> silent (print nothing)
            # verbose = 1  -> progress bar / logs
            # verbose = 2  -> one line per step 

            lastClssId = np.argmax(preds)
            lastConfidence = np.max(preds)
            lastPredTime = currentTime




        # Build display text
        if lastClssId is not None and lastConfidence >= 0.6:
            text = f"Gesture: {classNames[lastClssId]} ({lastConfidence:.2f})"
        else:
            text = "Gesture: Detecting..."
        
        # put the text on screen
        cv2.putText(
            img, # frame to draw on
            text, # string to display
            (20, 50), # pixel coordinates to put the text
            cv2.FONT_HERSHEY_SIMPLEX, # font 
            1, # font scale
            (0, 255, 0), # color
            2 # thickness 
        )

        cv2.imshow("Camera", img)


        # putText(
        #     image,
        #     text,
        #     position,
        #     font,
        #     size,
        #     color,
        #     thickness
        # )


        cv2.imshow("Processed Image", imgWhite)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
cv2.destroyAllWindows()







