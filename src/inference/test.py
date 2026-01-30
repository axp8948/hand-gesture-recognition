import cv2
import numpy as np
import tensorflow as tf
from ImageCapture import captureImage

# our training image was scaled to this size
IMG_SIZE = 128


# Load the trained model
model = tf.keras.models.load_model("../models/hand_gesture_cnn.h5")


# Class name adjustment
classNames = ["Thumb", "Fist", "Palm"]

while True:
    img, imgWhite = captureImage()

    if img is not None:
        cv2.imshow("Camera", img)

    if imgWhite is not None:
        # Currently our image is 600 x 600 with each pixel ranging from 0 to 255
        # Preprocess image to match training input

        imgInput = cv2.resize(imgWhite, (IMG_SIZE, IMG_SIZE)) # resize
        imgInput = imgInput / 255.0 # scale

        # THIS LINE IS CRITICAL
        # During Training we passed the batches of images, so the model expects a batch, something like (1, 128, 128, 3)
        # the following line converts (128, 128, 3)   -->   (1, 128, 128, 3)
        imgInput = np.expand_dims(imgInput, axis=0) 


        # Prediction
        preds = model.predict(imgInput, verbose=0)
        # this returns a NumPy array of class probabilities

        # verbose = 0  -> silent (print nothing)
        # verbose = 1  -> progress bar / logs
        # verbose = 2  -> one line per step 

        classId = np.argmax(preds)
        confidence = np.max(preds)

        # Display the prediction
        text = f"Gesture: {classNames[classId]} ({confidence:.2f})"
        
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







