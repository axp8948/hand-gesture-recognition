import cv2
import time
import os
from dataCollection import captureImage

DIRECTORY = "../../data/test/4"
os.makedirs(DIRECTORY, exist_ok=True)



def saveData():
    counter = 0
    while True:      
        img, image = captureImage()
        if img is None:
            continue
        cv2.imshow("Image", img)

        if image is not None:
            cv2.imshow("Processed Image", image)
            
        key = cv2.waitKey(1)

        if key == ord("s") and image is not None:
            cv2.imwrite(f'{DIRECTORY}/Image_{time.time()}.jpg', image)
            print("Image Saved")
            print(counter)
            counter += 1

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    saveData()



