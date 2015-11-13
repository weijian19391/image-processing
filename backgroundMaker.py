import cv2
import cv2.cv as cv
import numpy as np

# _,img = cap.read()
avgImg = np.float32(cv2.imread("panorama_frame_ 0.jpg"))
# cv2.imwrite("hahaa.jpg", avgImg)

for fr in range(1, 7000):
    img = cv2.imread("panorama_frame_ " + str(fr) +".jpg")
    avgImg = (fr/(fr+1.0))*avgImg
    avgImg3 = (1.0/(fr+1))* (img)
    avgImg = avgImg + avgImg3
    normImg = cv2.convertScaleAbs(avgImg)
    cv2.imwrite("finalresult.png", normImg)
    print fr










