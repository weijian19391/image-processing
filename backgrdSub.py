import cv2
import numpy as np

winname = "GRS"
#bgs_mog = cv2.BackgroundSubtractorMOG(1000, 6, 0.9, 0)
bgs_mog =  cv2.BackgroundSubtractorMOG()
capture = cv2.VideoCapture('football_right.mp4')
j = 0
if __name__ == "__main__":
    while capture.isOpened():
        _,frame = capture.read()
        fgmask = bgs_mog.apply(frame,learningRate=1.0/10)
        mask_rbg = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        draw = frame & mask_rbg
        cv2.imwrite("lala_contours frame " + str(j) + ".png", draw)
        j += 1

        #cv2.imshow(winname, draw)
        #c = cv2.waitKey(1)
        #if c == 27:
        #break

        if j == 20:
            exit()


    #cv2.destroyAllWindows()



