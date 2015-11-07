import cv2
import numpy as np

winname = "GRS"
#bgs_mog = cv2.BackgroundSubtractorMOG(1000, 6, 0.9, 0)
fgbg = cv2.BackgroundSubtractorMOG()
capture = cv2.VideoCapture('football_right.mp4')

j = 0

if __name__ == "__main__":
    while capture.isOpened():
        _,frame = capture.read()
        numRow, numCol, numRGB = frame.shape

        ##### TO INTENSIFY THE COLOURS OF THE PLAYERS
        image_blue = frame[:,:,0].flatten()
        image_green = frame[:,:,1].flatten()
        image_red = frame[:,:,2].flatten()
        cmax = np.maximum(np.maximum(image_blue,image_green), image_red)

        masked_unmax_image_red = np.ma.masked_where(cmax!=image_red, image_red) #red pixels w max values
        masked_lowerintensity_image_red = np.ma.masked_where(masked_unmax_image_red<= 80, masked_unmax_image_red)
        max_intensity_image_red = masked_lowerintensity_image_red/masked_lowerintensity_image_red * 255
        max_intensity_image_red.mask = np.ma.nomask

        max_intensity_image_red_nomask = np.ma.filled(max_intensity_image_red,0)
        #print type(max_intensity_image_red_nomask)

        final_red = max_intensity_image_red_nomask.reshape(numRow,numCol)
        final_blue = image_blue.reshape(numRow,numCol)
        final_green = image_green.reshape(numRow,numCol)

        final_intensified_image = np.dstack((final_blue,final_green,final_red))

        #print type(final_intensified_image)

        fgmask = fgbg.apply(final_intensified_image, learningRate = 1.0/10 )
        #mask_rbg = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        #draw = frame & mask_rbg
        contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for i in xrange(0,len(contours)):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            if cv2.contourArea(cnt) > 300:
                cv2.drawContours(fgmask,[box],0,(255,255,255),2)
                #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

        cv2.imwrite("intensified frame "+ str(j)+ ".png",fgmask)

        j +=1

        if j == 20:
            exit()


    #cv2.destroyAllWindows()


















