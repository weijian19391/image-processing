import cv2
import numpy as np
# np.set_printoptions(threshold='nan')
# MIN_AREA_OF_CONTOUR_BOX = 200
CONTOUR_COLOUR = (255,255,255)
CONTOUR_BTN_COLOUR = (0,0,255)
THRESHOLD = {"Red":80, "Blue":40}
MIN_AREA_OF_CONTOUR_BOX = {"Red":200, "Blue":125}
#this function takes in a color image frame, detects the colour to be detected, and return a black and white image whereby white parts are where the color is found
#color detection is based on the threshold, and whether the color channel is the max among the rest of the channel so as to detect more color gradient
#@para: frame: image array
#				colour: "Blue/Red/Green"
#				threshold: the minimum colour value of a pixel to be set to max
#@returns: an image with color intensified
def detectColour(frame, colour, threshold):
	numRow, numCol, numRGB = frame.shape
	colour_To_Be_Search = np.ndarray((numRow,numCol))
	image_blue = frame[:,:,0].flatten()
	image_green = frame[:,:,1].flatten()
	image_red = frame[:,:,2].flatten()
	cmax = np.maximum(np.maximum(image_blue,image_green), image_red)

	if colour == "Red" :
		colour_To_Be_Search = image_red
	elif colour == "Blue":
		colour_To_Be_Search = image_blue
	elif colour == "Green":
		colour_To_Be_Search = image_green

	masked_unmax_pixels = np.ma.masked_where(cmax!=colour_To_Be_Search, colour_To_Be_Search) #red pixels w max values
	masked_lowerintensity_pixels = np.ma.masked_where(masked_unmax_pixels<= THRESHOLD[colour], masked_unmax_pixels)
	masked_max_intensity_pixels= masked_lowerintensity_pixels/masked_lowerintensity_pixels * 255

	max_intensity_pixels = np.ma.filled(masked_max_intensity_pixels,0)

	return max_intensity_pixels.reshape(numRow,numCol)

#This function takes in a list of contours and draw the minAreaBox of them onto the frame
def drawPlayerOutline(frame,contours,colour):
	filled_contours=np.ndarray((frame.shape[0], frame.shape[1]))
	for i in xrange(0,len(contours)):
		cnt = contours[i]
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		if cv2.contourArea(cnt) > MIN_AREA_OF_CONTOUR_BOX[colour]:
			cv2.drawContours(frame,[box],0,CONTOUR_COLOUR,2)
			bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
			cv2.circle(frame, bottommost, 3,CONTOUR_BTN_COLOUR ,-1)
		else:
			cv2.drawContours(filled_contours,[box],0,CONTOUR_COLOUR,-1)
	return frame, filled_contours

if __name__ == "__main__":
	j = 0
	# fgbg = cv2.BackgroundSubtractorMOG()
	capture = cv2.VideoCapture('football_right.mp4')
	# capture = cv2.VideoCapture('input.avi')
	while capture.isOpened():
		_,frame = capture.read()
		contoured_frame = np.copy(frame)
		# covered_red_players = np.ndarray((frame.shape[0], frame.shape[1]))

		#------------------------------------------------detect Red Players----------------------------------------------------------
		player_detected = detectColour(frame, "Red", THRESHOLD["Red"])
		# cv2.imwrite("Images\detect " + colour + "frame "+ str(j)+ ".png",player_detected)
		contours, hierarchy = cv2.findContours(player_detected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_red, blue_players_limbs = drawPlayerOutline(contoured_frame, contours,"Red")
		# cv2.imwrite("Images\contoured Red Players frame "+ str(j)+ ".png",contoured_frame_red)

		#------------------------------------------------detect Blue Players----------------------------------------------------------
		player_detected = detectColour(frame, "Blue", THRESHOLD["Blue"])

		full_blue_player = player_detected.astype(int) + blue_players_limbs.astype(int)
		masked_wanted_points = np.ma.masked_where(full_blue_player!=0, full_blue_player)
		final_full_blue_player = np.ma.filled(masked_wanted_points,255)
		cv2.imwrite("Images\hahahae "+ str(j)+ ".png",final_full_blue_player)
		contours, hierarchy = cv2.findContours(final_full_blue_player.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_players,_ = drawPlayerOutline(contoured_frame_red, contours, "Blue")
		cv2.imwrite("Images\detect " + "both teams" + "frame "+ str(j)+ ".png",contoured_frame_players)

		j +=1

		if j == 100:
			exit()


	#cv2.destroyAllWindows()


















