import cv2
import numpy as np
np.set_printoptions(threshold='nan')
# MIN_AREA_OF_CONTOUR_BOX = 200
CONTOUR_COLOUR = (255,255,255)
CONTOUR_BTN_COLOUR = (0,0,255)
#THRESHOLD = {"Red":80, "Blue":40, "Green":150, "Black":0} #for the case of referee
THRESHOLD = {"Red":80, "Blue":40, "Green":120, "Black":0} #combined use for referee and goalkeeper

MIN_AREA_OF_CONTOUR_BOX = {"Red":200, "Blue":125, "Green":50}
MAX_AREA_OF_CONTOUR_BOX = {"Red":2000, "Blue":1000, "Green":300}
#BOUNDARIES_FOR_BACKGROUND = [([4,64,48],[69,121,95])] #BGR
#BOUNDARIES_FOR_BACKGROUND = [([4,60,48],[225,225,225])] #BGR
BOUNDARIES_FOR_BACKGROUND = [([4,50,1],[180,180,180])] #BGR


# BOUNDARIES_FOR_GOALKEEPER = [([25,86,67],[165,213,194])] #BGR

BOUNDARIES_FOR_BLACKSHORTS = [([16,19,17],[38,41,41])]

#BOUNDARIES_FOR_BLACKSHORTS = [([0,20,17],[38,65,60])] #less noisy one but there is one row of pixels not connected for referee
#BOUNDARIES_FOR_BLACKSHORTS = [([0,20,17],[38,91,72])] #can detect full body of referee sometimes
#BOUNDARIES_FOR_BLACKSHORTS = [([0,20,17],[40,90,75])] #can detect full body of referee sometimes (new)

#BOUNDARIES_FOR_BLACKSHORTS = [([0,20,17],[38,90,70])]

# BOUNDARIES_FOR_BLACKSHORTS = [([4,32,4],[38,41,41])]
# BOUNDARIES_FOR_BLACKSHORTS = [([16,19,17],[27,83,27])]


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

	if colour != "Black":
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

	else:
		for(lower, upper) in BOUNDARIES_FOR_BLACKSHORTS:
			lower = np.array(lower, dtype = "uint8")
			upper = np.array(upper, dtype="uint8")

			mask = cv2.inRange(frame, lower, upper)
			output = cv2.bitwise_and(frame,frame, mask=mask)
			masked_wanted_points = np.ma.masked_where(output!=0, output)
			max_intensity_pixels = np.ma.filled(masked_wanted_points,255)

		return max_intensity_pixels

#This function takes in a list of contours and draw the minAreaBox of them onto the frame
def drawPlayerOutline(frame,contours,colour):
	filled_contours=np.ndarray((frame.shape[0], frame.shape[1]))
	for i in xrange(0,len(contours)):
		cnt = contours[i]
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		if cv2.contourArea(cnt) > MIN_AREA_OF_CONTOUR_BOX[colour] and cv2.contourArea(cnt) < MAX_AREA_OF_CONTOUR_BOX[colour] :
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


		##-------------------------------trying to all the players on the field ---------------------------------
		for(lower, upper) in BOUNDARIES_FOR_BACKGROUND:
			lower = np.array(lower, dtype = "uint8")
			upper = np.array(upper, dtype="uint8")	
			smooth_frame = cv2.GaussianBlur(frame, (5,5), 0)
			mask = cv2.inRange(smooth_frame, lower, upper)
			output = cv2.bitwise_and(smooth_frame,smooth_frame, mask=mask)

			only_players_frame = smooth_frame - output
			# mask = cv2.inRange(frame, lower, upper)
			# output = cv2.bitwise_and(frame,frame, mask=mask)

			# only_players_frame = frame - output
			#cv2.imwrite("Images\detect"+str(j)+ ".png",only_players_frame)

			masked_wanted_points = np.ma.masked_where(output==0, only_players_frame)
			final_full_all_players_white = np.ma.filled(masked_wanted_points,255)
			#final_full_all_players = abs(255 -final_full_all_players)
		cv2.imwrite("Images\Players in white frame smooth"+str(j)+ ".png",final_full_all_players_white)

		# # #------------------------------------------------detect Red Players----------------------------------------------------------
		player_detected = detectColour(frame, "Red", THRESHOLD["Red"])
		# cv2.imwrite("Images\detect " + "Red" + "frame "+ str(j)+ ".png",player_detected)
		contours, hierarchy = cv2.findContours(player_detected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_red, blue_players_limbs = drawPlayerOutline(contoured_frame, contours,"Red")
		# cv2.imwrite("Images\contoured Red Players frame "+ str(j)+ ".png",contoured_frame_red)

		# # #------------------------------------------------detect Blue Players----------------------------------------------------------
		player_detected = detectColour(frame, "Blue", THRESHOLD["Blue"])

		full_blue_player = player_detected.astype(int) + blue_players_limbs.astype(int)
		masked_wanted_points = np.ma.masked_where(full_blue_player!=0, full_blue_player)
		final_full_blue_player = np.ma.filled(masked_wanted_points,255)
		#cv2.imwrite("Images\hahahae "+ str(j)+ ".png",final_full_blue_player)
		contours, hierarchy = cv2.findContours(final_full_blue_player.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_players,_ = drawPlayerOutline(contoured_frame_red, contours, "Blue")
		#cv2.imwrite("Images\detect " + "both teams" + "frame "+ str(j)+ ".png",contoured_frame_players)

		#------------------------------------------------detect Referee----------------------------------------------------------

		player_detected = detectColour(frame, "Green", THRESHOLD["Green"])
		# cv2.imwrite("Images\detectworiginal " + "Referee" + "frame "+ str(j)+ ".png",player_detected)

		referee_shorts = detectColour(frame, "Black", THRESHOLD["Black"])
		# cv2.imwrite("Images\detect " + "Referee_shorts" + "frame "+ str(j)+ ".png",referee_shorts)

		referee_shorts = referee_shorts[:,:,0]
		# cv2.imwrite("Images\Added_referee " + "frame "+ str(j)+ ".png",referee_shorts)

		full_referee = player_detected.astype(int) + referee_shorts.astype(int) #full body
		# cv2.imwrite("Images\detect referee " + "frame "+ str(j)+ ".png",full_referee) #gets full image of referee but too noisy

		# green_frame_only_players = only_players_frame[:,:,1] #green
		# cv2.imwrite("Images\onlyplayers_green " + "frame "+ str(j)+ ".png",green_frame_only_players) #gets full image of referee but too noisy
		#
		# output = cv2.bitwise_and(green_frame_only_players.astype(np.uint8),full_referee.astype(np.uint8))
		# cv2.imwrite("Images\Added_referee_afterminusnoise " + "frame "+ str(j)+ ".png",output)
		# masked_wanted_points = np.ma.masked_where(full_referee!=0, full_referee)
		# final_referee_player = np.ma.filled(masked_wanted_points,255)
		# cv2.imwrite("Images\Referee "+ str(j)+ ".png",final_referee_player)

		# green_frame_only_players = final_full_all_players_white[:,:,1] #green
		# full_referee_remove_backgroundnoise = cv2.bitwise_and(green_frame_only_players.astype(np.uint8),full_referee.astype(np.uint8))
		# cv2.imwrite("Images\Full_referee_remove_backgroundnoise " + "frame "+ str(j)+ ".png",full_referee_remove_backgroundnoise)
		#
		# contours, hierarchy = cv2.findContours(full_referee_remove_backgroundnoise.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		#contours, hierarchy = cv2.findContours(referee_shorts.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


		contours, hierarchy = cv2.findContours(full_referee.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_referee, _ = drawPlayerOutline(contoured_frame, contours,"Green")
		
		# cv2.imwrite("Images\contoured referee frame "+ str(j)+ ".png",contoured_referee)

		#------------------------------------------------detect GoalKeeper----------------------------------------------------------
		# player_detected = detectColour(frame, "Green", THRESHOLD["Green"])
		# cv2.imwrite("Images\detect " + "greenplayers" + "frame "+ str(j)+ ".png",player_detected)
		# #player_other_part = detectColour(frame, "LowRangeDarkGreen", THRESHOLD["Green"])
		# #full_green_player = player_detected.astype(int) + player_other_part.astype(int)
		# #masked_wanted_points = np.ma.masked_where(full_green_player!=0, full_green_player)
		# #final_full_green_player = np.ma.filled(masked_wanted_points,255)
		# contours, hierarchy = cv2.findContours(player_detected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# contoured_frame_allplayers,_ = drawPlayerOutline(contoured_frame_players, contours, "Green")
		# cv2.imwrite("Images\detect " + "allplayers" + "frame "+ str(j)+ ".png",contoured_frame_allplayers)

		# green_players_limbs = blue_players_limbs
		# full_green_player = player_detected.astype(int) + green_players_limbs.astype(int)
		# masked_wanted_points = np.ma.masked_where(full_green_player!=0, full_green_player)
		# final_full_green_player = np.ma.filled(masked_wanted_points,255)

		# cv2.imwrite("Images\detect " +"greenplayers " + "frame "+ str(j)+ ".png",player_detected)

		# contours, hierarchy = cv2.findContours(final_full_green_player.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# contoured_frame_all_players,_ = drawPlayerOutline(contoured_frame_players, contours, "Green")
		# #cv2.imwrite("Images\detect " + "both teams and goal keeper" + "frame "+ str(j)+ ".png",contoured_frame_players)
		# cv2.imwrite("detect " + "both teams and goal keeper" + "frame "+ str(j)+ ".png",contoured_frame_all_players)

		j +=1

		if j == 20:
			exit()


	#cv2.destroyAllWindows()

