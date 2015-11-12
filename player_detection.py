import cv2
import cv2.cv as cv
import numpy as np
import math
# np.set_printoptions(threshold='nan')
# MIN_AREA_OF_CONTOUR_BOX = 200
DEBUG = 1


class PlayerDectector():
	CONTOUR_COLOUR = (255,255,255)
	CONTOUR_BTN_COLOUR = (0,0,255)
	ICON_COLOUR = {"Red":(0,0,255), "Blue":(255,0,0), "Green": (0,255,0)}
	THRESHOLD = {"Red":50 , "Blue":50, "Green" : 80, "White" : 120, "Black":50}
	MIN_DIFF = {"Red":15 , "Blue":0, "Green" : 20}
	MIN_AREA_OF_CONTOUR_BOX = {"Red":200, "Blue":180, "Green":40, "White":200}
	# GAMMA = 1
	BACKGRD_THRESHOLD = 20
	BLUE_THRESHOLD = 20
	WHITE_OFFSET = 25 #to offset the y coordinate of the white goal keeper
	BLUE_GOLIE_OFFSET = 20
	REFEREE_OFFSET = 30
	MAX_PIXEL_MOVED = 30
	CLOSENESS_VALUE = 60
	BIG_NUMBER = 100000000
	def __init__(self, backgrd_image):
		# self.backgrd_image = self.adjustGamma(backgrd_image, self.GAMMA)
		self.backgrd_image = backgrd_image
		# self.backgrd_image = cv2.imread("normImg.jpg")
		# self.backgrd_image = cv2.medianBlur(backgrd_image,9)
		self.backgrd_blue = self.backgrd_image[:,:,0].astype(float)
		self.backgrd_green = self.backgrd_image[:,:,1].astype(float)
		self.backgrd_red = self.backgrd_image[:,:,2].astype(float)
		self.mean_kernel = np.ones((5,5),np.float32)/25
		self.red_position = [(373,4682), (271,4822), (309,5085), (377,5166), (578,5244), (316,5337), (275,5415), (490,6028)]#position for frame 0
		# self.red_position = [(373,4682), (271,4822), (309,5085), (377,5166), (346,5357), (316,5337), (275,5415), (490,6028)]#test overlap
		# self.red_position = [(381,4693),(276,4790), (327,5096), (408,5325), (603,5314), (319,5408), (281,5427), (556,6939)] #position for frame 74
		self.red_goalie_position = (377,3660)
		# self.red_nearby = self.nearbyPlayer(self.red_position)
		self.blue_position = [(385,4748),  (460,5127), (287,5243), (346,5357), (322,5430), (344,5478), (291,5489), (347,5637), (369,5750), (459,6011)]
		self.blue_golie_postion = (362,6256)
		# self.blue_nearby = self.nearbyPlayer(self.blue_position)
		self.referee_position = (340,5231)

	# def nearbyPlayer(self, player_position):
	# 	close_array = []
	# 	for i in range(len(player_position)):
	# 		close_array.append(-1)	
	# 	for j in range(len(player_position)):
	# 		min_dist = self.BIG_NUMBER
	# 		min_index = -1
	# 		for k in range(len(player_position)):
	# 			dist = self.distance(player_position[j], player_position[k])
	# 			if dist < min_dist and dist!=0:
	# 				min_dist = dist
	# 				min_index = k
	# 		if min_dist < self.CLOSENESS_VALUE : #assign the closer player index to the player in question
	# 			close_array[j] = min_index
	# 	# print close_array
	# 	if(sum(close_array) != len(player_position)*-1):
	# 		print close_array
	# 	return close_array

	"""
	 this function takes in a color image frame, detects the colour to be detected, and return a black and white image whereby white parts are where the color is found
	 color detection is based on the threshold, and whether the color channel is the max among the rest of the channel so as to detect more color gradient
	 @para: frame: image array
			colour: "Blue/Red/Green"
			threshold: the minimum colour value of a pixel to be set to max
	 @returns: an image with color intensified(set to 255 for one of the 3 colour channel)
	"""
	def detectColour(self,frame, colour, threshold):
		numRow, numCol, numRGB = frame.shape
		colour_To_Be_Search = np.ndarray((numRow,numCol))
		image_blue = frame[:,:,0].flatten()
		image_green = frame[:,:,1].flatten()
		image_red = frame[:,:,2].flatten()
		cmax = np.maximum(np.maximum(image_blue,image_green), image_red)

		if colour == "White" :
			masked_blue = np.ma.masked_where(frame[:,:,0]<self.THRESHOLD[colour], colour_To_Be_Search)
			masked_red = np.ma.masked_where(frame[:,:,1]<self.THRESHOLD[colour], masked_blue)
			masked_all_color = np.ma.masked_where(frame[:,:,2]<self.THRESHOLD[colour], masked_red)
			intensified_white_player = masked_all_color + 255
			complete_intensified_white_player = np.ma.filled(intensified_white_player,0)
			smooth_image = self.connectDots(complete_intensified_white_player)
			#check black pants 
			# cv2.imwrite("frame"+ ".jpg", frame)
			# print colour_To_Be_Search
			# bmasked_blue = np.ma.masked_where(frame[:,:,0]>self.THRESHOLD["Black"], colour_To_Be_Search)
			# bmasked_red = np.ma.masked_where(frame[:,:,1]>self.THRESHOLD["Black"], bmasked_blue)
			# bmasked_all_color = np.ma.masked_where(frame[:,:,2]>self.THRESHOLD["Black"], bmasked_red)
			# bintensified_white_player = bmasked_all_color + 250
			# bcomplete_intensified_white_player = np.ma.filled(bintensified_white_player,0)
			# cv2.imwrite("Contours\White\white players "+ ".jpg", self.connectDots(complete_intensified_white_player))
			return smooth_image
			
		else :	
			if colour == "Red" :
				colour_To_Be_Search = image_red
				colour_compare_one = image_blue
				colour_compare_two = image_green
			elif colour == "Blue":
				colour_To_Be_Search = image_blue
				colour_compare_one = image_red
				colour_compare_two = image_green
			elif colour == "Green":
				colour_To_Be_Search = image_green
				colour_compare_one = image_blue
				colour_compare_two = image_red

			masked_unmax_pixels = np.ma.masked_where(cmax!=colour_To_Be_Search, colour_To_Be_Search)
			masked_threshold = np.ma.masked_where( masked_unmax_pixels<self.THRESHOLD[colour], masked_unmax_pixels)
			masked_lowerintensity_pixels_one = np.ma.masked_where((masked_unmax_pixels-colour_compare_one<self.MIN_DIFF[colour]), masked_threshold)
			masked_lowerintensity_pixels_two = np.ma.masked_where( masked_unmax_pixels-colour_compare_two<self.MIN_DIFF[colour], masked_threshold)
			masked_max_intensity_pixels_one = masked_lowerintensity_pixels_one/masked_lowerintensity_pixels_one * 255
			masked_max_intensity_pixels_two = masked_lowerintensity_pixels_two/masked_lowerintensity_pixels_two * 255
			
			max_intensity_pixels_one = np.ma.filled(masked_max_intensity_pixels_one,0)
			max_intensity_pixels_two = np.ma.filled(masked_max_intensity_pixels_two,0)
			max_intensity_pixels = cv2.bitwise_and(max_intensity_pixels_one, max_intensity_pixels_two)



			return max_intensity_pixels.reshape(numRow,numCol)
	
	"""
	#This function takes in a list of contours and draw the minAreaBox of them onto the frame
	TODO: remove the last return parameter cos not used anymore------------------------------------------------------------------------------------------------------------------
	"""
	def drawPlayerOutline(self,frame,contours,colour):
		filled_contours=np.ndarray((frame.shape[0], frame.shape[1]))
		players_coordinate = []
		for i in xrange(0,len(contours)):
			cnt = contours[i]
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			if cv2.contourArea(cnt) > self.MIN_AREA_OF_CONTOUR_BOX[colour]:
				cv2.drawContours(frame,[box],0,self.CONTOUR_COLOUR,2)
				bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
				players_coordinate.append((bottommost[1],bottommost[0]))
				cv2.circle(frame, bottommost, 3,self.CONTOUR_BTN_COLOUR ,-1)
			else:
				cv2.drawContours(filled_contours,[box],0,self.CONTOUR_COLOUR,-1)
		return frame, filled_contours,players_coordinate

	"""
	 This function remove the background using a precomputed background image which is divided into its 3 channels as input.
	 @para: frame: image array
	 		backgrd_red, backgrd_blue, backgrd_green : the 3 channels of the precomputed background image			
	 @returns: the foreground with RGB channels
	"""
	def backgroundSubtraction(self,frame, backgrd_red, backgrd_blue, backgrd_green, j):
		image_blue = frame[:,:,0].astype(np.float32)
		image_green = frame[:,:,1].astype(np.float32)
		image_red = frame[:,:,2].astype(np.float32)
		diff_red = np.absolute(image_red - backgrd_red)/3.0
		diff_blue = np.absolute(image_blue - backgrd_blue)/3.0
		diff_green = np.absolute(image_green - backgrd_green)/3.0

		diff_red = diff_red + diff_blue
		backgrd_mask = (diff_red + diff_green).astype(np.uint8)
		# print "can print",backgrd_mask
		_,foreground = cv2.threshold(backgrd_mask, self.BACKGRD_THRESHOLD, 255, cv2.THRESH_BINARY) # to get a rough binary image of the foreground
		# cv2.imwrite("Contours\\backgrdmask\\backgrd mask " + str(j) + ".jpg", foreground)
		# cv2.imwrite("Contours\\backgrdmask\\backgrd mask filtered " + str(j) + ".jpg", cv2.medianBlur(foreground,5))
		blur_image = cv2.GaussianBlur(foreground,(5,5),0) #to smoothen the noise of the foreground
		_,new_foreground = cv2.threshold(blur_image, self.BACKGRD_THRESHOLD, 255, cv2.THRESH_BINARY) # to connect more points of the player
		remove_ring = cv2.medianBlur(new_foreground, 7) # remove the middle ring of the field
		# cv2.imwrite("Contours\\backgrdmask\\backgrd mask filtered " + str(j) + ".jpg",remove_ring)
		three_chan_fore = np.dstack((remove_ring,remove_ring,remove_ring))
		colour_foregrd = cv2.bitwise_and(three_chan_fore, frame)
		return colour_foregrd

	def drawPlayersPosition(self, frame, coordinates, colour):
		for coordinate in coordinates:
			coordinate = (coordinate[1], coordinate[0])
			cv2.circle(frame, coordinate, 3,self.ICON_COLOUR[colour] ,-1)
		return frame

	def distance(self, p0, p1):
	    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

	#This function takes in a set of contours and a coordinate of the player, and returns the closest coordinate point that is on the contour.
	#it returns a coordinate and the contour that the coordinate belongs to
	def closestCoordinate(self, coordinate, contour_set):
		min_dist_contour = self.BIG_NUMBER
		min_coor_contour = (-1,-1)
		contour_found = [[[]]]
		for cnt in contour_set:
			for coor in cnt:
				dist = self.distance(tuple(reversed(coor[0])), coordinate)
				if dist < min_dist_contour and dist < self.MAX_PIXEL_MOVED:
					min_dist_contour = dist
					min_coor_contour = tuple(reversed(coor[0]))
					contour_found = cnt
		return min_coor_contour, contour_found
	def detectPlayers(self, frame, j):
		contoured_frame = np.copy(frame)
		# frame = self.adjustGamma(frame, self.GAMMA)
		# frame = cv2.filter2D(frame, -1, self.mean_kernel)
		# frame = cv2.medianBlur(frame,3)

		#------------------------------------------------Do background Subtraction---------------------------------------------------
		foregrd_coloured = self.backgroundSubtraction(frame, self.backgrd_red, self.backgrd_blue, self.backgrd_green, j)
		# cv2.imwrite("Contours\\removebgrd\detect " + "after bgs" + "frame "+ str(j)+ ".jpg",foregrd_coloured)

		#------------------------------------------------detect White Players---------------------------------------------------------- 
		white_player_detected = self.detectColour(foregrd_coloured, "White", self.THRESHOLD["White"])
		# cv2.imwrite("Contours\White\detect " + "White" + "frame "+ str(j)+ ".jpg",white_player_detected)
		contours, hierarchy = cv2.findContours(white_player_detected.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_white,_,white_coordinates = self.drawPlayerOutline(np.copy(contoured_frame), contours,"White")
		# cv2.imwrite("Contours\White\Contoured White Players frame "+ str(j)+ ".jpg",contoured_frame_white)
		
		self.red_goalie_position,_ = self.closestCoordinate(self.red_goalie_position, contours)
		real_position = (self.red_goalie_position[0] + self.WHITE_OFFSET, self.red_goalie_position[1])
		# print self.red_goalie_position
		icon_frame = self.drawPlayersPosition(frame, [real_position], "Red")
		# cv2.imwrite("PlayerPosition\White\White Players frame "+ str(j)+ ".jpg",icon_frame)

		#------------------------------------------------detect Red Players----------------------------------------------------------
		red_player_detected = self.detectColour(foregrd_coloured, "Red", self.THRESHOLD["Red"])
		# cv2.imwrite("Contours\Red\detect " + "red" + "frame "+ str(j)+ ".jpg",red_player_detected)
		contours, hierarchy = cv2.findContours(red_player_detected.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_red, blue_players_limbs, red_coordinates = self.drawPlayerOutline(np.copy(contoured_frame), contours,"Red")
		cv2.imwrite("Contours\Red\Contoured Red Players frame "+ str(j)+ ".jpg",contoured_frame_red)
		new_red_position = []
		#find new position of red players using the contours from the new frame
		for position in self.red_position:
			new_position,contour_found = self.closestCoordinate(position, contours) # to find one coordinate from all contours that is the closest to previous position
			# print "before missing stuff"
			if new_position == (-1,-1):
				index_of_player =  self.red_position.index(position)
				new_red_position.append(position)
				print "missing contour " + str(index_of_player)
			else :
				new_red_position.append(new_position)
		self.red_position = new_red_position
		icon_frame_2 = self.drawPlayersPosition(icon_frame, self.red_position, "Blue") #printing blue on red player becos we cant see red on red
		cv2.imwrite("PlayerPosition\Red\Red Players frame "+ str(j)+ ".jpg",icon_frame_2)

		#------------------------------------------------detect Blue Players----------------------------------------------------------
		player_detected = self.detectColour(foregrd_coloured, "Blue", self.THRESHOLD["Blue"])
		# cv2.imwrite("Contours\Blue\player dected" + str(j) + ".jpg", player_detected)
		full_blue_player = player_detected.astype(int) + blue_players_limbs.astype(int)
		masked_wanted_points = np.ma.masked_where(full_blue_player!=0, full_blue_player)
		noisy_full_blue_player = np.ma.filled(masked_wanted_points,255)
		# cv2.imwrite("Contours\Blue\\full blue"+ str(j)+ ".jpg",noisy_full_blue_player)
		#connect the white dots of the player
		gaus_blue_player = cv2.GaussianBlur(noisy_full_blue_player.astype(np.uint8), (5,5), 0)
		_,final_full_blue_player = cv2.threshold(gaus_blue_player, 20, 255, cv2.THRESH_BINARY) # to connect more points of the player
		median_blue_player = cv2.medianBlur(final_full_blue_player, 7) 
		# cv2.imwrite("Contours\Blue\\full blue with connectivity"+ str(j)+ ".jpg",median_blue_player)
		
		contours, hierarchy = cv2.findContours(median_blue_player.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_blue,_,blue_coordinate = self.drawPlayerOutline(np.copy(contoured_frame), contours, "Blue")
		cv2.imwrite("Contours\Blue\Contoured Blue Players frame "+ str(j)+ ".jpg",contoured_frame_blue)
		new_blue_position = []
		#find new position of blue players using the contours from the new frame
		for position in self.blue_position:
			new_position,contour_found = self.closestCoordinate(position, contours) # to find one coordinate from all contours that is the closest to previous position
			# print "before missing stuff"
			if new_position == (-1,-1):
				index_of_player =  self.blue_position.index(position)
				new_blue_position.append(position)
				print "missing contour " + str(index_of_player)
			else :
				new_blue_position.append(new_position)
		self.blue_position = new_blue_position
		icon_frame_3 = self.drawPlayersPosition(icon_frame_2, self.blue_position, "Red") #printing blue on red player becos we cant see red on red
		cv2.imwrite("PlayerPosition\Blue\Blue Players frame "+ str(j)+ ".jpg",icon_frame_3)
		# cv2.imwrite("Contours\Blue\Contoured Blue Players frame "+ str(j)+ ".jpg",contoured_frame_blue)

		#------------------------------------------------detect Green Players----------------------------------------------------------
		both_teams_detected = (red_player_detected + median_blue_player).astype(np.uint8)

		inverse_both_teams_detected = (both_teams_detected*-1 + 255).astype(np.uint8) #cos -1 makes it int16
		three_chan_inver = np.dstack((inverse_both_teams_detected,inverse_both_teams_detected,inverse_both_teams_detected))
		green_players = cv2.bitwise_and(three_chan_inver, foregrd_coloured)
		green_channel = self.detectColour(green_players, "Green", self.THRESHOLD["Green"])
		contours, hierarchy = cv2.findContours(green_channel.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_green,_,green_coordinate = self.drawPlayerOutline(np.copy(contoured_frame), contours, "Green")
		cv2.imwrite("Contours\Green\Contoured Green Players frame " + str(j)+".jpg", contoured_frame_green)
		#-----------Blue team goalie-----------------------
		blue_golie_new_position, contour_found = self.closestCoordinate(self.blue_golie_postion, contours)
		if blue_golie_new_position == (-1,-1):
			blue_golie_new_position = self.blue_golie_postion
			print "missing contour for blue golie" 
		self.blue_golie_postion = blue_golie_new_position
		real_blue_golie_position = (self.blue_golie_postion[0] + self.BLUE_GOLIE_OFFSET, self.blue_golie_postion[1])
		icon_frame_4 = self.drawPlayersPosition(icon_frame_3, [real_blue_golie_position], "Red") #printing blue on red player becos we cant see red on red
		cv2.imwrite("PlayerPosition\BlueGolie\Blue golie frame "+ str(j)+ ".jpg",icon_frame_4)

		#-----------Referee--------------------------
		referee_new_position, contour_found = self.closestCoordinate(self.referee_position, contours)
		if referee_new_position == (-1,-1):
			referee_new_position = self.referee_position
			print "missing contour for blue golie" 
		self.referee_position = referee_new_position
		real_referee_position = (self.referee_position[0] + self.REFEREE_OFFSET, self.referee_position[1])
		icon_frame_5 = self.drawPlayersPosition(icon_frame_4, [real_referee_position], "Red") #printing blue on red player becos we cant see red on red
		cv2.imwrite("PlayerPosition\Referee\Referee frame "+ str(j)+ ".jpg",icon_frame_5)
		return self.red_position, self.red_goalie_position, self.blue_position, self.blue_golie_postion, self.referee_position

	def connectDots(self, frame):
		gaus_blur = cv2.GaussianBlur(frame.astype(np.uint8), (5,5), 0)
		_,thresholding = cv2.threshold(gaus_blur, 20, 255, cv2.THRESH_BINARY) # to connect more points of the player
		final_frame = cv2.medianBlur(thresholding, 7) 
		return final_frame

	# def obtainBackgrd(self, video_dir):
	# 	cap = cv2.VideoCapture(video_dir)
	# 	width, height, fps, frames_count = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT),  cap.get(cv.CV_CAP_PROP_FPS), cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

	# 	width = int(width)
	# 	height = int(height)
	# 	fps = int(fps)
	# 	frames_count = int(frames_count)

	# 	_,img = cap.read()
	# 	avgImg = np.float32(img)
	# 	normImg = np.float32(img)

	# 	for fr in range(1,4000):
	# 		# print fr
	# 		_,img = cap.read()
	# 		Img_ApdatedBG =((fr/fr+1.0)*avgImg)+(1.0/fr+1)*np.float32(img) #using fixed alpha would mean that when frame count increases, it means that future Contours are weighted too much.
	# 		avgImg = Img_ApdatedBG
	# 		normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image
	# 		print "running average frame " + str(fr)
	# 	cv2.imwrite('normImg.jpg', normImg)
	# 	return normImg

	# def adjustGamma(self, image, gamma=1.0):
	# 	# build a lookup table mapping the pixel values [0, 255] to
	# 	# their adjusted gamma values
	# 	invGamma = 1.0 / gamma
	# 	table = np.array([((i / 255.0) ** invGamma) * 255
	# 		for i in np.arange(0, 256)]).astype("uint8")
	 
	# 	# apply gamma correction using the lookup table
	# 	return cv2.LUT(image, table)
# DEBUG = 0
if DEBUG :

	detect_player = PlayerDectector(cv2.imread("backgrd.png"))
	# print detect_player.distance(detect_player.blue_position[4], detect_player.blue_position[3])
	for j in range(0, 7000):
		print "running frame " + str(j)
		frame = cv2.imread("C:\Users\weijian\Desktop\FullSize\panorama_frame_ " + str(j) +".jpg")
		red_position, red_goalie_position, blue_position, blue_golie_postion, referee_position = detect_player.detectPlayers(frame, j)
		print "red position ", red_position
		print "red golie", red_goalie_position
		print "blue position", blue_position
		print "blue goalie", blue_golie_postion
		print "referee", referee_position













