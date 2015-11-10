import cv2
import cv2.cv as cv
import numpy as np
# np.set_printoptions(threshold='nan')
# MIN_AREA_OF_CONTOUR_BOX = 200
DEBUG = 0


class PlayerDectector():
	CONTOUR_COLOUR = (255,255,255)
	CONTOUR_BTN_COLOUR = (0,0,255)
	THRESHOLD = {"Red":50 , "Blue":50, "Green" : 80}
	MIN_AREA_OF_CONTOUR_BOX = {"Red":100, "Blue":35, "Green":40}
	GAMMA = 3

	def __init__(self, backgrd_image):
		self.backgrd_image = self.adjustGamma(backgrd_image, self.GAMMA)
		# self.backgrd_image = cv2.imread("normImg.jpg")
		self.backgrd_blue = self.backgrd_image[:,:,0].astype(float)
		self.backgrd_green = self.backgrd_image[:,:,1].astype(float)
		self.backgrd_red = self.backgrd_image[:,:,2].astype(float)
		
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
		# colour_compare_one = np.ndarray((numRow,numCol))
		# colour_compare_two = np.ndarray((numRow,numCol))
		image_blue = frame[:,:,0].flatten()
		image_green = frame[:,:,1].flatten()
		image_red = frame[:,:,2].flatten()
		cmax = np.maximum(np.maximum(image_blue,image_green), image_red)

		if colour == "Red" :
			colour_To_Be_Search = image_red
			# colour_compare_one = image_blue
			# colour_compare_two = image_green
		elif colour == "Blue":
			colour_To_Be_Search = image_blue
			# colour_compare_one = image_red
			# colour_compare_two = image_green
		elif colour == "Green":
			colour_To_Be_Search = image_green
			# colour_compare_one = image_blue
			# colour_compare_two = image_red

		masked_unmax_pixels = np.ma.masked_where(cmax!=colour_To_Be_Search, colour_To_Be_Search)
		# masked_lowerintensity_pixels_one = np.ma.masked_where((masked_unmax_pixels-colour_compare_one<15), masked_unmax_pixels)
		# masked_lowerintensity_pixels_two = np.ma.masked_where( masked_unmax_pixels-colour_compare_two<15, masked_unmax_pixels)
		masked_lowerintensity_pixels_two = np.ma.masked_where( masked_unmax_pixels<self.THRESHOLD[colour], masked_unmax_pixels)
		masked_max_intensity_pixels= masked_lowerintensity_pixels_two/masked_lowerintensity_pixels_two * 255
		max_intensity_pixels = np.ma.filled(masked_max_intensity_pixels,0)

		return max_intensity_pixels.reshape(numRow,numCol)
	
	"""
	#This function takes in a list of contours and draw the minAreaBox of them onto the frame
	"""
	def drawPlayerOutline(self,frame,contours,colour):
		filled_contours=np.ndarray((frame.shape[0], frame.shape[1]))
		for i in xrange(0,len(contours)):
			cnt = contours[i]
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			if cv2.contourArea(cnt) > self.MIN_AREA_OF_CONTOUR_BOX[colour]:
				cv2.drawContours(frame,[box],0,self.CONTOUR_COLOUR,2)
				bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
				cv2.circle(frame, bottommost, 3,self.CONTOUR_BTN_COLOUR ,-1)
			else:
				cv2.drawContours(filled_contours,[box],0,self.CONTOUR_COLOUR,-1)
		return frame, filled_contours

	"""
	 This function remove the background using a precomputed background image which is divided into its 3 channels as input.
	 @para: frame: image array
	 		backgrd_red, backgrd_blue, backgrd_green : the 3 channels of the precomputed background image			
	 @returns: the foreground with RGB channels
	"""
	def backgroundSubtraction(self,frame, backgrd_red, backgrd_blue, backgrd_green):
		image_blue = frame[:,:,0].astype(float)
		image_green = frame[:,:,1].astype(float)
		image_red = frame[:,:,2].astype(float)
		diff_red = np.absolute(image_red - backgrd_red)/3.0
		diff_blue = np.absolute(image_blue - backgrd_blue)/3.0
		diff_green = np.absolute(image_green - backgrd_green)/3.0

		# trying = np.divide(diff_red,3)
		# trying2 = trying +trying
		# print "trying2", trying2
		# hahaha = 1/3.0*diff_red
		# hehehe = np.add(hahaha,1/3.0*diff_blue)
		# print "can print haha",hahaha
		# # lalla = 1/3.0*diff_blue
		# # hahaha + lalla
		diff_red = diff_red + diff_blue
		backgrd_mask = (diff_red + diff_green).astype(np.uint8)
		# print "can print",backgrd_mask
		_,foreground = cv2.threshold(backgrd_mask, 30, 255, cv2.THRESH_BINARY)
		cv2.imwrite("backgrd mask.jpg", foreground)
		three_chan_fore = np.dstack((foreground,foreground,foreground))
		colour_foregrd = cv2.bitwise_and(three_chan_fore, frame)
		return colour_foregrd

	def detectPlayers(self, frame, j):
		contoured_frame = np.copy(frame)

		#------------------------------------------------Do background Subtraction---------------------------------------------------
		foregrd_coloured = self.backgroundSubtraction(frame, self.backgrd_red, self.backgrd_blue, self.backgrd_green)
		cv2.imwrite("Contours\detect " + "after bgs" + "frame "+ str(j)+ ".jpg",foregrd_coloured)
		#------------------------------------------------detect Red Players----------------------------------------------------------
		red_player_detected = self.detectColour(foregrd_coloured, "Red", self.THRESHOLD["Red"])
		cv2.imwrite("Contours\Red\detect " + "red" + "frame "+ str(j)+ ".jpg",red_player_detected)
		contours, hierarchy = cv2.findContours(red_player_detected.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_red, blue_players_limbs = self.drawPlayerOutline(np.copy(contoured_frame), contours,"Red")
		cv2.imwrite("Contours\Red\Contoured Red Players frame "+ str(j)+ ".jpg",contoured_frame_red)

		#------------------------------------------------detect Blue Players----------------------------------------------------------
		player_detected = self.detectColour(foregrd_coloured, "Blue", self.THRESHOLD["Blue"])

		full_blue_player = player_detected.astype(int) + blue_players_limbs.astype(int)
		masked_wanted_points = np.ma.masked_where(full_blue_player!=0, full_blue_player)
		final_full_blue_player = np.ma.filled(masked_wanted_points,255)
		cv2.imwrite("Contours\Blue\\full blue"+ str(j)+ ".jpg",final_full_blue_player)
		contours, hierarchy = cv2.findContours(final_full_blue_player.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_blue,_ = self.drawPlayerOutline(np.copy(contoured_frame), contours, "Blue")
		cv2.imwrite("Contours\Blue\Contoured Blue Players frame "+ str(j)+ ".jpg",contoured_frame_blue)

		#------------------------------------------------detect Green Players----------------------------------------------------------
		both_teams_detected = (red_player_detected + final_full_blue_player).astype(np.uint8)

		inverse_both_teams_detected = (both_teams_detected*-1 + 255).astype(np.uint8) #cos -1 makes it int16
		three_chan_inver = np.dstack((inverse_both_teams_detected,inverse_both_teams_detected,inverse_both_teams_detected))
		green_players = cv2.bitwise_and(three_chan_inver, foregrd_coloured)
		green_channel = self.detectColour(green_players, "Green", self.THRESHOLD["Green"])
		contours, hierarchy = cv2.findContours(green_channel.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_green,_ = self.drawPlayerOutline(contoured_frame, contours, "Green")
		# cv2.imwrite("Contours\Green\Contoured Green Players frame " + str(j)+".jpg", contoured_frame_green)

	def obtainBackgrd(self, video_dir):
		cap = cv2.VideoCapture(video_dir)
		width, height, fps, frames_count = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT),  cap.get(cv.CV_CAP_PROP_FPS), cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

		width = int(width)
		height = int(height)
		fps = int(fps)
		frames_count = int(frames_count)

		_,img = cap.read()
		avgImg = np.float32(img)
		normImg = np.float32(img)

		for fr in range(1,4000):
			# print fr
			_,img = cap.read()
			Img_ApdatedBG =((fr-1.0)/fr)*avgImg+(1.0/fr)*np.float32(img); #using fixed alpha would mean that when frame count increases, it means that future Contours are weighted too much.
			avgImg = Img_ApdatedBG;
			normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image
			print "running average frame " + str(fr)
		cv2.imwrite('normImg.jpg', normImg)
		return normImg

	def adjustGamma(self, image, gamma=1.0):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	 
		# apply gamma correction using the lookup table
		return cv2.LUT(image, table)

if DEBUG :
	j=0
	detPlayer = PlayerDectector("football_panorama.MOV")
	capture = cv2.VideoCapture('football_panorama.MOV')
	while capture.isOpened():
		print "processing frame " + str(j)
		_, frame = capture.read()
		detPlayer.detectPlayers(frame, j)
		j +=1
		if j == 5:
			exit()

















