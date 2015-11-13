import cv2
import cv2.cv as cv
import numpy as np
from math import sqrt
import ast
import time
# np.set_printoptions(threshold='nan')
# MIN_AREA_OF_CONTOUR_BOX = 200
DEBUG = 1

WHITE_OFFSET = 25 
BLUE_GOLIE_OFFSET = 15 
REFEREE_OFFSET = 30
BLUE_PLAYER_OFFSET = 30
RED_PLAYER_OFFSET = 30

class PlayerDectector():
	SAVE_FREQUENCY = 1
	CONTOUR_COLOUR = (255,255,255)
	CONTOUR_BTN_COLOUR = (0,0,255)
	ICON_COLOUR = {"Red":(0,0,255), "Blue":(255,0,0), "Green": (0,255,0), "Yellow":(0,255,255)}
	THRESHOLD = {"Red":50 , "Blue":50, "Green" : 80, "White" : 120, "Black":50}
	MIN_DIFF = {"Red":15 , "Blue":0, "Green" : 20}
	MIN_AREA_OF_CONTOUR_BOX = {"Red":210, "Blue":150, "Green":40, "White":200}
	# GAMMA = 1
	BACKGRD_THRESHOLD = 20
	BLUE_THRESHOLD = 20
	#to offset the y coordinate of the white goal keeper
	#BLUE_GOLIE_OFFSET = 20 #some parts went off previously hence we are assigning a smaller value now 
	
	
	
	MAX_PIXEL_MOVED = 80
	BIG_NUMBER = 100000000
	def __init__(self, backgrd_image, red_pos, red_goalie_post, blue_pos, blue_golie_pos, referee_pos):
		self.backgrd_image = backgrd_image
		self.backgrd_blue = self.backgrd_image[:,:,0].astype(float)
		self.backgrd_green = self.backgrd_image[:,:,1].astype(float)
		self.backgrd_red = self.backgrd_image[:,:,2].astype(float)
		self.mean_kernel = np.ones((5,5),np.float32)/25
		self.red_position = red_pos
		# self.red_centroid = red_centroid
		self.red_old_centroid_overlap = [(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1)] #each is a coor ,used to track how much does the new overlapped centroid move
		self.red_goalie_position = red_goalie_post
		self.blue_position = blue_pos
		# self.blue_centroid = blue_centroid
		self.blue_old_centroid_overlap = [(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1), (-1,-1), (-1,-1)]
		self.blue_golie_postion = blue_golie_pos
		self.referee_position = referee_pos

		self.red_directionchange = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
		self.blue_directionchange = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
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
		contours_found = []
		for i in xrange(0,len(contours)):
			cnt = contours[i]
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			if cv2.contourArea(cnt) > self.MIN_AREA_OF_CONTOUR_BOX[colour]:
				cv2.drawContours(frame,[box],0,self.CONTOUR_COLOUR,2)
				bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
				contours_found.append(cnt)
				cv2.circle(frame, bottommost, 3,self.CONTOUR_BTN_COLOUR ,-1)
			else:
				cv2.drawContours(filled_contours,[box],0,self.CONTOUR_COLOUR,-1)
		return frame, filled_contours,contours_found

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
	    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

	#This function takes in a set of contours and a coordinate of the player, and returns the closest coordinate point that is on the contour.
	#it returns a coordinate and the contour that the coordinate belongs to
	def closestCoordinate(self, coordinate, contour_set):
		min_dist_contour = self.BIG_NUMBER
		min_coor_centre = (-1,-1)
		contour_found = [[[]]]
		for cnt in contour_set:
			#for coor in cnt:
			M = cv2.moments(cnt)
			new_centroid = (int(M['m01']/M['m00']),int(M['m10']/M['m00']))
			dist = self.distance(new_centroid, coordinate)
			if dist < min_dist_contour and dist < self.MAX_PIXEL_MOVED:
				min_dist_contour = dist
				min_coor_centre = new_centroid
				contour_found = cnt
		return min_coor_centre, contour_found #min coor here is the centre point of the contour_found 
	
	def detectPlayers(self, frame, j):
		contoured_frame = np.copy(frame)
		t0 = time.clock()
		#------------------------------------------------Do background Subtraction---------------------------------------------------
		foregrd_coloured = self.backgroundSubtraction(frame, self.backgrd_red, self.backgrd_blue, self.backgrd_green, j)
		# cv2.imwrite("Contours\\removebgrd\detect " + "after bgs" + "frame "+ str(j)+ ".jpg",foregrd_coloured)
		# print "Time taken to get foregrd" + str(time.clock() - t0)
		t0 = time.clock()
		#------------------------------------------------detect White Players---------------------------------------------------------- 
		white_player_detected = self.detectColour(foregrd_coloured, "White", self.THRESHOLD["White"])
		# cv2.imwrite("Contours\White\detect " + "White" + "frame "+ str(j)+ ".jpg",white_player_detected)
		contours, hierarchy = cv2.findContours(white_player_detected.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_white,_,contours_found_white = self.drawPlayerOutline(np.copy(contoured_frame), contours,"White")
		# cv2.imwrite("Contours\White\Contoured White Players frame "+ str(j)+ ".jpg",contoured_frame_white)
		

		red_golie_new_position, contour_found = self.closestCoordinate(self.red_goalie_position, contours_found_white)
		if red_golie_new_position == (-1,-1):
			red_golie_new_position = self.red_goalie_position
			print "missing red goalie contour " 
		self.red_goalie_position = red_golie_new_position
		real_position = (self.red_goalie_position[0] + WHITE_OFFSET, self.red_goalie_position[1])

		icon_frame = self.drawPlayersPosition(frame, [real_position], "Red")
		# if j%self.SAVE_FREQUENCY == 0: 
		# 	cv2.imwrite("PlayerPosition\White\White Players frame "+ str(j)+ ".jpg",icon_frame)
		# print "Time taken to get white player position " + str(time.clock() - t0)
		
		#------------------------------------------------detect Red Players----------------------------------------------------------
		t0 = time.clock()
		red_player_detected = self.detectColour(foregrd_coloured, "Red", self.THRESHOLD["Red"])
		# cv2.imwrite("Contours\Red\detect " + "red" + "frame "+ str(j)+ ".jpg",red_player_detected)
		contours, hierarchy = cv2.findContours(red_player_detected.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_red, blue_players_limbs, contours_found_red = self.drawPlayerOutline(np.copy(contoured_frame), contours,"Red")
		if j%self.SAVE_FREQUENCY == 0:
			cv2.imwrite("Contours\Red\Contoured Red Players frame "+ str(j)+ ".jpg",contoured_frame_red)
		new_red_position = []
		contour_used = np.full((1,len(contours_found_red)), -1)[0].astype(np.int) #which player is using which contour to track for overlaps

		#find new position of red players using the contours from the new frame
		for position in self.red_position:
			index_of_player =  self.red_position.index(position)
			centroid_assigned = False
			contour_to_check = np.copy(contours_found_red)
			#first find a contour that matches 
			while not centroid_assigned:
				new_position,contour_found = self.closestCoordinate(position, contour_to_check) # to find one coordinate from all contours that is the closest to previous position
				
				#if i cannot find a close match centre after iterating through the whole set of contours_found_red, means my contour missing
				#new position refers to the centre of contour 
				if new_position == (-1,-1):
					new_position = (position[0] + self.red_directionchange[index_of_player][0], position[1] + self.red_directionchange[index_of_player][1])
					new_red_position.append(new_position)
					self.red_old_centroid_overlap[index_of_player] = (-1,-1)
					print "missing red contour " + str(index_of_player)
					centroid_assigned = True
				else :
					#check if the new centroid is near enough to the old centroid 
					index_of_contour_found = -1
					for i in range(len(contour_to_check)):
						M = cv2.moments(contour_to_check[i])
						this_centroid = (int(M['m01']/M['m00']),int(M['m10']/M['m00']))
						if this_centroid == new_position:
							index_of_contour_found = i
							break
					print new_position
					print self.red_position[index_of_player]
					if self.distance(new_position, self.red_position[index_of_player]) < 30:
						#check if contour is used, case for overlap 
						if contour_used[index_of_contour_found] == -1: #means is not used by someone else yet
							new_red_position.append(new_position)
							self.red_old_centroid_overlap[index_of_player] = (-1,-1)
							#udate the direction change when we got confirmed new position
							self.red_directionchange[index_of_player][0] = new_position[0] - self.red_position[index_of_player][0]
							self.red_directionchange[index_of_player][0] = new_position[0] - self.red_position[index_of_player][0]
							centroid_assigned = True
							contour_used[index_of_contour_found] = index_of_player
							# print "found normally red player"+ str(index_of_player)
						else: #means contour alr taken by someone else
							#calculate offset of overlap centroid based on my new overlap centroid
							overlap_centroid_offset = [0,0]
							if self.red_old_centroid_overlap[index_of_player] != (-1,-1): #means that prev frame there is overlap alr
								overlap_centroid_offset[0] = new_position[0] - self.red_old_centroid_overlap[index_of_player][0] #take new - old
								overlap_centroid_offset[1] = new_position[1] - self.red_old_centroid_overlap[index_of_player][1]
							print "*"*40, overlap_centroid_offset
							#reset that other person's temp new position
							someone_else_old_position = self.red_position[contour_used[index_of_contour_found]]
							my_old_position = self.red_position[index_of_player]
							# someone_else_new_coor, _= self.closestCoordinate(someone_else_centroid, [contour_found])
							#because the other person claim control of the new contour, he got the new centriod, here we are changing it back to the old centroid plus some offset
							new_red_position[contour_used[index_of_contour_found]] = (someone_else_old_position[0] + overlap_centroid_offset[0],someone_else_old_position[1]+overlap_centroid_offset[1])
							#settle my new position based on some offset to my previous position
							new_red_position.append((my_old_position[0] + overlap_centroid_offset[0],my_old_position[1]+overlap_centroid_offset[1]))
							centroid_assigned = True
							#set that these indexes of players uses the overlapped contour
							self.red_old_centroid_overlap[index_of_player] = new_position
							self.red_old_centroid_overlap[contour_used[index_of_contour_found]] = new_position
							print "found someone who share red contour with previous another one"
					else:
						print "found a contour but not what i want, Red player "+ str(index_of_player)
						# print new_position, self.red_position[index_of_player]
						contour_to_check = np.delete(contour_to_check,index_of_contour_found,0)
		self.red_position = new_red_position
		# print self,red_centroid
		icon_frame_2 = self.drawPlayersPosition(icon_frame, self.red_position, "Yellow") #printing blue on red player becos we cant see red on red
		real_red_position = np.add(np.array(self.red_position),[RED_PLAYER_OFFSET,0])
		icon_frame_2 = self.drawPlayersPosition(icon_frame_2, real_red_position, "Blue")
		# if j%self.SAVE_FREQUENCY:
		# 	cv2.imwrite("PlayerPosition\Red\Red Players frame "+ str(j)+ ".jpg",icon_frame_2)
		# print "Time taken to get red player position " + str(time.clock() - t0)
		#------------------------------------------------detect Blue Players----------------------------------------------------------
		t0 = time.clock()
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
		contoured_frame_blue,_,contours_found_blue = self.drawPlayerOutline(np.copy(contoured_frame), contours, "Blue")
		if j%self.SAVE_FREQUENCY == 0:
			cv2.imwrite("Contours\Blue\Contoured Blue Players frame "+ str(j)+ ".jpg",contoured_frame_blue)
		new_blue_position = []
		contour_used = np.full((1,len(contours_found_blue)), -1)[0].astype(np.int) #which player is using which contour to track for overlaps

		#find new position of red players using the contours from the new frame
		for position in self.blue_position:
			index_of_player =  self.blue_position.index(position)
			centroid_assigned = False
			contour_to_check = np.copy(contours_found_blue)
			#first find a contour that matches 
			while not centroid_assigned:
				new_position,contour_found = self.closestCoordinate(position, contour_to_check) # to find one coordinate from all contours that is the closest to previous position
				
				#if i cannot find a close match centre after iterating through the whole set of contours_found_red, means my contour missing
				#new position refers to the centre of contour 
				if new_position == (-1,-1):
					new_position = (position[0] + self.blue_directionchange[index_of_player][0], position[1] + self.blue_directionchange[index_of_player][1])
					new_blue_position.append(new_position)
					# new_red_centroid.append(self.red_centroid[index_of_player]) # assign back the same centroid
					self.blue_old_centroid_overlap[index_of_player] = (-1,-1)
					print "missing blue contour " + str(index_of_player)
					centroid_assigned = True
				else :
					#check if the new centroid is near enough to the old centroid 
					# M = cv2.moments(contour_found)
					# new_centroid = (int(M['m01']/M['m00']),int(M['m10']/M['m00']))
					index_of_contour_found = -1
					for i in range(len(contour_to_check)):
						M = cv2.moments(contour_to_check[i])
						this_centroid = (int(M['m01']/M['m00']),int(M['m10']/M['m00']))
						if this_centroid == new_position:
							index_of_contour_found = i
							break
					if self.distance(new_position, self.blue_position[index_of_player]) < 50:
						#check if contour is used, case for overlap 
						if contour_used[index_of_contour_found] == -1: #means is not used by someone else yet
							new_blue_position.append(new_position)
							self.blue_old_centroid_overlap[index_of_player] = (-1,-1)
							self.blue_directionchange[index_of_player][0] = new_position[0] - self.blue_position[index_of_player][0]
							self.blue_directionchange[index_of_player][0] = new_position[0] - self.blue_position[index_of_player][0]
							centroid_assigned = True
							contour_used[index_of_contour_found] = index_of_player
							# print "found normally blue player"+ str(index_of_player)
						else: #means contour alr taken by someone else
							#calculate offset of overlap centroid based on my new overlap centroid
							overlap_centroid_offset = [0,0]
							if self.blue_old_centroid_overlap[index_of_player] != (-1,-1): #means that prev frame there is overlap alr
								overlap_centroid_offset[0] = new_position[0] - self.blue_old_centroid_overlap[index_of_player][0] #take new - old
								overlap_centroid_offset[1] = new_position[1] - self.blue_old_centroid_overlap[index_of_player][1]
							print "*"*40, overlap_centroid_offset
							#reset that other person's temp new position
							someone_else_old_position = self.blue_position[contour_used[index_of_contour_found]]
							my_old_position = self.blue_position[index_of_player]							
							#because the other person claim control of the new contour, he got the new centriod, here we are changing it back to the old centroid plus some offset
							new_blue_position[contour_used[index_of_contour_found]] = (someone_else_old_position[0] + overlap_centroid_offset[0],someone_else_old_position[1]+overlap_centroid_offset[1])
							#settle my new position based on some offset to my previous position
							new_blue_position.append((my_old_position[0] + overlap_centroid_offset[0],my_old_position[1]+overlap_centroid_offset[1]))
							centroid_assigned = True
							#set that these indexes of players uses the overlapped contour
							self.blue_old_centroid_overlap[index_of_player] = new_position
							self.blue_old_centroid_overlap[contour_used[index_of_contour_found]] = new_position
							print "found someone who share blue contour with previous another one"
					else:
						print "found a contour but not what i want, blue player "+ str(index_of_player)
						print new_position, self.blue_position[index_of_player]
						contour_to_check = np.delete(contour_to_check,index_of_contour_found,0)
		self.blue_position = new_blue_position
		real_blue_position = np.add(np.array(self.blue_position),[BLUE_PLAYER_OFFSET,0])
		icon_frame_3 = self.drawPlayersPosition(icon_frame_2, self.blue_position,"Green" ) #printing blue on red player becos we cant see red on red
		icon_frame_3 = self.drawPlayersPosition(icon_frame_3, real_blue_position, "Red")
		# if j%self.SAVE_FREQUENCY == 0:
		# 	cv2.imwrite("PlayerPosition\Blue\Blue Players frame "+ str(j)+ ".jpg",icon_frame_3)
		# cv2.imwrite("Contours\Blue\Contoured Blue Players frame "+ str(j)+ ".jpg",contoured_frame_blue)
		# print "Time taken to get blue player position " + str(time.clock() - t0)
		#------------------------------------------------detect Green Players----------------------------------------------------------
		t0 = time.clock()
		both_teams_detected = (red_player_detected + median_blue_player).astype(np.uint8)

		inverse_both_teams_detected = (both_teams_detected*-1 + 255).astype(np.uint8) #cos -1 makes it int16
		three_chan_inver = np.dstack((inverse_both_teams_detected,inverse_both_teams_detected,inverse_both_teams_detected))
		green_players = cv2.bitwise_and(three_chan_inver, foregrd_coloured)
		green_channel = self.detectColour(green_players, "Green", self.THRESHOLD["Green"])
		contours, hierarchy = cv2.findContours(green_channel.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contoured_frame_green,_,contours_found_green = self.drawPlayerOutline(contoured_frame, contours, "Green")
		if j%self.SAVE_FREQUENCY == 0:
			cv2.imwrite("Contours\Green\Contoured Green Players frame " + str(j)+".jpg", contoured_frame_green)
		#-----------Blue team goalie-----------------------
		blue_golie_new_position, contour_found = self.closestCoordinate(self.blue_golie_postion, contours_found_green)
		if blue_golie_new_position == (-1,-1):
			blue_golie_new_position = self.blue_golie_postion
			print "missing blue goalie contour " 
		self.blue_golie_postion = blue_golie_new_position
		real_blue_golie_position = (self.blue_golie_postion[0] + BLUE_GOLIE_OFFSET, self.blue_golie_postion[1])
		icon_frame_4 = self.drawPlayersPosition(icon_frame_3, [real_blue_golie_position], "Red") #printing blue on red player becos we cant see red on red
		# if j%self.SAVE_FREQUENCY == 0:
		# 	cv2.imwrite("PlayerPosition\BlueGolie\Blue golie frame "+ str(j)+ ".jpg",icon_frame_4)

		#-----------Referee--------------------------
		referee_new_position, contour_found = self.closestCoordinate(self.referee_position, contours_found_green)
		if referee_new_position == (-1,-1):
			referee_new_position = self.referee_position
			print "missing referee contour " 
		self.referee_position = referee_new_position
		real_referee_position = (self.referee_position[0] + REFEREE_OFFSET, self.referee_position[1])
		icon_frame_5 = self.drawPlayersPosition(icon_frame_4, [real_referee_position], "Red") #printing blue on red player becos we cant see red on red
		if j%self.SAVE_FREQUENCY == 0:
			cv2.imwrite("PlayerPosition\Referee\Referee frame "+ str(j)+ ".jpg",icon_frame_5)
		# print "Time taken to get green player position " + str(time.clock() - t0)
		return real_red_position , self.red_goalie_position, real_blue_position, self.blue_golie_postion, self.referee_position

	def connectDots(self, frame):
		gaus_blur = cv2.GaussianBlur(frame.astype(np.uint8), (5,5), 0)
		_,thresholding = cv2.threshold(gaus_blur, 20, 255, cv2.THRESH_BINARY) # to connect more points of the player
		final_frame = cv2.medianBlur(thresholding, 7) 
		return final_frame


# DEBUG = 0
if DEBUG :
	startFrame = 200
	if startFrame == 0:
		#--------------------------------------frame 0 -------------------------------------------------------------------------------------------------
		red_position = [(348, 4683), (250, 4822), (294, 5088), (351, 5165), (538, 5234), (301, 5345), (256, 5418), (464, 6042)]#position for frame 0
		red_golie = (377,3660)
		blue_position = [(367,4753), (429,5125), (274,5246), (325,5359), (304,5433), (330,5486), (276,5490), (325,5641), (350,5755), (426,6012)]
		blue_goalie = (362,6256)
		referee_position = (340,5231)
		open('playersCoordinate.txt', 'w').close()
	else:
		fo = open('playersCoordinate200.txt', 'r')
		# fo.seek(0, 2)
		lines = fo.readlines()
		line = ast.literal_eval(lines[len(lines)-1])
		fo.close()
		red_position = np.subtract(line["red position "], [RED_PLAYER_OFFSET,0]).tolist()
		red_golie = (line["red golie"][0]-WHITE_OFFSET,line["red golie"][1])

		blue_position = np.subtract(line["blue position"], [BLUE_PLAYER_OFFSET ,0]).tolist()
		blue_goalie = (line["blue goalie"][0]-BLUE_GOLIE_OFFSET,line["red golie"][1])

		referee_position = (line["referee"][0]-REFEREE_OFFSET,line["referee"][1])
		
	detect_player = PlayerDectector(cv2.imread("backgrd.png"), red_position, red_golie, blue_position, blue_goalie, referee_position)
	
	for j in range(startFrame, 399):
		
		print "running frame " + str(j)
		frame = cv2.imread("C:\Users\weijian\Desktop\FullSize\panorama_frame_ " + str(j) +".jpg")
		red_position, red_goalie_position, blue_position, blue_golie_postion, referee_position = detect_player.detectPlayers(frame, j)
		# print "red position ", red_position
		# print "red golie", red_goalie_position
		# print "blue position", blue_position
		# print "blue goalie", blue_golie_postion
		# print "referee", referee_position

		coordinate = {"red position ":red_position.tolist(), "red golie":red_goalie_position, "blue position":blue_position.tolist(), "blue goalie":blue_golie_postion, "referee":referee_position}
		# print coordinate
		fo = open("playersCoordinate200.txt", "r+")
		fo.seek(0, 2)
		line = fo.write('\n' + str(coordinate))
		fo.close()