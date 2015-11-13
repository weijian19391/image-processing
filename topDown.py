import numpy as np
import argparse
import cv2
import cv2.cv as cv
import ast

def convert3Dto2D(coord3d, M, tupSize):
  if (tupSize == 0):
    (numplayers, coord) = coord3d.shape
    coord2d = np.ones((numplayers, 3))
    finalPlayer2D = np.ones((numplayers,2))
    i = 0
    
    while i in range (numplayers):
      coord2d[i] = np.array([coord3d[i,1], coord3d[i,0],1])
      aftHomog = np.dot(M, coord2d[i].transpose())
      coord2d[i] = np.array([aftHomog[0]/aftHomog[2], aftHomog[1]/aftHomog[2], aftHomog[2]])
      finalPlayer2D[i] = np.array([coord2d[i,0], coord2d[i,1]])
      i = i+1
  else:
    numplayers = 1;
    coord2d = np.array([coord3d[1], coord3d[0],1])
    aftHomog = np.dot(M, coord2d.transpose())
    coord2d = np.array([aftHomog[0]/aftHomog[2], aftHomog[1]/aftHomog[2], aftHomog[2]])
    finalPlayer2D = np.array([coord2d[0], coord2d[1]])
      
  return finalPlayer2D
  
def drawPlayer(field, teamRed, teamBlue, keeperRed, keeperBlue, referee):

  numRed,coord2d = teamRed.shape
  numBlue,coord2d = teamBlue.shape
  
  i=0
  while i in range (numRed):
    cv2.circle(field,(teamRed[i,0].astype(int), teamRed[i,1].astype(int)),10,(0,0,255),-1) #red
    i = i+1
  
  j=0
  while j in range(numBlue):
    cv2.circle(field,(teamBlue[j,0].astype(int), teamBlue[j,1].astype(int)),10,(255,0,0),-1)#blue
    j = j+1
  
  #keeperRed
  cv2.circle(field,(keeperRed[0].astype(int), keeperRed[1].astype(int)),10,(0,0,0),-1)
  #keeperBlue
  cv2.circle(field,(keeperBlue[0].astype(int), keeperBlue[1].astype(int)),10,(255,255,255),-1)
  #referee
  cv2.circle(field,(referee[0].astype(int), referee[1].astype(int)),10,(0,255,0),-1)
  
  return field
  #cv2.imwrite("field.jpg",field)
  
def four_point_transform(image, pts):
  
  #get a consistent order of the points and unpack them individually
  #rect = order_points(pts)
  (tl, tr, br, bl) = pts
  
  dst = np.array([
    [0, 0],
    [481*3,0],
    [481*3,315*3],
    [0,315*3]], dtype = "float32")
 
  # compute the perspective transform matrix and then apply it
  
  M , v= cv2.findHomography(pts, dst)
  warped = cv2.warpPerspective(image, M, (490*3,317*3))
 
  return warped,M

def topDown(team1, keeper1, team2, keeper2, referee):

  #football field
  image = cv2.imread("backgrd.png")
  pts = np.array([(3399,240),(5710,228),(9080,1019),(145,1120)],dtype = "float32" )
  #keeper1 = (377,3660)
  #keeper2 = (362,6256)
  warped,M = four_point_transform(image, pts)

  #team1 = np.array([(373,4682), (271,4822), (309,5085), (377,5166), (578,5244), (316,5337), (275,5415), (490,6028)])
  red2d = convert3Dto2D(team1,M,0)
  
  keeper1 = np.array(keeper1)
  keeperRed = convert3Dto2D(keeper1,M,1)
  
  #team2 = np.array([(385,4748),  (460,5127), (287,5243), (346,5357), (322,5430), (344,5478), (291,5489), (347,5637), (369,5750), (459,6011)])
  blue2d = convert3Dto2D(team2,M,0)
  
  keeper2 = np.array(keeper2)
  keeperBlue = convert3Dto2D(keeper2,M,1)
  
  referee = np.array(referee)
  ref = convert3Dto2D(referee,M,1)
  
  return drawPlayer (warped, red2d, blue2d, keeperRed, keeperBlue, ref )
  
fo =open("playersCoordinate - Copy.txt", "r")
lines = fo.readlines()
lines.remove('\n')
video_output_size = (1470,951)
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
fps = 23
video_writer = cv2.VideoWriter('Field\\top_down.mov', fourcc, fps, video_output_size)

for i in range(len(lines)):
  line = ast.literal_eval(lines[i])
  print i
  field = topDown(np.array(line["red position "]) , line["red golie"], np.array(line["blue position"]), line["blue goalie"], line["referee"])
  video_writer.write(field)
  cv2.imwrite("Field\\field " + str(i)+ ".jpg",field)
video_writer.release()
  
