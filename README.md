## CS4243 Computer Vision and Pattern Recognition Project
This is a project that invovles stitching 3 videos of a soccer match into a panorama view, and transform the view into a top down view with the players' position being shown.

###Requirements: 
1. 3 videos of the soccer match named the following: 
  - football_left.mp4
  - football_mid.mp4
  - football_right.mp4
  
  
###Procedure:
1. Run ````Generate_Panorama.py```` to generate the stitched up video, and a folder named "FullSize" which contains the full resolution of the stitched up frames for further processing.
**Do take note that the FullSize folder is more than 14GB**

2. Next, using the frames in FullSize folder, we aim to track the players and the referee.
  1. Generate a background image by running ````backgroundMaker.py````, which takes in the frames from FullSize folder, and runs an averaging algorithm to get the background of the scene.
  2. Run ````Player_detection_final```` to generate a text document of the players' position

3. Finally, call upon ````topDown.py```` to produce the top down video, and the panorama video with the offside line. 
