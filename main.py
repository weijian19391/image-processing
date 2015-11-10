import numpy as np
import cv2
import cv2.cv as cv
from player_detection import PlayerDectector


def calculate_size(img1, img2, homography_matrix):
	height1, width1 = img1[:2]
	height2, width2 = img2[:2]

	corner_image1_pts = np.float32([[0, 0],[0, height1],[width1, height1],[width1, 0]]).reshape(-1, 1, 2)
	corner_image2_pts = np.float32([[0, 0],[0, height2],[width2, height2],[width2, 0]]).reshape(-1, 1, 2)
	projected_corner_image2_pts = cv2.perspectiveTransform(corner_image2_pts, homography_matrix)

	#xmin = np.minimum(np.min(corner_image1_pts[:, :, 0]),np.min(projected_corner_image2_pts[:, :, 0]))
	xmax = np.maximum(np.max(corner_image1_pts[:, :, 0]),np.max(projected_corner_image2_pts[:, :, 0]))
	#ymin = np.minimum(np.min(corner_image1_pts[:, :, 1]),np.min(projected_corner_image2_pts[:, :, 1]))
	ymax = np.maximum(np.max(corner_image1_pts[:, :, 1]),np.max(projected_corner_image2_pts[:, :, 1]))

	return xmax, ymax


def stitch_image1(img1, img2, homography_matrix_file):
	homography_matrix = np.loadtxt(homography_matrix_file)
	size = calculate_size(img1.shape, img2.shape, homography_matrix)
	warped_img2 = cv2.warpPerspective(img2, homography_matrix, size)

	h, w = img1.shape[:2]
	warped_img2[:h, :w] = img1

	return warped_img2


def stitch_image2(img1, img2, homography_matrix_file):
	homography_matrix = np.loadtxt(homography_matrix_file)
	offset_x = 4331
	translation_matrix = np.float32([[1, 0, offset_x],[0, 1, 0]])
	cropped_img1 = img1[:, 3:]
	h, w = np.int32(cropped_img1.shape[:2])
	offset_w = w + offset_x
	offset_img1 = cv2.warpAffine(cropped_img1, translation_matrix, (offset_w, h))
	size = calculate_size(offset_img1.shape, img2.shape, homography_matrix)
	warped_img2 = cv2.warpPerspective(img2, homography_matrix, size)
	warped_img2[:, offset_x:w+offset_x] = cropped_img1

	return warped_img2


def generate_panorama(left_view, mid_view, right_view):
	backgrd = cv2.imread("backgrd.png")
	detPlayer = PlayerDectector(backgrd)

	video_capture_left = cv2.VideoCapture(left_view)
	video_capture_mid = cv2.VideoCapture(mid_view)
	video_capture_right = cv2.VideoCapture(right_view)

	fps = video_capture_left.get(cv.CV_CAP_PROP_FPS)
	frame_count = int(video_capture_left.get(cv.CV_CAP_PROP_FRAME_COUNT))
	fourcc = video_capture_left.get(cv.CV_CAP_PROP_FOURCC)
	print fps
	print fourcc
	"""
	_, img_left = video_capture_left.read()
	_, img_mid = video_capture_mid.read()
	_, img_right = video_capture_right.read()

	stitched_img_mid_and_img_right = stitch_image1(img_mid, img_right, "h2_matrix.txt")
	cv2.imwrite('stitched_img_mid_and_img_right.jpg',stitched_img_mid_and_img_right)
	panorama = stitch_image2(stitched_img_mid_and_img_right, img_left, "h1_matrix.txt")
	cropped_panorama = panorama[:1200,800:]
	cv2.imwrite('panorama.jpg',cropped_panorama)

	"""
	# video_output_size = (9400, 1200)
	video_output_size = (5672,724)
	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	video_writer = cv2.VideoWriter('football_panorama.mov', fourcc, fps, video_output_size)
	#for frame in range(np.int32(frame_count)):

	for frame in range(10):
		print "Progress... %0.2f%%, current frame is %d" % (np.around((frame/float(frame_count) * 100), decimals=2), frame)
		_, img_left = video_capture_left.read()
		_, img_mid = video_capture_mid.read()
		_, img_right = video_capture_right.read()
		stitched_img_mid_and_img_right = stitch_image1(img_mid, img_right, "h2_matrix.txt")
		panorama = stitch_image2(stitched_img_mid_and_img_right, img_left, "h1_matrix.txt")
		cropped_panorama = panorama[:1200, 800:10200]
		detPlayer.detectPlayers(cropped_panorama,frame)
		# resize = cv2.resize(cropped_panorama, video_output_size, interpolation=cv2.INTER_AREA)
		cv2.imwrite('C:\Users\weijian\Desktop\FullSize\panorama_frame_ ' + str(frame) +'.jpg',cropped_panorama)
		# video_writer.write(resize)



	video_capture_left.release()
	video_capture_mid.release()
	video_capture_right.release()
	video_writer.release()

	print "video output completed"


def main():
	video_left_name = 'football_left.mp4'
	video_mid_name = 'football_mid.mp4'
	video_right_name = 'football_right.mp4'

	generate_panorama('football_left.mp4', 'football_mid.mp4', 'football_right.mp4')
if __name__ == "__main__":
	main()