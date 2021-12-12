import time
import cv2
import numpy as np

from detector1K.utils import download_image, download_video
from detector1K import Detector1K

if __name__ == '__main__':
	
	draw_scores = False

	detection_model_path = 'models/object_localizer_float32.onnx'
	detection_threshold = 0.22

	classification_model_path = 'models/resnet50_vd_ssld_float32.onnx'
	classification_threshold = 0.5

	# Initialize test video, Sample from: https://www.pexels.com/video/a-medusa-jellyfish-swimming-gracefully-underwater-2731905/
	video_url = "https://vod-progressive.akamaized.net/exp=1639333714~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F184%2F14%2F350923088%2F1421472259.mp4~hmac=e2aa39c7fbf2fe1ccaead03cf77dc2b729c98586e155b06c3d9c63c8a14a1634/vimeo-prod-skyfire-std-us/01/184/14/350923088/1421472259.mp4?download=1&filename=Pexels+Videos+2731905.mp4"
	download_video(video_url)
	cap = cv2.VideoCapture("test_video.mp4")

	detector = Detector1K(detection_model_path, classification_model_path, detection_threshold, classification_threshold)

	cv2.namedWindow("Detected objects", cv2.WINDOW_NORMAL)	

	# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (3840,2160))

	while cap.isOpened():

		# Read frame from the video
		ret, frame = cap.read()

		if ret:	

			detections = detector(frame)
			detection_img = detector.draw_detections(frame, draw_scores)

			cv2.imshow("Detected objects", detection_img)
			# out.write(detection_img)

		else:
			break

		# Press key q to stop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	# out.release()
