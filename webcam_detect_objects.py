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

	# Initialize the webcam
	cap = cv2.VideoCapture(0)

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
		if cv2.waitKey(1) & 0xFF  == ord('q'):
			break

	cap.release()
	# out.release()
