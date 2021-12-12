import cv2
import imread_from_url

from detector1K.utils import download_image
from detector1K import Detector1K

if __name__ == '__main__':

	draw_scores = True

	detection_model_path = 'models/object_localizer_float32.onnx'
	detection_threshold = 0.3

	classification_model_path = 'models/resnet50_vd_ssld_float32.onnx'
	classification_threshold = 0.5

	# image = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Il_cuore_di_Como.jpg/800px-Il_cuore_di_Como.jpg")
	image = download_image("https://upload.wikimedia.org/wikipedia/commons/5/55/Il_cuore_di_Como.jpg")
	detector = Detector1K(detection_model_path, classification_model_path, detection_threshold, classification_threshold)

	# Detect objects
	detections = detector(image)
	detection_img = detector.draw_detections(image, draw_scores)
	
	cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
	cv2.imwrite("output.jpg", detection_img)
	cv2.imshow("Detections", detection_img)
	cv2.waitKey(0)