import time
import cv2
import numpy as np
import onnx
import onnxruntime
from imread_from_url import imread_from_url

from .utils import download_gdrive_tar_model, download_image
from .resnet50Classifier import Resnet50Classifier

np.random.seed(2021)
colors = np.random.randint(255, size=(1000, 3), dtype=int)

class Detector1K():

	def __init__(self, detector_model_path, classifier_model_path, detector_threshold = 0.2, classifier_threshold = 0.6):

		self.threshold = detector_threshold

		# Initialize detection model (download if necessary)
		models_gdrive_id = "1mVxOy65EsLhNgtqGrydOrhMkZpaktW0w"
		download_gdrive_tar_model(models_gdrive_id, detector_model_path)
		self.initialize_model(detector_model_path)

		# Initialize classification model (download if necessary)
		self.classifier = Resnet50Classifier(classifier_model_path, classifier_threshold)

		# Set the crop offset
		self.crop_offset = 0

	def __call__(self, image):

		return self.detect_objects(image)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path)

		# Get model info
		self.get_model_input_details()
		self.get_model_output_details()

	def detect_objects(self, image):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		detections = self.process_output(outputs)

		# Classify detections
		self.label_detections = self.classify_objects(detections, image)

		return self.label_detections

	def prepare_input(self, img):

		self.img_height, self.img_width, self.img_channels = img.shape

		# Transform the image for inference
		img = cv2.resize(img,(self.input_width, self.input_height))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img = img.transpose(2, 0, 1)
		input_tensor = img[np.newaxis,:,:,:].astype(np.float32)

		return input_tensor

	def inference(self, input_tensor):

		start = time.time()
		outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
		# print("Detection time: ", time.time() - start)

		return outputs

	def process_output(self, outputs):  

		# Get all output details
		boxes = np.squeeze(outputs[0])
		classes = np.squeeze(outputs[1])
		scores = np.squeeze(outputs[2])
		num_objects = int(outputs[3][0])

		results = []
		for i in range(num_objects):
			if scores[i] >= self.threshold:

				y1 = (self.img_height * boxes[i][0]).astype(int)
				y2 = (self.img_height * boxes[i][2]).astype(int)
				x1 = (self.img_width * boxes[i][1]).astype(int)
				x2 = (self.img_width * boxes[i][3]).astype(int)

				result = {
				  'bounding_box': np.array([x1, y1, x2, y2],dtype=int),
				  'class_id': 0,
				  'label': "",
				  'detection_score': scores[i],
				  'classification_score': 0,
				}
				results.append(result)
		return results

	def classify_objects(self, detections, img):

		label_detections = detections.copy()

		for det_idx, detection in enumerate(detections):

			# Crop image using the bounding boc and pass it to the classifier
			box = detection['bounding_box']
			crop_left = max(box[0]-self.crop_offset,0)
			crop_top = max(box[1]-self.crop_offset,0)
			crop_right = min(box[2]+self.crop_offset,self.img_width)
			crop_bottom = min(box[3]-self.crop_offset,self.img_height)
			crop_img = img[crop_top:crop_bottom,crop_left:crop_right]

			# Classify the object and update the detection data
			label_id, label, score = self.classifier(crop_img)

			if label_id:
				label_detections[det_idx]['class_id'] = label_id
				label_detections[det_idx]['label'] = label
				label_detections[det_idx]['classification_score'] = score

		return label_detections

	def get_model_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_name = self.session.get_inputs()[0].name

		self.input_shape = self.session.get_inputs()[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_model_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

	def draw_detections(self, image, draw_scores):

		for idx, detection in enumerate(self.label_detections):

			box = detection['bounding_box']
			class_id = detection['class_id']
			label = detection['label']
			class_score = int(100*detection['classification_score'])
			det_score = int(100*detection['detection_score'])

			color = (int(colors[class_id,0]), int(colors[class_id,1]), int(colors[class_id,2]))
			size = min([self.img_height, self.img_width])*0.002
			text_thickness = int(min([self.img_height, self.img_width])*0.004)
			textSize = cv2.getTextSize(text=f'class {class_score}% - det {det_score}%', 
				fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=text_thickness)[0][1]*1.6
			
			cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, text_thickness)

			if not detection['class_id']:
				continue
			cv2.putText(image, f'{label}', (box[0], box[1] + int(textSize*0.8)), 
						cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)
			if not draw_scores:
				continue
			cv2.putText(image, f'class: {class_score}%', (box[0], box[1] + int(textSize*1.8)), 
						cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)
			cv2.putText(image, f'det: {det_score}%', (box[0], box[1] + int(textSize*2.8)), 
						cv2.FONT_HERSHEY_SIMPLEX, size, color, text_thickness, cv2.LINE_AA)

		return image

if __name__ == '__main__':

	draw_scores = False

	detection_model_path = '../models/object_localizer_float32.onnx'
	detection_threshold = 0.3

	classification_model_path = '../models/resnet50_vd_ssld_float32.onnx'
	classification_threshold = 0.5

	# image = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Il_cuore_di_Como.jpg/800px-Il_cuore_di_Como.jpg")
	image = download_image("https://upload.wikimedia.org/wikipedia/commons/5/55/Il_cuore_di_Como.jpg")
	detector = Detector1K(detection_model_path, classification_model_path, detection_threshold, classification_threshold)

	start_time = time.time()
	detections = detector(image)
	detection_img = detector.draw_detections(image, draw_scores)
	# print("total: ",time.time()-start_time)
	
	cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
	cv2.imwrite("../output.jpg", detection_img)
	cv2.imshow("Detections", detection_img)
	cv2.waitKey(0)