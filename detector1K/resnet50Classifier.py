import time
import pickle
import cv2
import os
import numpy as np
import onnx
import onnxruntime
from imread_from_url import imread_from_url

from .utils import download_gdrive_file_model

class Resnet50Classifier():

    def __init__(self, model_path, threshold = 0.6):

        self.threshold = threshold

        # Initialize classification model (download if necessary)
        model_gdrive_id = "1ZFRIXTCA1fS26MT_m9A1xIgtM5e8h_gT"
        download_gdrive_file_model(model_gdrive_id, model_path)
        self.initialize_model(model_path)

        # Load id to class dictionary
        self.load_label_dict()

    def __call__(self, image):

        return self.classify_object(image)

    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path)

        # Get model info
        self.get_model_input_details()
        self.get_model_output_details()

    def load_label_dict(self):

        with open(f"{os.path.dirname(__file__)}\\imagenet1000_clsid_to_human.pkl",'rb') as f:
            self.clsid_to_label_dict = pickle.load(f)

    def classify_object(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)

        cls_idx = np.argmax(output)  
        score = output[cls_idx]
        label = self.clsid_to_label_dict[cls_idx]

        if score < self.threshold:
            return None, None, None

        return cls_idx, label, score

    def prepare_input(self, img):

        self.img_height, self.img_width, self.img_channels = img.shape

        # Transform the image for inference
        img = cv2.resize(img,(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale input pixel values to -1 to 1
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        img = ((img/ 255.0 - mean) / std)

        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis,:,:,:].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):

        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        # print("Classification time: ", time.time() - start)

        return np.squeeze(outputs[0])

    def get_model_input_details(self):

        model_inputs = self.session.get_inputs()
        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_model_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        # print(self.output_names)

if __name__ == '__main__':

    model_path='../models/resnet50_vd_ssld_opt_fix.onnx'
    threshold = 0.6

    image = imread_from_url("https://wikimediafoundation.org/de/wp-content/uploads/sites/30/2018/07/640px-Space_Shuttle_Endeavor_in_Los_Angeles_-_2012_37919560104.jpg?w=640.jpg")

    classifier = Resnet50Classifier(model_path, threshold)

    label_id, label, score = classifier(image)
    print(label_id, label, score)
