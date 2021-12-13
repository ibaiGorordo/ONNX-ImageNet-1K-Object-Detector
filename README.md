# ONNX-ImageNet-1K-Object-Detector
Python scripts for performing object detection with the 1000 labels of the ImageNet dataset in ONNX. The repository combines a class agnostic object localizer to first detect the objects in the image, and next a ResNet50 model trained on ImageNet is used to label each box.

![Imagenet 1K Object Detection](https://github.com/ibaiGorordo/ONNX-ImageNet-1K-Object-Detector/blob/main/doc/img/output_balloon.jpg)
*Original image: https://commons.wikimedia.org/wiki/File:Il_cuore_di_Como.jpg*

# Why
There are a lot of object detection models, but since most of them are trained in the COCO dataset, most of them can only detect a maximum of 80 classes. This repository proposes a "quick and dirty" solution to be able to detect the 1000 objects available in the ImageNet dataset.

# :exclamation:Important:exclamation:
- This model uses a lightweight class agnostic object localizer to first detect the objects. Therefore, this repository is not going to behave as well as other object detection models in complex scenes. In those cases, the object localizer will fail quickly and therefore no objects will be detected.
- The ResNet50 clasifier is fast in a desktop GPU, however, since it needs to run for each of the detected boxes, the performance might be affected for images with many objects.

# Requirements

 * Check the requirements.txt file.
 
# Installation
```
pip install -r requirements.txt
```

# ONNX model

- **Class Agnostic Object Localizer**:
The original model from TensorflowHub (link at the bottom) was converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), the models can be found in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer). This repository will automatically download the model if the model is not found in the models folder.

- **ResNet50 Classifier**:
The original model from PaddleClas (link at the bottom) was converted to ONNX format using a similar procedure as the one described in [this article](https://zenn.dev/pinto0309/scraps/cf319db8fea4c3) by [PINTO0309](https://github.com/PINTO0309). This repository will automatically download the model.

# How to use

 * **Image inference**:
 
 ```
 python image_object_detection.py
 ```
 
  * **Video inference**:
 
 ```
 python video_object_detection.py
 ```
 
  * **Webcam inference**:
 
 ```
 python webcam_object_detection.py
 ```

 # Examples

## Macaque Detection
![Macaque Detection](https://github.com/ibaiGorordo/ONNX-ImageNet-1K-Object-Detector/blob/main/doc/img/macaque_output.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Onsen_Monkey.JPG*

## Christmas Stocking Detection
![Christmas Stocking Detection](https://github.com/ibaiGorordo/ONNX-ImageNet-1K-Object-Detector/blob/main/doc/img/stocking_output.jpg)
 *Original image: https://unsplash.com/photos/paSqTlm3DsA*

## Burrito Detection
![Burrito Detection](https://github.com/ibaiGorordo/ONNX-ImageNet-1K-Object-Detector/blob/main/doc/img/burrito_output.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Breakfast_burrito_(cropped).jpg*

## Bridge Detection
![Bridge Detection](https://github.com/ibaiGorordo/ONNX-ImageNet-1K-Object-Detector/blob/main/doc/img/bridge_output.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Bayonne_Bridge_Collins_Pk_jeh-2.JPG*

 # [Inference video Example]
https://user-images.githubusercontent.com/43162939/145718532-c674bb42-a8f6-4744-8c78-cc093c6ad64e.mp4

 Original video: https://www.pexels.com/video/a-medusa-jellyfish-swimming-gracefully-underwater-2731905/ (by 
Vova Krasilnikov)

# References
- **Original Class Agnostic Object Localizer**: https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1
- **Original Resnet50 (ResNet50_vd_ssld) Classifier from PaddleClass**: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md
- **PINTO0309's model zoo**: https://github.com/PINTO0309/PINTO_model_zoo
- **PINTO0309's model conversion tool**: https://github.com/PINTO0309/openvino2tensorflow
- **PaddlePaddle to ONNX conversion article**: https://zenn.dev/pinto0309/scraps/cf319db8fea4c3
