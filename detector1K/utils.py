import os
import tarfile
import shutil
import cv2
import urllib.request
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_gdrive_tar_model(gdrive_id, model_path):

    model_name = "model_float32"

    if not os.path.exists(model_path):
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path='./tmp/tmp.tar.gz')
        tar = tarfile.open("tmp/tmp.tar.gz", "r:gz")
        tar.extractall(path="tmp/")
        tar.close()

        shutil.move(f"tmp/saved_model/{model_name}.onnx", model_path)
        shutil.rmtree("tmp/")

def download_gdrive_file_model(gdrive_id, model_path):
    if not os.path.exists(model_path):
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                    dest_path=model_path)

def download_image(url):

    urllib.request.urlretrieve(url, "test.jpg")
    img = cv2.imread("test.jpg")
    return img

def download_video(url):

    urllib.request.urlretrieve(url, "test_video.mp4")