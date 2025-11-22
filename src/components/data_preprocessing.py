import os
import pandas
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import numpy as np
from src.constants import RAW_DATA_IMAGE_PATH, PREPROCESSED_DATA_PATH, DATA_PREPROCESSING_SCRIPT_NAME, DATA_PREPROCESSING_LOGFILE_NAME
from src.logger import get_logger


logger = get_logger(DATA_PREPROCESSING_SCRIPT_NAME, DATA_PREPROCESSING_LOGFILE_NAME)


class DataPreprocessor:
    """
    Preprocesses face images by detecting, aligning, and saving cropped faces
    for each person in the specified folders.
    """
    def __init__(self):
        self.raw_data_path = RAW_DATA_IMAGE_PATH
        self.preprocessed_data_path = PREPROCESSED_DATA_PATH

    def preprocessor(self):
        try:
            logger.info("Preprocessing started")
            folder_names = os.listdir(self.raw_data_path)
            for folder in folder_names:
                folder_path = os.path.join(self.raw_data_path, folder)
                destination_path = os.path.join(self.preprocessed_data_path, folder)
                os.makedirs(destination_path, exist_ok=True)
                file_names = os.listdir(folder_path)
                for file in file_names:
                    img = cv2.imread(os.path.join(folder_path, file))
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    detector = MTCNN()
                    try:
                        detections = detector.detect_faces(img_rgb)
                    except MemoryError:
                        logger.warning(f"Skipping {file} due to memory error")
                        continue
                    if len(detections) == 0:
                        continue
                    keypoints = detections[0]['keypoints']
                    left_eye = keypoints['left_eye']
                    right_eye = keypoints['right_eye']
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
                    M = cv2.getRotationMatrix2D(midpoint, angle, 1.0)
                    aligned = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]))
                    aligned_faces = detector.detect_faces(aligned)
                    if len(aligned_faces) == 0:
                        continue
                    x, y, width, height = aligned_faces[0]['box']
                    x = max(0, x)
                    y = max(0, y)
                    face = aligned[y:y+height, x:x+width]
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    face = cv2.resize(face, (160, 160))
                    cv2.imwrite(f"{destination_path}/{file}", face)
                logger.info(f" Completed Processing {folder}")
            logger.info("Preprocessing completed!")
        except Exception as e:
            logger.error("Unexpected error occurred while preprocessing: %s", e)
            raise
        
        
# if __name__ == "__main__":
#     folder_names = os.listdir(RAW_DATA_IMAGE_PATH)
#     preprocessor(folder_names,RAW_DATA_IMAGE_PATH)