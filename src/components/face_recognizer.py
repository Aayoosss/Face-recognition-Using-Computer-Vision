import os
import numpy as np
from mtcnn import MTCNN
import cv2
from src.constants import EMBEDDINGS_PATH, FACE_RECOGNIZER_LOGFILE_NAME, FACE_RECOGNIZER_SCRIPT_NAME, MODEL_NAME
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import get_logger
from src.components.face_embeddings_generater import Embedder
logger = get_logger(FACE_RECOGNIZER_SCRIPT_NAME, FACE_RECOGNIZER_LOGFILE_NAME)

class Recognizer:
    """
    A class for encapsulating all the loading & extraction functions required for recognizing the person.
    """
    def __init__(self):
        try:
            self.embedding_path = EMBEDDINGS_PATH
            if not os.path.exists(self.embedding_path):
                raise FileNotFoundError(f"Embedding file not found at {self.embedding_path}")
        except FileNotFoundError as e:
            logger.error(f"File does not exist at: {self.embedding_path}\n{e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred while initialising Recogniser: {e}")
            raise
            
    def load_embeddings_file(self):
        try:
            self.data = np.load(self.embedding_path, allow_pickle=True)
            logger.info("Embeddings file found and loaded successfully.")
            return self.data
        except Exception as e:
            logger.error("Unexpected error occurred while loading the embeddings.npz file: %s", e)
            raise     
    
    def load_image(self, image_path: str = None, img: np.ndarray = None):
        try:
            if image_path is not None:
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            detector = MTCNN()
            try:
                detections = detector.detect_faces(img_rgb)
            except MemoryError:
                logger.warning(f"Skipping {image_path} due to memory error")
                
            if len(detections) == 0:
                logger.info("Face not found in the imag4e")
                return None, None
            else:
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
                    logger.info("Face not detected after alignment")
                    return None, None
                
                else:
                    x, y, width, height = aligned_faces[0]['box']
                    x = max(0, x)
                    y = max(0, y)
                    face = aligned[y:y+height, x:x+width]
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    face = cv2.resize(face, (160, 160))
                    
                    embedder = Embedder(model_name = MODEL_NAME)

                    embeddings = embedder.generate_embeddings(face)
                    return img, embeddings
        except Exception as e:
            logger.error("Unexpected error occurred while generating embeddings: %e", e)
            raise
           
    def get_names_embeddings(self):
        try:
            data = self.load_embeddings_file()
            if data:
                names = data['names']
                embeddings = data['mean_embeddings']
                return names, embeddings
        except Exception as e:
            logger.error("Unexpected error occured while loading extracting names and embeddings: %s", e)
            raise

            
