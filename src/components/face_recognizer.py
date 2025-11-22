import os
import numpy as np
import cv2
from src.constants import FACE_EMBEDDINGS_PATH, FACE_RECOGNIZER_LOGFILE_NAME, FACE_RECOGNIZER_SCRIPT_NAME, MODEL_NAME
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
            self.embedding_path = FACE_EMBEDDINGS_PATH
            if not os.path.exists(self.embedding_path):
                raise FileNotFoundError(f"Embedding file not found at {self.embedding_path}")
        except FileNotFoundError as e:
            logger.error(f"File does not exist at: {self.embedding_path}\n{e}")
            raise  # optional: re-raise if you want to stop initialization
        except Exception as e:
            logger.error(f"Unexpected error occurred while initialising Recogniser: {e}")
            raise
            
    def load_embeddings_file(self):
        try:
            data = np.load(self.embedding_path, allow_pickle=True)
            logger.info("Embeddings file found and loaded successfully.")
            return data
        except Exception as e:
            logger.error("Unexpected error occurred while loading the embeddings.npz file: %s", e)
            raise     
    
    def load_image(self, image_path: str):
        try:
            img = cv2.imread(image_path)
            embedder = Embedder(model_name = MODEL_NAME)
            embeddings = embedder.generate_embeddings(img)
            return img, embeddings
        except Exception as e:
            logger.error("Unexpected error occurred while generating embeddings: %e", e)
            raise
            
            
    
    def get_names_embeddings(self):
        try:
            data = self.load_embeddings_file()
            if self.data:
                names = data['names']
                embeddings = data['embeddings']
                return names, embeddings
        except Exception as e:
            logger.error("Unexpected error occured while loading extracting names and embeddings: %s", e)
            raise

            
