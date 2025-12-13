from src.components import Recognizer
from src.constants import RECOGNISE_PIPELINE_LOGFILE_NAME, RECOGNISE_PIPELINE_SCRIPT_NAME, SAMPLE_TEXT_IMAGE_PATH
from src.logger import get_logger
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

Logger = get_logger(RECOGNISE_PIPELINE_SCRIPT_NAME, RECOGNISE_PIPELINE_LOGFILE_NAME)

def recognise(img_path: str = None, img_array: np.ndarray = None):
    try:
        recogniser = Recognizer()
        if img_path is not None:
            img, new_embedding = recogniser.load_image(image_path = img_path)
        elif img_array is not None:
            img, new_embedding = recogniser.load_image(img = img_array)
        Logger.info("Image loaded successfully")
        names, embeddings = recogniser.get_names_embeddings()
        max_similarity = -9999999999
        person = None
        for name, embedding in zip(names,embeddings):
            similar = cosine_similarity(new_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]

            if max_similarity < similar:
                max_similarity = similar
                person = name
          
        if person and max_similarity > 0.10:
            Logger.info("Person found successfully")
            print(f"Person identified: {person}")
            print(f"Similarity calculated: {max_similarity}")
            return person, max_similarity
            
        else:
            Logger.info("Could not recognise the person in the image. Check whether the person is registered on not. If not please register first.")
            return("Unknown",max_similarity)
    
    except Exception as e:
        Logger.error("Unexpected error occurred while recognising the person: %s", e)
        raise e
    
if __name__ == "__main__":
    img = cv2.imread(SAMPLE_TEXT_IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    person, max_similarity = recognise(img_path = SAMPLE_TEXT_IMAGE_PATH)
            
        