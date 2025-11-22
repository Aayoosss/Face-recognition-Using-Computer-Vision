from src.components import Recognizer
from src.constants import RECOGNISE_PIPELINE_LOGFILE_NAME, RECOGNISE_PIPELINE_SCRIPT_NAME, SAMPLE_TEXT_IMAGE_PATH
from src.logger import get_logger
from sklearn.metrics.pairwise import cosine_similarity

Logger = get_logger(RECOGNISE_PIPELINE_SCRIPT_NAME, RECOGNISE_PIPELINE_LOGFILE_NAME)

def recognise(img_path: str):
    try:
        recogniser = Recognizer()
        img, new_embedding = recogniser.load_image(img_path)
        Logger.info("Image loaded successfully")
        names, embeddings = recogniser.get_names_embeddings()
        max_similarity = -9999999999
        person = None
        # print(f"Total names = {names.shape}")
        # print(f"Total embeddings = {embeddings.shape}")
        for name, embedding in zip(names,embeddings):
            similar = cosine_similarity(new_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]

            if max_similarity < similar:
                max_similarity = similar
                person = name
          
        if person and max_similarity > 0.45:
            Logger.info("Person found successfully")
            print(f"Person identified: {person}")
            print(f"Similarity calculated: {max_similarity}")
            
        else:
            print("Could not recognise the person in the image. Check whether the person is registered on not. If not please register first.")
    
    except Exception as e:
        Logger.error("Unexpected error occurred while recognising the person: %s", e)
        raise e
    
if __name__ == "__main__":
    recognise(SAMPLE_TEXT_IMAGE_PATH)
            
        