import os
from src.components.data_preprocessing import DataPreprocessor
from src.components.face_embeddings_generater import Embedder
from src.constants import MODEL_NAME
from src.constants import EMBEDDINGS_GENERATOR_SCRIPT_NAME, EMBEDDINGS_GENERATOR_LOGFILE_NAME
from src.logger import get_logger

logger = get_logger(EMBEDDINGS_GENERATOR_SCRIPT_NAME, EMBEDDINGS_GENERATOR_LOGFILE_NAME)

def embedder_pipeline():
    try:
        logger.info("STARTED THE PIPELINE...")
        # Preprocessor = DataPreprocessor()
        logger.info("PREPROCESSOR INSTANTIATED SUCCESFULLY...STARTED PREPROCESSING....")
        # Preprocessor.preprocessor()
        logger.info("PREPROCESSING COMPLLETED...STARTING EMBEDDING GENERATION....")
        Emb1 = Embedder(model_name = MODEL_NAME)
        logger.info("EMBEDDER SUCCESSFULLY GENERATED")
        Emb1.save_embeddings()
        logger.info("EMBEDDINGS GENERATED AND SAVED SUCCESSFULLY")
                
    except Exception as e:
        logger.error("Unexpected error occurred in embedder pipeline: %s", e)
        raise 
    
    
if __name__ == "__main__":
    embedder_pipeline()