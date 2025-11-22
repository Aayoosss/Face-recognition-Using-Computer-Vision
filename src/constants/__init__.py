import os

#Paths and constants for data_preprocessing.py component
RAW_DATA_IMAGE_PATH = "Original Images/Original Images"
FOLDER_NAMES = os.listdir(RAW_DATA_IMAGE_PATH)
PREPROCESSED_DATA_PATH = "Faces/my_faces"
DATA_PREPROCESSING_SCRIPT_NAME = "data_preprocessing.py"
DATA_PREPROCESSING_LOGFILE_NAME = "data_preprocessing.log"

#Paths and constants for face_embeddings_generter.py component
MODEL_NAME = "Facenet512"
PREPROCESSED_DATA_PATH = "Faces/my_faces"
EMBEDDINGS_PATH = "Embeddings/Embeddings.npz"
EMBEDDINGS_CSV_PATH = os.path.join(EMBEDDINGS_PATH, "embeddings_index.csv")
FACE_EMBEDDINGS_SCRIPT_NAME = "face_embeddings_generater.py"
FACE_EMBEDDINGS_LOGFILE_NAME = "face_embeddings_generater.log"

#Paths and constants for pipeline - face_recogniser.py component
EMBEDDINGS_PATH = "Embeddings/Embeddings.npz"



#Paths and constants for pipeline - Embeddings_generator.py pipeline
EMBEDDINGS_GENERATOR_SCRIPT_NAME = "Embeddings_generator.py"
EMBEDDINGS_GENERATOR_LOGFILE_NAME = "Embeddings_generator.log"