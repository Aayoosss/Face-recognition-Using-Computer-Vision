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
FACE_EMBEDDINGS_SCRIPT_NAME = "face_embeddings_component.py"
FACE_EMBEDDINGS_LOGFILE_NAME = "face_embeddings_component.log"

#Paths and constants for pipeline - face_recognizer.py component
MODEL_NAME = "Facenet512" 
EMBEDDINGS_PATH = "Embeddings/Embeddings.npz"
FACE_RECOGNIZER_SCRIPT_NAME = "face_recognizer_component.py"
FACE_RECOGNIZER_LOGFILE_NAME = "face_recognizer_component.log"



#Paths and constants for pipeline - generate_embeddings.py pipeline
EMBEDDINGS_GENERATOR_SCRIPT_NAME = "Embeddings_generator_pipeline.py"
EMBEDDINGS_GENERATOR_LOGFILE_NAME = "Embeddings_generator_pipeline.log"

#Paths and constants for pipeline - recognise.py pipeline
SAMPLE_TEXT_IMAGE_NAME = "Akshay Kumar_1.jpeg"
SAMPLE_TEXT_IMAGE_PATH = os.path.join("Faces/test_images",SAMPLE_TEXT_IMAGE_NAME)
RECOGNISE_PIPELINE_SCRIPT_NAME = "recognise_pipeline.py"
RECOGNISE_PIPELINE_LOGFILE_NAME = "recognise_pipeline.log"
