import os
from keras.models import load_model
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace
import cv2
from src.constants import PREPROCESSED_DATA_PATH, EMBEDDINGS_PATH, FACE_EMBEDDINGS_LOGFILE_NAME, FACE_EMBEDDINGS_SCRIPT_NAME
from src.logger import get_logger

logger = get_logger(FACE_EMBEDDINGS_SCRIPT_NAME, FACE_EMBEDDINGS_LOGFILE_NAME)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.destination_path = EMBEDDINGS_PATH
        self.source_path = PREPROCESSED_DATA_PATH

    def get_embedding(self, face_pixels):
        """
        Use DeepFace.represent directly on the face pixels.
        enforce_detection=False because we are passing already detected and cropped faces.
        """
        try:
            embedding_result = DeepFace.represent(
                img_path=face_pixels,
                model_name=self.model_name,
                enforce_detection=False
            )
            if embedding_result:
                embedding = embedding_result[0]["embedding"]
                embedding_vector = np.array(embedding)
                return embedding_vector / norm(embedding_vector)
            else:
                logger.info("DeepFace.represent did not return an embedding for the provided pixels.")
                return None
        except Exception as e:
            logger.info(f"Error generating embedding with DeepFace.represent: {e}")
            return None

    def generate_embeddings(self, face_array: np.array, face: str = None):
        """
        Convert a face image (BGR format) to an embedding vector.
        """
        try:
            if face_array is None:
                logger.info("No image passed")
                return None
            img_rgb = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
            embedding_vector = self.get_embedding(img_rgb)
            return embedding_vector
        except Exception as e:
            logger.error("Unexpected error occurred while generating the embeddings %s", e)
            raise

    def save_embeddings(self):
        """
        Generate mean embeddings for all persons in the dataset and save them in one npz file.
        """
        try:
            folders = os.listdir(self.source_path)
            mean_embeddings = []
            names = []

            for folder in folders:
                folder_path = os.path.join(self.source_path, folder)
                logger.info(f"Generating embeddings for {folder}")
                if not os.path.isdir(folder_path):
                    continue

                embeddings = []
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    face_array = cv2.imread(file_path)
                    if face_array is None:
                        logger.warning(f"Could not read image: {file_path}")
                        continue
                    emb = self.generate_embeddings(face_array=face_array)
                    if emb is not None:
                        embeddings.append(emb)

                if len(embeddings) > 0:
                    logger.info(f"Embeddings generated successfully for {folder}")
                    embeddings = np.array(embeddings)
                    mean_embedding = np.mean(embeddings, axis=0)
                    mean_embedding = mean_embedding / norm(mean_embedding)
                    mean_embeddings.append(mean_embedding)
                    names.append(folder)
                else:
                    logger.error(f"No valid embeddings for {folder}")

            mean_embeddings = np.array(mean_embeddings)
            names = np.array(names)
            np.savez(self.destination_path, names=names, mean_embeddings=mean_embeddings)
            logger.info("Embeddings saved successfully!")

        except Exception as e:
            logger.error("Unexpected error occurred while saving the embeddings %s", e)
            raise
        
    def update_person(self, person_name: str):
        """
        Add or update embeddings for a single person (folder in source_path)
        without rebuilding the entire embeddings file.
        """
        try:
            person_folder = os.path.join(self.source_path, person_name)
            if not os.path.isdir(person_folder):
                logger.error(f"Folder not found for person: {person_name}")
                return

            # Load existing data if available
            if os.path.exists(self.destination_path):
                data = np.load(self.destination_path, allow_pickle=True)
                names = list(data["names"])
                mean_embeddings = list(data["mean_embeddings"])
            else:
                names, mean_embeddings = [], []

            # Compute new mean embedding for this person
            embeddings = []
            for file in os.listdir(person_folder):
                file_path = os.path.join(person_folder, file)
                face_array = cv2.imread(file_path)
                if face_array is None:
                    logger.warning(f"⚠️ Could not read image: {file_path}")
                    continue
                emb = self.generate_embeddings(face_array=face_array)
                if emb is not None:
                    embeddings.append(emb)

            if len(embeddings) == 0:
                logger.error(f"No valid embeddings found for {person_name}")
                return

            embeddings = np.array(embeddings)
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / norm(mean_emb)

            # Update existing entry or append new one
            if person_name in names:
                idx = names.index(person_name)
                mean_embeddings[idx] = mean_emb
                logger.info(f"Updated existing person: {person_name}")
            else:
                names.append(person_name)
                mean_embeddings.append(mean_emb)
                logger.info(f"Added new person: {person_name}")

            # Save updated arrays back
            np.savez(self.destination_path, names=np.array(names), mean_embeddings=np.array(mean_embeddings))
            logger.info(f"Embeddings updated successfully for {person_name}")

        except Exception as e:
            logger.error(f"Unexpected error occurred while updating {person_name}: {e}")
            raise
    