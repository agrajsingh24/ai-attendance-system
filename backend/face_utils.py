import json
import numpy as np
from deepface import DeepFace


def get_embedding(image_path):
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    return json.dumps(embedding)


def load_embedding(embedding_str):
    return np.array(json.loads(embedding_str))


def detect_faces(image_path):
    faces = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )
    return faces