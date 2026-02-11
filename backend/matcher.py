import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_face(input_embedding, students, threshold=0.6):
    best_match = None
    best_score = -1

    for student in students:
        score = cosine_similarity(input_embedding, student["embedding"])
        if score > best_score:
            best_score = score
            best_match = student

    if best_score > threshold:
        return best_match, best_score

    return None, None
