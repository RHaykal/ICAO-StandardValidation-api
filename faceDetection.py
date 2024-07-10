import dlib
import os
import numpy as np

def getFacialLandmarks(image_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join(os.path.dirname(os.path.realpath(__file__)), "shape_predictor_68_face_landmarks.dat"))

        # Use the NumPy array directly with dlib detector
        faces = detector(image_path)

        if len(faces) > 1:
            return None, "Terdapat lebih dari 1 orang di dalam gambar. Harap pilih foto lain"
        elif len(faces) == 0:
            return None, "Tidak ada wajah yang terdeteksi. Harap pilih foto lain"
        else:
            rect = faces[0]
            sp = predictor(image_path, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            return (rect, landmarks), None
    except Exception as e:
        return None, f"gagal mendeteksi wajah: {str(e)}"