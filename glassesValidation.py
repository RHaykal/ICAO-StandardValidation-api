import numpy as np
import dlib
import cv2
from PIL import Image
import statistics

def checkGlasses(img, landmarks):
    try:
        nose_bridge_x = []
        nose_bridge_y = []
        #using only the value that is in the tuple bellow for a smaller detection area to enhance accuracy
        for i in [28,29,30,32]:
                nose_bridge_x.append(landmarks[i][0])
                nose_bridge_y.append(landmarks[i][1])

        ### x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)
        ### ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]
        # img2 = Image.open(path)
        img2 = img.crop((x_min,y_min,x_max,y_max))

        # untuk filter gambar agar mendapatkan hasil deteksi edge pixel yang lebih bersih dari noise, digunakan params apertureSize=3 dan L2gradient true
        # nilai params threshold 1 dan 2 dibiarkan 100 dan 200 agar tingkat deteksi edge lebih optimal
        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)

        #center strip
        edges_center = edges.T[(int(len(edges.T)/2))]
        # print(edges_center)

        count_255 = np.sum(edges_center == 255)
        # print(count_255)

        # if 255 in edges_center:
        if count_255 > 2 :
            return True, f"Foto terdeteksi memakai kacamata. Harap ganti foto lain"
        else:
            return False, None
    except Exception as e:
        return f"error when checking for glasses: {str(e)}"
