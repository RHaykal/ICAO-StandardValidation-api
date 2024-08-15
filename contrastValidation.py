import cv2
import numpy
from math import sqrt

def _checkContrast(image, facial_landmarks):
    try:
        #load image data
        image_data = numpy.array(image)
        image_gray = cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
        
        #calculate global contrast
        contrast_global = (numpy.amax(image_gray) - numpy.amin(image_gray))
        contrast_global_precentage = contrast_global / 255   
        
        #calculate local contrast
        contrast_local = 0.0
        #get face area
        leftEyeCenter = (int((facial_landmarks[43][0] + facial_landmarks[44][0] + facial_landmarks[46][0] + facial_landmarks[47][0]) / 4), int((facial_landmarks[43][1] + facial_landmarks[44][1] + facial_landmarks[46][1] + facial_landmarks[47][1]) / 4))
        rightEyeCenter = (int((facial_landmarks[37][0] + facial_landmarks[38][0] + facial_landmarks[40][0] + facial_landmarks[41][0]) / 4), int((facial_landmarks[37][1] + facial_landmarks[38][1] + facial_landmarks[40][1] + facial_landmarks[41][1]) / 4))
        M = (int((leftEyeCenter[0] + rightEyeCenter[0]) / 2), int((leftEyeCenter[1] + rightEyeCenter[1]) / 2))
        y_hairline = M[1] -  int((facial_landmarks[8][1]-M[1])*(2.0/3.0))

        image_gray_face = image_gray[y_hairline : facial_landmarks[8][1] , facial_landmarks[0][0] : facial_landmarks[16][0]]

        #median filter kernel   
        M = numpy.asfarray((
        [1/8, 1/8, 1/8],
        [1/8, 0,1/8],
        [1/8,1/8, 1/8]))

        average_neighbors = (cv2.filter2D(image_gray_face, -1, M)).astype('uint8')  
        diff = image_gray_face - average_neighbors
        contrast_local = int((1/image_gray_face.size) * numpy.sum(diff))

        #calculate average deviation
        average_deviation = int(sqrt(numpy.var(image_gray)))

        #check if the check passed or not
        check = True
        if (contrast_local <= 60 or average_deviation <= 45 or average_deviation >= 100 or contrast_global <= 200):
            check = f"Foto Tidak Memiliki Kontras yang Sesuai Standard. Harap Gunakan Foto Lain {str(contrast_local), str(average_deviation), str(average_deviation), str(contrast_global),}"

        if check == True:
            return True, None 
        else :
            return False, check
    except Exception as e:
        return f"Gagal mendeteksi kontras foto: {str(e)}"