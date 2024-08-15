import cv2
import numpy
import numpy.ma as ma
from matplotlib import pyplot as plt
import math
import sys
import dlib
from scipy import ndimage

def checkBackground(image):
    try:
        # """ Convert PIL image to an OpenCV image """
        open_cv_image = numpy.array(image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        #read image and convert to gray
        # image_data = cv2.imread(open_cv_image.image_path + open_cv_image.image_name )
        image_gray = cv2.cvtColor(open_cv_image,cv2.COLOR_BGR2GRAY)

        # Check if the image has a white background
        white_background_threshold = 200  # Threshold to consider a pixel as white
        white_background_max_percentage_threshold = 0.57  # Percentage of maximum pixels that should be white
        white_background_min_percentage_threshold = 0.30 # Percentage of minimum pixels that should be white

        #checking the red, green, and blue value for detecting white pixels
        white_pixels = numpy.sum(
            (open_cv_image[:, :, 0] > white_background_threshold) &
            (open_cv_image[:, :, 1] > white_background_threshold) &
            (open_cv_image[:, :, 2] > white_background_threshold)
        )
        total_pixels = open_cv_image.shape[0] * open_cv_image.shape[1]
        white_background_percentage = white_pixels / total_pixels

        # white_pixels = numpy.sum(image_gray > white_background_threshold)
        # total_pixels = image_gray.size
        # white_background_percentage = white_pixels / total_pixels

        if not (white_background_min_percentage_threshold <= white_background_percentage <= white_background_max_percentage_threshold):
            return False, f"Background tidak sesuai. Harap pilih foto dengan latar belakang putih/Foto yang menggunakan baju selain putih. {str(white_pixels), str(total_pixels), str(white_background_percentage), str(white_background_max_percentage_threshold), str(white_background_min_percentage_threshold)}"
        
        #edge detection
        image_filter = cv2.Canny(image_gray,10,80)
        
        #closing of edges
        image_filter = cv2.dilate(image_filter, None,iterations=10)
        image_filter = cv2.erode(image_filter, None,iterations=10)

        #find longest contour at edges
        #used http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html
        contour_info = []
        contours, _= cv2.findContours(image_filter, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]

        #create background mask
        mask = numpy.zeros(image_filter.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))
        
        # Use float matrices
        mask_stack  = mask.astype('float32') / 255.0 
        #  for easy blending           
        img         = image_gray.astype('float32') / 255.0    
        # Blend mask with picture
        masked =  ((1-mask_stack) * img) 
        # Convert back to 8-bit
        masked = (masked * 255).astype('uint8')                
        
        #create masked array
        image_gray_masked = ma.masked_array(image_gray,mask_stack)  

        #calculate entropy with histogram
        hist = numpy.histogram(image_gray_masked.compressed(),range(0, 256),density=True)
        entropy = 0
        for x in hist[0]:
            if (x != 0.0) : entropy = entropy - (x * math.log2(x))

        #create mask for floodfill
        h, w = image_gray.shape[:2]
        mask_flood = numpy.pad(mask,1,mode="constant").astype('uint8') 
        mask_flood = numpy.array_split(mask_flood,2,axis=1)
        mask_flood = numpy.concatenate((mask_flood[1],mask_flood[0]), axis=1)

        #split picture vertical, and switch sides
        image_gray_split = numpy.array_split(image_gray,2,axis=1)
        image_gray_split = numpy.concatenate((image_gray_split[1],image_gray_split[0]), axis=1)

        
        
        image_floodfill = image_gray_split.copy()
        # Floodfill from point (0, 0)
        cv2.floodFill(image_floodfill, mask_flood, (int(w/2),0), 0, loDiff=1,upDiff=1 )
        
        
        #calculate how many pixels are not in floodfill and mask
        image_floodfill = numpy.array_split(image_floodfill,2,axis=1)
        image_floodfill = numpy.concatenate((image_floodfill[1],image_floodfill[0]), axis=1)
        image_floodfill  = image_floodfill.astype('float32') / 255.0        
        masked         = masked.astype('float32') / 255.0                 
        image_backgroundresult =  (image_floodfill * masked) 
        image_backgroundresult = (image_backgroundresult * 255).astype('uint8')   

        nonzeros = cv2.countNonZero(image_backgroundresult)


        #compare median from corner with background pixels
        #from 5% left corner
        background_average = int(numpy.average(image_gray[0:int(image_gray.shape[1]*0.05), 0:int(image_gray.shape[1]*0.05)]))
        background_mask = mask.copy()
        background_mask [mask == 0.0] = 255
        background_mask [mask == 255.0] = 0
        background_count = numpy.count_nonzero(background_mask)
        background_mask = background_mask.astype('float32') / 255.0

        background_averagedeviation = ((background_average-50 <= image_gray) & (image_gray <= background_average+50))
        background_averagedeviation[background_averagedeviation==True] = 255 
        background_averagedeviation[background_averagedeviation==False] = 0
        background_averagedeviation.astype('float32') / 255.0
        background_averagedeviation = ((background_averagedeviation*background_mask)*255).astype('uint8')


        background_averagedeviation_count = numpy.count_nonzero(background_averagedeviation) 
        background_inconform_pixels = background_count - background_averagedeviation_count
        background_inconform_pixels_percentage = background_inconform_pixels / image_gray.size

        #check if the check passed or not
        # if background_inconform_pixels_percentage > 0.02 :
        if background_inconform_pixels_percentage > 0.25 :
            # if entropy > 1:
            #     return False, "Struktur data foto terdeteksi bermasalah. Harap pilih foto lain"
            # else :
            return False, f"Background tidak sesuai. Harap pilih foto lain {str(background_inconform_pixels_percentage)}"
        return True, None
    except Exception as e:
        return False, f"Gagal Membaca background foto: {str(e)}"


# def is_background_white(image, threshold=200, percentage=0.8):
#     try:
#         image_np = np.array(image)

#         # Calculate the portion of the border to check
#         start = int(image_np.shape[0] * 0.15)
#         end = int(image_np.shape[0] * 0.85)

#         # Check if the background is white by examining the border pixels
#         top_border = image_np[0, :]  # top border
#         left_border = image_np[start:end, 0]  # middle 70% of the left border
#         right_border = image_np[start:end, -1]  # middle 70% of the right border

#         # Concatenate the border pixels
#         border_pixels = np.concatenate([top_border, left_border, right_border])

#         # Count white pixels
#         white_pixels = np.sum(border_pixels >= threshold)

#         # Check the percentage of white pixels
#         total_pixels = border_pixels.size
#         if white_pixels / total_pixels >= percentage:
#             return True, None
#         else:
#             return False, "Background foto bukan berwarna putih. Harap mengganti foto"
#     except Exception as e:
#         return False, f"Gagal Membaca background foto: {str(e)}"