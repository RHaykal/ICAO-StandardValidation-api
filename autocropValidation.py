import numpy as np
from PIL import Image
import base64
import io

def autoCrop(image, rect):
    try:
        image = np.array(image)

        original_width, original_height = image.shape[:2]

        if original_height == original_width:
            image_array = Image.fromarray(image)
            image_base64 = pil_image_to_base64(image_array)
            return True, image_base64
        else:
            # Calculate the size of the crop area (2x2 inches) in pixels based on the face size
            crop_size = int(rect.width() * 2)  # Assuming the face width is proportional to the actual face size

            # Find the hair top (first black pixel from the top)
            y_center = None
            for y in range(image.shape[0]):
                if image[y, rect.left()][0] == 0:  # Check if the pixel is black (assuming BGR image)
                    y_center = (rect.top() + y//3) -70
                    break

            # Calculate the size of the crop area (2x2 inches) in pixels based on the face size
            crop_size = int(rect.width() * 2)  # Assuming the face width is proportional to the actual face size

            # Use the hair top as y_center if found, otherwise use the center of the face
            if y_center is None:
                y_center = rect.top() + (rect.height() // 2)

            x_center = rect.left() + (rect.width() // 2)

            x_start = max(x_center - crop_size // 2, 0)
            y_start = max(y_center - crop_size // 2, 0)
            x_end = min(x_start + crop_size +50, image.shape[1])
            y_end = min(y_start + crop_size +50, image.shape[0])

            # Crop the image
            cropped_image = image[y_start:y_end, x_start:x_end]

            # Resize the cropped image to make width and height equal
            if cropped_image.shape[0] != cropped_image.shape[1]:
                min_dim = min(cropped_image.shape[0], cropped_image.shape[1])
                cropped_image = cropped_image[:min_dim, :min_dim]

            # Convert the cropped image to PIL format
            cropped_image_pil = Image.fromarray(cropped_image)
            image_base64 = pil_image_to_base64(cropped_image_pil)

            return True, image_base64
    except Exception as e:
        return False, f"Gagal melakukan cropping foto."

def pil_image_to_base64(img):
    try:
        # Create a byte stream
        buffer = io.BytesIO()
        
        # Save the image to the byte stream
        img.save(buffer, format="JPEG")  # or another format like "PNG" if needed
        
        # Get the byte data from the byte stream
        byte_data = buffer.getvalue()
        
        # Encode the byte data to a base64 string
        base64_string = base64.b64encode(byte_data).decode('utf-8')
        
        return base64_string
    except Exception as e:
        return f"Gagal Melakukan Konversi Foto. Harap Mengganti Foto Dengan Format JPEG"