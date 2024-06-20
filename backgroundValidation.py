import numpy as np

def is_background_white(image, threshold=200, percentage=0.8):
    try:
        image_np = np.array(image)

        # Calculate the portion of the border to check
        start = int(image_np.shape[0] * 0.15)
        end = int(image_np.shape[0] * 0.85)

        # Check if the background is white by examining the border pixels
        top_border = image_np[0, :]  # top border
        left_border = image_np[start:end, 0]  # middle 70% of the left border
        right_border = image_np[start:end, -1]  # middle 70% of the right border

        # Concatenate the border pixels
        border_pixels = np.concatenate([top_border, left_border, right_border])

        # Count white pixels
        white_pixels = np.sum(border_pixels >= threshold)

        # Check the percentage of white pixels
        total_pixels = border_pixels.size
        if white_pixels / total_pixels >= percentage:
            return True, None
        else:
            return False, "Background foto bukan berwarna putih. Harap mengganti foto"
    except Exception as e:
        return False, f"Gagal Membaca background foto: {str(e)}"