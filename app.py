import json
import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
from faceDetection import getFacialLandmarks
from dateValidation import get_date_taken
from expressionValidation import _checkExpression
from glassesValidation import checkGlasses
from lightingValidation import computeImage
from backgroundValidation import is_background_white
from colorValidation import _checkDynamicRange
from contrastValidation import _checkContrast
from autocropValidation import autoCrop

app = Flask(__name__)

def stringToImage(base64_string):
    try:
        imgdata = base64.b64decode(base64_string)
        return io.BytesIO(imgdata)
    except Exception as e:
        print(f"Error decoding base64 string: {str(e)}")
        return None

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/validate", methods=['POST'])
def validate():
    try:
        request_data = request.get_json()
        photo = request_data['photo_upload']
        
        convertedPhoto = stringToImage(photo)
        if convertedPhoto is None:
            return jsonify({'isDateValid': 'Invalid image data'}), 400
        
        # Load the image using PIL and save the loaded image to get the path for face detection function
        image = Image.open(convertedPhoto)
        temp_file_path = image.save("temp_image.jpg")

        # 1. detect face
        (face_data, error_message) = getFacialLandmarks("temp_image.jpg")
        if error_message:
            response = jsonify({
                "message": error_message,
                'status': 400
            })
            return response
        else:
            faceDetected, faceLandmarks = face_data

        # 2. check date when the photo is taken
        isDateValid, date_message = get_date_taken(image)
        # 3. check expression
        isExpressionValid, expression_message = _checkExpression(faceLandmarks)
        # 4. check for glasses
        isWearingGlasses, glasses_message = checkGlasses(image, faceLandmarks)
        # 5. check for lighting
        isLightingVisible, light_message = computeImage(image, faceLandmarks)
        # 6. check for white background
        isBgWhite, bg_message = is_background_white(image)
        # 7. check color range
        isColorVisible, color_message = _checkDynamicRange(image, faceLandmarks)
        # 8. check photo contrast
        isContrastVisible, contrast_message = _checkContrast(image, faceLandmarks)
        # 9. auto crop photo
        croppedImage, crop_message = autoCrop(image, faceDetected)
        
        response = jsonify({
            'isFaceDetected': {"status": True if faceDetected else False, "message": error_message if error_message else "Valid"},
            'isDateValid': {"status": isDateValid, "message": date_message},
            'isExpressionValid': {"status": isExpressionValid, "message": expression_message if expression_message else "Valid"},
            'isWearingGlasses': {"status": True if isWearingGlasses == False else False, "message": glasses_message if glasses_message else "Valid"},
            'isLightingVisible': {"status": isLightingVisible, "message": light_message if light_message else "Valid"},
            'isBgWhite': {"status": isBgWhite, "message": bg_message if bg_message else "Valid"},
            'isColorVisible': {"status": isColorVisible, "message": color_message if color_message else "Valid"},
            'isContrastVisible': {"status": isContrastVisible, "message": contrast_message if contrast_message else "Valid"},
            'croppedImage': {"status": croppedImage, "message": crop_message if crop_message else "Valid"}
        })

        os.remove("temp_image.jpg")

        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)