from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from PIL import Image

def get_date_taken(image):
    try:
        # print(image)
        exif = image._getexif()
        if not exif:
            return False, 'Gagal mendapatkan metadata foto. Harap pilih foto yang lain'
        date_time_photo = exif[36867].split()[0]
        photo_date = datetime.strptime(date_time_photo, "%Y:%m:%d")
        
        expiry_date = datetime.today() + relativedelta(months=-6)
        if photo_date < expiry_date:
            return False, "Foto ini diambil lebih dari 6 bulan yang lalu. Harap pilih foto yang lebih baru"
        else:
            return True, str(photo_date)
    except Exception as e:
        return False, f"Gagal Membaca metadata foto: {str(e)}"