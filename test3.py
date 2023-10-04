import cv2
import numpy as np

# Baca citra dari file
image = cv2.imread('jeruk.jpeg', cv2.IMREAD_GRAYSCALE)

# Buat kernel 3x3 untuk operasi morfologi
kernel = np.ones((3, 3), np.uint8)

# Lakukan erosi diikuti oleh dilasi (opening)
opening_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Tampilkan citra asli dan hasil opening
cv2.imshow('Citra Asli', image)
cv2.imshow('Hasil Opening', opening_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
