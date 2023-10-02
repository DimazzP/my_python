from rembg import remove
import cv2
import matplotlib.pyplot as plt

# Membaca gambar input
input_image = cv2.imread('jeruk.jpeg')

# Menghapus latar belakang gambar
output_image = remove(input_image)

# Menampilkan gambar hasil
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
