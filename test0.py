import cv2
import numpy as np
import pandas as pd


# def get_pixel_values(image_path):
#     image = cv2.imread(Smage_path)
#     b, g, r = cv2.split(image)
#     pixels = cv2.merge((r, g, b))
#     return pixels


# image_path = 'cabai.png'
# pixels = get_pixel_values(image_path)
# for pixel in pixels:
#     print(f"R: {pixel[0]}, G:{pixel[1]}, B:{pixel[2]}")

def extract_features(image_path):
    image = cv2.imread(image_path)
    r, g, b = cv2.split(image)
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    return [avg_r, avg_g, avg_b]


image_path = "cabai.png"
features = extract_features(image_path)
feature_matrix = np.reshape(features, (1, len(features)))
data = pd.DataFrame(feature_matrix, columns=[
                    'Average R', 'Average G', 'Average B'])
data.to_excel('image_features.xlsx', index=False)
