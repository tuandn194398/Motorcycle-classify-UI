import os
import cv2
import numpy as np


def img_equalize(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the mean pixel intensity
    mean_intensity = np.mean(gray_image)
    if (20 < mean_intensity < 80) or (mean_intensity > 135):
        b, g, r = cv2.split(image)

        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        eq_image = cv2.merge((b_eq, g_eq, r_eq))

        return eq_image
    else:
        return image


folder_path = "E:/Users/Admin/Desktop/yolov9/data/input_data2"
output_folder = "E:/Users/Admin/Desktop/yolov9/data/equalized_input_data2"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        print(file_path)
        equalized_image = img_equalize(file_path)

        new_filename = f"equalized_{filename}"
        new_file_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(new_file_path, equalized_image)

print("Image processing complete.")
