import sys
import cv2
import os

prefix = "../../../../data/cityscapes/gtFine/train/"
pix_label_numbers = [ 0 for i in range(20) ]
label_paths = []

for root, dirs, files in os.walk(prefix):
    for c_file in files:
        if c_file.endswith('labelTrainIds.png'):
            label_path = os.path.join(root, c_file)
            label_paths.append(label_path)
            print(label_path)


for label_path in label_paths:
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    h,w = label.shape
    for i in range(h):
        for j in range(w):
            e = int(label[i][j])
            if e!=255:
               pix_label_numbers[e] += 1

for i in range(20):
    open("cityscapes_pix_label_stats.txt","a+").write(str(pix_label_numbers[i])+"\n")
