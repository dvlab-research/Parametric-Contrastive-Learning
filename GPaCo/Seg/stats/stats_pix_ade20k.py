import sys
import cv2

prefix = "../../../../../dataset/ADEChallengeData2016/annotations/training/"
train_list = open(sys.argv[1],"r").readlines()

pix_label_numbers = [ 0 for i in range(151) ]
label_paths = []
for line in train_list:
    label_path = prefix+line[:-1]
    label_paths.append(label_path)

for label_path in label_paths:
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    h,w = label.shape
    for i in range(h):
        for j in range(w):
            e = int(label[i][j])
            if e !=255:
               pix_label_numbers[e] += 1

for i in range(151):
    open("ade20k_pix_label_stats.txt","a+").write(str(pix_label_numbers[i])+"\n")
