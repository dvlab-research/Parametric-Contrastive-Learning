import sys
import cv2

prefix = "../../../../data/cocostuff_164k/annotations/train2017/"
train_list = open(sys.argv[1],"r").readlines()

image_label_numbers = [ 0 for i in range(171) ]
label_paths = []
for line in train_list:
    if line.endswith('labelTrainIds.png\n'):
       label_path = prefix+line[:-1]
       label_paths.append(label_path)

for label_path in label_paths:
    print(label_path," 88888888888888 ")
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    h,w = label.shape
    s = set()
    for i in range(h):
        for j in range(w):
            e = int(label[i][j])
            if e!=255:
               s.add(e)
    for e in s:
        image_label_numbers[e] += 1

for i in range(171):
    open("tmp.txt","a+").write(str(image_label_numbers[i])+"\n")
