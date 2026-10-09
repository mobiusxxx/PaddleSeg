import cv2
import os
img_floder = r'F:\datasets\rope_ladder\images/'
label_floder = r'F:\datasets\rope_ladder\labels/'

#读取所有图片，如果读取为空删除该图片以及对应的标签
img_names = os.listdir(img_floder)
for img_name in img_names:
    img_path = img_floder + img_name
    img = cv2.imread(img_path)
    if img is None:
        os.remove(img_path)
        label_path = label_floder + img_name.split('.')[0] + '.png'
        os.remove(label_path)
        print(img_path + ' has been removed')

a = cv2.imread(r'F:/datasets/rope_ladder/747.png')
print(a.shape)