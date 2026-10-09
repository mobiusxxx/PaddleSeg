'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2024-01-24 10:05:53
Description: 推理
'''
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
import paddleseg.transforms as T
import os
from paddleseg.core import predict
backbone = STDC1()
 
model = PPLiteSeg(num_classes=2,
                 backbone= backbone,
                 arm_out_chs = [32, 64, 128],
                 seg_head_inter_chs = [32, 64, 64],
                 pretrained=None)
 
transforms = T.Compose([
    T.Resize(target_size=(960, 720)),
    # T.RandomHorizontalFlip(),
    T.Normalize()
])
 
def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )
 
    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')
 
    return image_list, image_dir
 
if __name__ == '__main__':
    image_path = r'F:\datasets\rope_ladder\images\427.png' # 也可以输入一个包含图像的目录
    image_list, image_dir = get_image_list(image_path)
    predict(
            model,
            model_path='output/best_model/model.pdparams',
            transforms=transforms,
            image_list=image_list,
            image_dir=image_dir,
            save_dir='output/results'
        )
