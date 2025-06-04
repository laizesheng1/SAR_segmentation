import random
from PIL import Image, ImageEnhance
import numpy as np

def random_brightness(img, lab, low=0.5, high=1.5):
    ''' 随机改变亮度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Brightness(img).enhance(x)
    return img, lab

def random_contrast(img, lab, low=0.5, high=1.5):
    ''' 随机改变对比度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Contrast(img).enhance(x)
    return img, lab

def random_color(img, lab, low=0.5, high=1.5):
    ''' 随机改变饱和度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Color(img).enhance(x)
    return img, lab

def random_sharpness(img, lab, low=0.5, high=1.5):
    ''' 随机改变清晰度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Sharpness(img).enhance(x)
    return img, lab

def random_rotate(img, lab, low=0, high=360):
    ''' 随机旋转图像(0~360度) '''
    angle = random.choice(range(low, high))
    img, lab = img.rotate(angle), lab.rotate(angle)
    return img, lab

def random_flip(img, lab, prob=0.5):
    ''' 随机翻转图像(p=0.5) '''
    if random.random() < prob:   # 上下翻转
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        lab = lab.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < prob:   # 左右翻转
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    return img, lab

def random_noise(img, lab, low=0, high=10):
    ''' 随机加高斯噪声(0~10) '''
    img = np.asarray(img).astype(np.float32)
    sigma = np.random.uniform(low, high)
    noise = np.random.randn(*img.shape) * sigma  # 不管是灰度还是RGB都能匹配
    img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img, lab

def image_augment(img, lab, prob=0.5):
    ''' 叠加多种数据增强方法 '''
    opts = [random_brightness, random_contrast, random_color, random_flip,
            random_noise, random_rotate, random_sharpness,]  # 数据增强方法
    for func in opts:
        if random.random() < prob:
            img, lab = func(img, lab)   # 处理图像和标签
    return img, lab


