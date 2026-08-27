# coding: utf-8
import os
import numpy as np
# from scipy.misc import imresize,imread,imsave
from PIL import Image
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, resnet_v1, vgg, nets_factory

slim = tf.contrib.slim


def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]

def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2

def inv_vgg_normalization(image):
  return np.clip(image + [123.68, 116.78, 103.94],0,255)

def inv_inception_normalization(image):
  return np.clip((image + 1.0) * 0.5 * 255,0,255)

normalization_fn_map = {
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': inception_normalization,
    'resnet_v2_101': inception_normalization,
    'resnet_v2_152': inception_normalization,
    'resnet_v2_200': inception_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
}

inv_normalization_fn_map = {
    'inception_v1': inv_inception_normalization,
    'inception_v2': inv_inception_normalization,
    'inception_v3': inv_inception_normalization,
    'inception_v4': inv_inception_normalization,
    'inception_resnet_v2': inv_inception_normalization,
    'resnet_v1_50': inv_vgg_normalization,
    'resnet_v1_101': inv_vgg_normalization,
    'resnet_v1_152': inv_vgg_normalization,
    'resnet_v1_200': inv_vgg_normalization,
    'resnet_v2_50': inv_inception_normalization,
    'resnet_v2_101': inv_inception_normalization,
    'resnet_v2_152': inv_inception_normalization,
    'resnet_v2_200': inv_inception_normalization,
    'vgg_16': inv_vgg_normalization,
    'vgg_19': inv_vgg_normalization,
}

offset = {
    'inception_v1': 1,
    'inception_v2': 1,
    'inception_v3': 1,
    'inception_v4': 1,
    'inception_resnet_v2': 1,
    'resnet_v1_50': 0,
    'resnet_v1_101': 0,
    'resnet_v1_152': 0,
    'resnet_v1_200': 0,
    'resnet_v2_50': 1,
    'resnet_v2_101': 1,
    'resnet_v2_152': 1,
    'resnet_v2_200': 1,
    'vgg_16': 0,
    'vgg_19': 0,
  }

image_size={
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'inception_v4': 299,
    'inception_resnet_v2': 299,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg_16': 224,
    'vgg_19': 224,
  }
base_path='./models_tf'

checkpoint_paths = {
    'inception_v1': None,
    'inception_v2': None,
    'inception_v3': base_path+'/inception_v3.ckpt',
    'inception_v4': base_path+'/inception_v4.ckpt',
    'inception_resnet_v2': base_path+'/inception_resnet_v2_2016_08_30.ckpt',
    'resnet_v1_50': base_path+'/resnet_v1_50.ckpt',
    'resnet_v1_101': None,
    'resnet_v1_152': base_path+'/resnet_v1_152.ckpt',
    'resnet_v1_200': None,
    'resnet_v2_50': base_path+'/resnet_v2_50/resnet_v2_50.ckpt',
    'resnet_v2_101': None,
    'resnet_v2_152': base_path+'/resnet_v2_152/resnet_v2_152.ckpt',
    'resnet_v2_200': None,
    'vgg_16': base_path+'/vgg_16.ckpt',
    'vgg_19': base_path+'/vgg_19.ckpt',
    'adv_inception_v3':base_path+'/adv_inception_v3/adv_inception_v3.ckpt',
    'adv_inception_resnet_v2':base_path+'/adv_inception_resnet_v2/adv_inception_resnet_v2.ckpt',
    'ens3_adv_inception_v3':base_path+'/ens3_adv_inception_v3/ens3_adv_inception_v3.ckpt',
    'ens4_adv_inception_v3':base_path+'/ens4_adv_inception_v3/ens4_adv_inception_v3.ckpt',
    'ens_adv_inception_resnet_v2':base_path+'/ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt'
  }

ground_truth=None
with open('./labels.txt') as f:
    ground_truth=f.read().split('\n')[:-1]

# 获取制作对抗样本的图片,同时也获取基线图片,暂时一张一张的获取
def load_image(image_path, image_size, batch_size, load_num=1000):
    images = []
    images_xr = []
    images_xr_1 = []
    images_xr_2 = []  # 多用两张图片，IG2里面也是这么实现的
    images_xr_3 = []
    images_xr_4 = []

    filenames=[]
    labels=[]
    idx = 0
    count=0

    files = os.listdir(image_path)
    files.sort(key=lambda x: int(x[:-4]))
    for i, filename in enumerate(files):
        # 需要将参考参考图片的文件名称获取出来
        # 分割文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 第一张参考图片
        if (int(name) + 1) != 1000:
            xr_filename_number = (int(name) + 1) % 1000
        else:
            xr_filename_number = (int(name) + 1) # 变更为加1， 不然会因为批次的不同导致结果不同
        # 生成新文件名
        xr_filename = f"{xr_filename_number}{ext}"  # 这是参考图片的文件名称，用的是不同的类
        print(f'filename is: {filename}, xr_filename is:{xr_filename}')
        image_xr = Image.open(image_path + xr_filename)  # 获取参考图片
        image_xr = image_xr.resize((image_size, image_size))
        image_xr = np.array(image_xr)
        images_xr.append(image_xr)  # 不需要用到其他的东西，仅需要一张图片就可以了
        # 第2张
        if (int(name) + 2) != 1000:
            xr_filename_number_1 = (int(name) + 2) % 1000
        else:
            xr_filename_number_1 = (int(name) + 2) # 变更为加1， 不然会因为批次的不同导致结果不同
            # 生成新文件名
        xr_filename_1 = f"{xr_filename_number_1}{ext}"  # 这是参考图片的文件名称，用的是不同的类
        print(f'filename is: {filename}, xr_filename_1 is:{xr_filename_1}')
        image_xr_1 = Image.open(image_path + xr_filename_1)  # 获取参考图片
        image_xr_1 = image_xr_1.resize((image_size, image_size))
        image_xr_1 = np.array(image_xr_1)
        images_xr_1.append(image_xr_1)  # 不需要用到其他的东西，仅需要一张图片就可以了
        # 第3张
        if (int(name) + 3) != 1000:
            xr_filename_number_2 = (int(name) + 3) % 1000
        else:
            xr_filename_number_2 = (int(name) + 3) # 变更为加1， 不然会因为批次的不同导致结果不同
            # 生成新文件名
        xr_filename_2 = f"{xr_filename_number_2}{ext}"  # 这是参考图片的文件名称，用的是不同的类
        print(f'filename is: {filename}, xr_filename_2 is:{xr_filename_2}')
        image_xr_2 = Image.open(image_path + xr_filename_2)  # 获取参考图片
        image_xr_2 = image_xr_2.resize((image_size, image_size))
        image_xr_2 = np.array(image_xr_2)
        images_xr_2.append(image_xr_2)  # 不需要用到其他的东西，仅需要一张图片就可以了
        # 第4张
        if (int(name) + 4) != 1000:
            xr_filename_number_3 = (int(name) + 4) % 1000
        else:
            xr_filename_number_3 = (int(name) + 4)  # 变更为加1， 不然会因为批次的不同导致结果不同
            # 生成新文件名
        xr_filename_3 = f"{xr_filename_number_3}{ext}"  # 这是参考图片的文件名称，用的是不同的类
        print(f'filename is: {filename}, xr_filename_3 is:{xr_filename_3}')
        image_xr_3 = Image.open(image_path + xr_filename_3)  # 获取参考图片
        image_xr_3 = image_xr_3.resize((image_size, image_size))
        image_xr_3 = np.array(image_xr_3)
        images_xr_3.append(image_xr_3)  # 不需要用到其他的东西，仅需要一张图片就可以了

        # 第5张
        if (int(name) + 5) != 1000:
            xr_filename_number_4 = (int(name) + 5) % 1000
        else:
            xr_filename_number_4 = (int(name) + 5)  # 变更为加1， 不然会因为批次的不同导致结果不同
            # 生成新文件名
        xr_filename_4 = f"{xr_filename_number_4}{ext}"  # 这是参考图片的文件名称，用的是不同的类
        print(f'filename is: {filename}, xr_filename_4 is:{xr_filename_4}')
        image_xr_4 = Image.open(image_path + xr_filename_4)  # 获取参考图片
        image_xr_4 = image_xr_4.resize((image_size, image_size))
        image_xr_4 = np.array(image_xr_4)
        images_xr_4.append(image_xr_4)  # 不需要用到其他的东西，仅需要一张图片就可以了

        # 下面这段代码是加载用于制作对抗样本的良性样本的图片
        image = Image.open(image_path + filename)
        image = image.resize((image_size,image_size))
        image = np.array(image)
        images.append(image)
        filenames.append(filename)
        labels.append(int(ground_truth[i]))
        idx += 1
        if idx == batch_size:
            yield np.array(images),np.array(filenames),np.array(labels), np.array(images_xr), np.array(images_xr_1), np.array(images_xr_2), np.array(images_xr_3),np.array(images_xr_4)
            idx = 0
            images = []
            images_xr = []  #
            images_xr_1 = []  #
            images_xr_2 = []  #
            images_xr_3 = []  #
            images_xr_4 = []  #
            filenames = []
            labels = []
        count += 1
    if idx>0:
        yield np.array(images), np.array(filenames),np.array(labels), np.array(images_xr), np.array(images_xr_1), np.array(images_xr_2), np.array(images_xr_3),np.array(images_xr_4)

def save_image(images,names,output_dir):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        # imsave(output_dir+name,images[i].astype('uint8'))
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


if __name__=='__main__':
    pass


