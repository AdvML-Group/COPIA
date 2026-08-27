import torch.utils.data as data
import os
import torch
import json
from PIL import Image
from torchvision import transforms
import glob
import numpy as np
import random
import pandas as pd


seed = 2013
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果你有多个 GPU
random.seed(seed)
np.random.seed(seed)



name_to_class_ids_file = os.path.join('', 'image_name_to_class_id_and_name.json')
# params of dataset
# INPUT_SIZE = (3, 224, 224)
INTERPOLATION = 'bicubic'
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Imagenet 21k
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def params(model_name):
    if model_name in ['vit_deit_base_distilled_patch16_224','levit_256','pit_b_224','cait_s24_224','convit_base', 'visformer_small', 'deit_base_distilled_patch16_224','resnet50','resnet152']:
        params = {'mean': IMAGENET_DEFAULT_MEAN,
                  'std': IMAGENET_DEFAULT_STD,
                  'interpolation': INTERPOLATION,
                  'crop_pct':0.9}
    else:
        params = {'mean': IMAGENET_INCEPTION_MEAN,
                  'std': IMAGENET_INCEPTION_STD,
                  'interpolation': INTERPOLATION,
                  'crop_pct': 0.9}
    return params

def transforms_imagenet_wo_resize(params):

    tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(params['mean']),
                     std=torch.tensor(params['std']))
        ]
    return transforms.Compose(tfl)

class CounterfactualDataset(data.Dataset):
    def __init__(self, model_name, adv_path, name_to_class_ids_file):
        # 假设 params 和 transforms_imagenet_wo_resize 已经在你的环境中定义
        self.transform = transforms_imagenet_wo_resize(params(model_name))
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print('Using ', len(paths), ' images.')
        paths = [i.strip() for i in paths]
        
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        self.model_name = model_name
        
        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def _process_image(self, path):
        """将图片读取与预处理提取为独立方法，方便复用"""
        img = Image.open(path).convert('RGB')
        if self.model_name == "tf2torch_resnet_v2_101":
            img = transforms.Resize((299, 299))(img)
            img = transforms.Compose([transforms.ToTensor()])(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
        return img

    def __getitem__(self, index):
        # 1. 获取当前目标图像 (原图)
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        
        img = self._process_image(path)

        # 2. 获取反事实样本 (Counterfactual Image)
        # 策略：从数据集中随机抽取一张非当前的图片作为反事实样本
        cf_index = (index + 1) % 1000
            
        cf_path = self.paths[cf_index]
        cf_img = self._process_image(cf_path)

        # 返回时新增 cf_img
        return img, class_id, class_name, image_name, cf_img


class CounterfactualDataset_5cf_NAA_dataset(data.Dataset):
    def __init__(self, model_name, adv_path, csv_file):
        """
        Args:
            model_name: 模型名称，用于确定预处理尺寸
            adv_path: 图像文件存放的根目录
            csv_file: 包含 image_name 和 class_id 的 CSV 文件路径
        """
        # 加载 CSV 文件
        self.df = pd.read_csv(csv_file)
        self.adv_path = adv_path
        self.model_name = model_name

        # 假设 params 和 transforms_imagenet_wo_resize 已定义
        # 如果没有这些自定义函数，可以使用标准的 transforms.ToTensor()
        try:
            self.transform = transforms_imagenet_wo_resize(params(model_name))
        except NameError:
            self.transform = None

        print(f'Using {len(self.df)} images from CSV.')

    def __len__(self):
        return len(self.df)

    def _process_image(self, image_name):
        """
        根据图片名称构建路径并进行预处理
        """
        path = os.path.join(self.adv_path, image_name)
        img = Image.open(path).convert('RGB')

        # Inception 系列和特定的 ResNet 变体使用 299 尺寸
        target_size_models = [
            "tf2torch_resnet_v2_101",
            "inception_v3",
            "inception_v4",
            "inception_resnet_v2"
        ]

        if self.model_name in target_size_models:
            # 组合 Resize 和 ToTensor
            p = params(self.model_name)  # 获取对应 mean/std
            transform_pipeline = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])
            ])
            img = transform_pipeline(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
            else:
                # 默认保底处理
                img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])(img)
        return img

    def __getitem__(self, index):
        # 1. 从 DataFrame 中获取当前行信息
        row = self.df.iloc[index]
        image_name = row['Image_name']
        class_id = int(row['Class_id'])

        # 兼容原代码中的 class_name 逻辑，如果 CSV 没这一列则返回空
        class_name = row['Class_name'] if 'class_name' in self.df.columns else ""

        # 处理主图像
        img = self._process_image(image_name)

        # 2. 随机抽取 5 个反事实图像名称
        # 从所有索引中排除当前索引
        all_indices = list(range(len(self.df)))
        all_indices.remove(index)
        cf_indices = random.sample(all_indices, 5)

        cf_imgs = []
        for cf_idx in cf_indices:
            cf_row = self.df.iloc[cf_idx]
            cf_image_name = cf_row['Image_name']
            cf_imgs.append(self._process_image(cf_image_name))

        # 将 5 张图片堆叠，形状变为 [5, C, H, W]
        cf_imgs_tensor = torch.stack(cf_imgs, dim=0)

        return img, class_id, class_name, image_name, cf_imgs_tensor


class CounterfactualDataset_5cf_NAA_dataset_resnet(data.Dataset):
    def __init__(self, model_name, adv_path, csv_file):
        """
        Args:
            model_name: 模型名称，用于确定预处理尺寸与归一化参数
            adv_path: 图像文件存放的根目录
            csv_file: 包含 image_name 和 class_id 的 CSV 文件路径
        """
        # 加载 CSV 文件
        self.df = pd.read_csv(csv_file)
        self.adv_path = adv_path
        self.model_name = model_name

        # 针对不作特殊处理的ViT/Transformer或ResNet模型，使用通用的 transform
        try:
            self.transform = transforms_imagenet_wo_resize(params(model_name))
        except NameError:
            self.transform = None

        print(f'Using {len(self.df)} images from CSV. Adapt to model: {model_name}')

    def __len__(self):
        return len(self.df)

    def _process_image(self, image_name):
        """
        根据图片名称构建路径并进行预处理（已完美适配 ResNet-50 和 ResNet-152）
        """
        path = os.path.join(self.adv_path, image_name)
        img = Image.open(path).convert('RGB')

        # 明确指定需要缩放到 299x299 尺寸的 CNN 模型列表
        target_299_models = [
            "tf2torch_resnet_v2_101",
            "inception_v3",
            "inception_v4",
            "inception_resnet_v2"
        ]

        # ✅ 新增：明确指定需要缩放到 224x224 尺寸的 ResNet 模型列表
        target_224_resnet_models = [
            "resnet50",
            "resnet152"
        ]

        if self.model_name in target_299_models:
            # 1. Inception 系列：缩放到 299x299，并采用对应归一化
            p = params(self.model_name)
            transform_pipeline = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])
            ])
            img = transform_pipeline(img)

        elif self.model_name in target_224_resnet_models:
            # 2. ✅ ResNet 系列分支：缩放到 224x224，并使用 ImageNet 默认的归一化参数
            p = params(self.model_name)  # 根据你的 params(model_name) 逻辑，会正确匹配 IMAGENET_DEFAULT_MEAN / STD
            transform_pipeline = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])
            ])
            img = transform_pipeline(img)

        else:
            # 3. 其它模型（如 Transformer 家族等），直接应用初始定义的 self.transform
            # 如果初始 transform 缺失 resize 逻辑，则在此处进行保底处理
            if self.transform is not None:
                # 注意：你原本的 transforms_imagenet_wo_resize 缺少 Resize
                # 对于那些原生要求 224 的模型，安全起见我们加上强制缩放
                p = params(self.model_name)
                transform_pipeline = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=p['mean'], std=p['std'])
                ])
                img = transform_pipeline(img)
            else:
                # 默认绝对保底处理
                img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])(img)

        return img

    def __getitem__(self, index):
        # 1. 从 DataFrame 中获取当前行信息
        row = self.df.iloc[index]
        image_name = row['Image_name']
        class_id = int(row['Class_id'])

        # 兼容原代码中的 class_name 逻辑，如果 CSV 没这一列则返回空
        class_name = row['Class_name'] if 'Class_name' in self.df.columns else ""

        # 处理主图像
        img = self._process_image(image_name)

        # 2. 随机抽取 5 个反事实图像名称
        # 从所有索引中排除当前索引
        all_indices = list(range(len(self.df)))
        all_indices.remove(index)
        cf_indices = random.sample(all_indices, 5)

        cf_imgs = []
        for cf_idx in cf_indices:
            cf_row = self.df.iloc[cf_idx]
            cf_image_name = cf_row['Image_name']
            cf_imgs.append(self._process_image(cf_image_name))

        # 将 5 张图片堆叠，形状变为 [5, C, H, W]
        cf_imgs_tensor = torch.stack(cf_imgs, dim=0)

        return img, class_id, class_name, image_name, cf_imgs_tensor

class CounterfactualDataset_5cf(data.Dataset):
    def __init__(self, model_name, adv_path, name_to_class_ids_file):
        # 假设 params 和 transforms_imagenet_wo_resize 已经在你的环境中定义
        self.transform = transforms_imagenet_wo_resize(params(model_name))
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print('Using ', len(paths), ' images.')
        paths = [i.strip() for i in paths]
        
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        self.model_name = model_name
        
        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def _process_image(self, path):
        """将图片读取与预处理提取为独立方法，方便复用"""
        img = Image.open(path).convert('RGB')
        p = params(self.model_name)  # 获取对应 mean/std
        if self.model_name == "tf2torch_resnet_v2_101":
            img = transforms.Resize((299, 299))(img)
            img = transforms.Compose([transforms.ToTensor()])(img)
        elif self.model_name == "inception_v3":
            p = params(self.model_name)
            img = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])  # ← 补上
            ])(img)
        elif self.model_name == "inception_v4":
            p = params(self.model_name)
            img = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])  # ← 补上
            ])(img)
        elif self.model_name == "inception_resnet_v2":
            p = params(self.model_name)
            img = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])  # ← 补上
            ])(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
        return img

    def __getitem__(self, index):
        # 1. 获取当前目标图像 (原图)
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        
        img = self._process_image(path)

        cf_imgs = []
        available_indices = list(range(len(self.paths)))
        available_indices.remove(index)
        cf_indices = random.sample(available_indices, 5) # 随机不重复抽5张
        
        for cf_idx in cf_indices:
            cf_path = self.paths[cf_idx]
            cf_imgs.append(self._process_image(cf_path))

        # 将5张图片堆叠，形状变为 [5, C, H, W]
        # 在 DataLoader 输出后，形状将变为 [B, 5, C, H, W]
        cf_imgs_tensor = torch.stack(cf_imgs, dim=0)

        return img, class_id, class_name, image_name, cf_imgs_tensor

class AdvDataset(data.Dataset):
    def __init__(self, model_name, adv_path, name_to_class_ids_file=name_to_class_ids_file):
        self.transform = transforms_imagenet_wo_resize(params(model_name))
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        self.model_name = model_name
        
        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        # deal with image
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((224,224))(img)
        if self.model_name == "tf2torch_resnet_v2_101":
            img = transforms.Resize((299,299))(img)
            img = transforms.Compose([transforms.ToTensor()])(img)

        # elif self.model_name == "inception_v3":
        #     img = transforms.Resize((299,299))(img)
        #     img = transforms.Compose([transforms.ToTensor()])(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
        # image = Image.open(path)
        # image = image.resize((224, 224)).convert('RGB')
        # # Images for inception classifier are normalized to be in [-1, 1] interval.
        # image = np.array(image).astype(np.float32)/255
        # img = torch.from_numpy(image).permute(2, 0, 1)
        return img, class_id, class_name, image_name



class CNNDataset(data.Dataset):
    def __init__(self, model_name, adv_path, name_to_class_ids_file=name_to_class_ids_file):
        self.transform = transforms_imagenet_wo_resize(params(model_name))
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        
        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        # deal with image
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((299,299))(img)
        img = transforms.Compose([transforms.ToTensor()])(img)

        return img, class_id, class_name, image_name
