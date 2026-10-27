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
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)



name_to_class_ids_file = os.path.join('', 'image_name_to_class_id_and_name.json')

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
        img = Image.open(path).convert('RGB')
        if self.model_name == "tf2torch_resnet_v2_101":
            img = transforms.Resize((299, 299))(img)
            img = transforms.Compose([transforms.ToTensor()])(img)
        else:
            if self.transform is not None:
                img = self.transform(img)
        return img

    def __getitem__(self, index):

        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]

        img = self._process_image(path)
        cf_index = (index + 1) % 1000

        cf_path = self.paths[cf_index]
        cf_img = self._process_image(cf_path)

        return img, class_id, class_name, image_name, cf_img


class CounterfactualDataset_5cf_NAA_dataset(data.Dataset):
    def __init__(self, model_name, adv_path, csv_file):
        """
        Args:
            model_name: Model name, used to determine the preprocessing size.
            adv_path: Root directory where the image files are stored.
            csv_file: Path to the CSV file containing image_name and class_id.
        """
        self.df = pd.read_csv(csv_file)
        self.adv_path = adv_path
        self.model_name = model_name
        try:
            self.transform = transforms_imagenet_wo_resize(params(model_name))
        except NameError:
            self.transform = None

        print(f'Using {len(self.df)} images from CSV.')

    def __len__(self):
        return len(self.df)

    def _process_image(self, image_name):

        path = os.path.join(self.adv_path, image_name)
        img = Image.open(path).convert('RGB')

        target_size_models = [
            "tf2torch_resnet_v2_101",
            "inception_v3",
            "inception_v4",
            "inception_resnet_v2"
        ]

        if self.model_name in target_size_models:

            p = params(self.model_name)
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

                img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])(img)
        return img

    def __getitem__(self, index):

        row = self.df.iloc[index]
        image_name = row['Image_name']
        class_id = int(row['Class_id'])
        class_name = row['Class_name'] if 'class_name' in self.df.columns else ""
        img = self._process_image(image_name)
        all_indices = list(range(len(self.df)))
        all_indices.remove(index)
        cf_indices = random.sample(all_indices, 5)

        cf_imgs = []
        for cf_idx in cf_indices:
            cf_row = self.df.iloc[cf_idx]
            cf_image_name = cf_row['Image_name']
            cf_imgs.append(self._process_image(cf_image_name))

        cf_imgs_tensor = torch.stack(cf_imgs, dim=0)
        return img, class_id, class_name, image_name, cf_imgs_tensor


class CounterfactualDataset_5cf_NAA_dataset_resnet(data.Dataset):
    def __init__(self, model_name, adv_path, csv_file):
        """
            Args:
                model_name: Model name, used to determine the preprocessing size.
                adv_path: Root directory where the image files are stored.
                csv_file: Path to the CSV file containing image_name and class_id.
        """

        self.df = pd.read_csv(csv_file)
        self.adv_path = adv_path
        self.model_name = model_name

        try:
            self.transform = transforms_imagenet_wo_resize(params(model_name))
        except NameError:
            self.transform = None
        print(f'Using {len(self.df)} images from CSV. Adapt to model: {model_name}')

    def __len__(self):
        return len(self.df)
    def _process_image(self, image_name):

        path = os.path.join(self.adv_path, image_name)
        img = Image.open(path).convert('RGB')
        target_299_models = [
            "tf2torch_resnet_v2_101",
            "inception_v3",
            "inception_v4",
            "inception_resnet_v2"
        ]

        target_224_resnet_models = [
            "resnet50",
            "resnet152"
        ]
        if self.model_name in target_299_models:
            p = params(self.model_name)
            transform_pipeline = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])
            ])
            img = transform_pipeline(img)
        elif self.model_name in target_224_resnet_models:
            p = params(self.model_name)
            transform_pipeline = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=p['mean'], std=p['std'])
            ])
            img = transform_pipeline(img)
        else:
            if self.transform is not None:
                p = params(self.model_name)
                transform_pipeline = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=p['mean'], std=p['std'])
                ])
                img = transform_pipeline(img)
            else:
                img = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])(img)
        return img

    def __getitem__(self, index):

        row = self.df.iloc[index]
        image_name = row['Image_name']
        class_id = int(row['Class_id'])
        class_name = row['Class_name'] if 'Class_name' in self.df.columns else ""
        img = self._process_image(image_name)
        all_indices = list(range(len(self.df)))
        all_indices.remove(index)
        cf_indices = random.sample(all_indices, 5)
        cf_imgs = []
        for cf_idx in cf_indices:
            cf_row = self.df.iloc[cf_idx]
            cf_image_name = cf_row['Image_name']
            cf_imgs.append(self._process_image(cf_image_name))
        cf_imgs_tensor = torch.stack(cf_imgs, dim=0)
        return img, class_id, class_name, image_name, cf_imgs_tensor

class CounterfactualDataset_5cf(data.Dataset):
    def __init__(self, model_name, adv_path, name_to_class_ids_file):
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

        img = Image.open(path).convert('RGB')
        p = params(self.model_name)
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

        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        img = self._process_image(path)
        cf_imgs = []
        available_indices = list(range(len(self.paths)))
        available_indices.remove(index)
        cf_indices = random.sample(available_indices, 5)
        for cf_idx in cf_indices:
            cf_path = self.paths[cf_idx]
            cf_imgs.append(self._process_image(cf_path))
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


        else:
            if self.transform is not None:
                img = self.transform(img)
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

        img = Image.open(path).convert('RGB')
        img = transforms.Resize((299,299))(img)
        img = transforms.Compose([transforms.ToTensor()])(img)

        return img, class_id, class_name, image_name
