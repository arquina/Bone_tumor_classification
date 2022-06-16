from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import json
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
import cv2
import pandas as pd

def collate_fn(batch):
    return tuple(zip(*batch))

class xray_dataset_mask_multiview(Dataset):
    def __init__(self, root_dir, df, annotation, phase):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(df,'multiview_' + str(phase) + '.csv'))
        with open(annotation) as json_file:
            self.img_labels = json.load(json_file)
        new_dict = {}
        for key in self.img_labels['_via_img_metadata'].keys():
            new_key = key.split('.png')[0] + '.png'
            new_dict[new_key] = self.img_labels['_via_img_metadata'][key]
        self.img_labels['_via_img_metadata'] = new_dict
        self.phase = phase
        self.filelist = os.listdir(self.root_dir)

        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize((1024, 1024))

        self.normalize = transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                              std=[0.2422, 0.2422, 0.2422])

    def __len__(self):
        return max(list(self.df['multiview_index'])) + 1

    def __getitem__(self, idx):
        data_df = self.df[self.df['multiview_index'] == idx]
        identifier1 = list(data_df['image'])[0]
        identifier2 = list(data_df['image'])[1]

        info_dict_1 = self.img_labels['_via_img_metadata'][identifier1]
        info_dict_2 = self.img_labels['_via_img_metadata'][identifier2]
        img_path_1 = os.path.join(self.root_dir, info_dict_1['filename'])
        img_path_2 = os.path.join(self.root_dir, info_dict_2['filename'])
        image_1 = Image.open(img_path_1)
        image_2 = Image.open(img_path_2)
        self.mask_1 = np.zeros(image_1.size)
        self.mask_2 = np.zeros(image_2.size)

        region_dict_1 = info_dict_1['regions'][0]
        region_dict_2 = info_dict_2['regions'][0]

        info_dict_1 = info_dict_1['file_attributes']
        info_dict_2 = info_dict_2['file_attributes']
        shape_attribute_1 = region_dict_1['shape_attributes']
        shape_attribute_2 = region_dict_2['shape_attributes']
        polygon_1 = np.array(list(zip(shape_attribute_1['all_points_y'], shape_attribute_1['all_points_x'])))
        polygon_2 = np.array(list(zip(shape_attribute_2['all_points_y'], shape_attribute_2['all_points_x'])))
        cv2.fillPoly(self.mask_1, [polygon_1], 1)
        cv2.fillPoly(self.mask_2, [polygon_2], 1)
        mask_1 = Image.fromarray(self.mask_1.T)
        mask_2 = Image.fromarray(self.mask_2.T)

        patient = data_df.iloc[0,:]

        boxes_1 = []
        boxes_2 = []

        image_1 = self.grayscale(image_1)
        image_2 = self.grayscale(image_2)

        image_1 = self.resize(image_1)
        image_2 = self.resize(image_2)

        original_image_1 = image_1
        original_image_2 = image_2
        mask_1 = self.resize(mask_1)
        mask_2 = self.resize(mask_2)

        if self.phase == 'train':
            if random.random() > 0.5:
                image_1 = transforms.functional.vflip(image_1)
                mask_1 = transforms.functional.vflip(mask_1)
                image_2 = transforms.functional.vflip(image_2)
                mask_2 = transforms.functional.vflip(mask_2)

        pos_1 = np.where(mask_1)
        xmin_1 = np.min(pos_1[1])
        xmax_1 = np.max(pos_1[1])
        ymin_1 = np.min(pos_1[0])
        ymax_1 = np.max(pos_1[0])
        boxes_1.append([xmin_1, ymin_1, xmax_1, ymax_1])

        pos_2 = np.where(mask_2)
        xmin_2 = np.min(pos_2[1])
        xmax_2 = np.max(pos_2[1])
        ymin_2 = np.min(pos_2[0])
        ymax_2 = np.max(pos_2[0])
        boxes_2.append([xmin_2, ymin_2, xmax_2, ymax_2])

        image_1 = transforms.functional.to_tensor(image_1)
        mask_1 = transforms.functional.to_tensor(mask_1)

        image_2 = transforms.functional.to_tensor(image_2)
        mask_2 = transforms.functional.to_tensor(mask_2)

        #image = self.normalize(image)
        boxes_1 = torch.as_tensor(boxes_1, dtype=torch.float32)
        boxes_2 = torch.as_tensor(boxes_2, dtype=torch.float32)

        target_1 = {}
        target_1["boxes"] = boxes_1
        target_1["labels"] = torch.as_tensor([patient['type_code'].item()], dtype=torch.int64)
        target_1["subtype"] = torch.as_tensor(patient['subtype_code'].item(), dtype=torch.int64)
        target_1["fracture"] = torch.as_tensor(patient['fracture_code'].item(), dtype=torch.int64)
        target_1["modality"] = torch.as_tensor(patient['modality_code'].item(), dtype=torch.int64)
        target_1["image_id"] = torch.as_tensor([int(patient['image'].split('.')[0])], dtype=torch.int64)
        target_1["area"] = (boxes_1[:, 3] - boxes_1[:, 1]) * (boxes_1[:, 2] - boxes_1[:, 0])
        target_1["iscrowd"] = torch.zeros((1,), dtype=torch.uint8)
        target_1["masks"] = torch.as_tensor(mask_1, dtype=torch.uint8)

        target_2 = {}
        target_2["boxes"] = boxes_2
        target_2["labels"] = torch.as_tensor([patient['type_code'].item()], dtype=torch.int64)
        target_2["subtype"] = torch.as_tensor(patient['subtype_code'].item(), dtype=torch.int64)
        target_2["fracture"] = torch.as_tensor(patient['fracture_code'].item(), dtype=torch.int64)
        target_2["modality"] = torch.as_tensor(patient['modality_code'].item(), dtype=torch.int64)
        target_2["image_id"] = torch.as_tensor([int(patient['image'].split('.')[0])], dtype=torch.int64)
        target_2["area"] = (boxes_2[:, 3] - boxes_2[:, 1]) * (boxes_2[:, 2] - boxes_2[:, 0])
        target_2["iscrowd"] = torch.zeros((1,), dtype=torch.uint8)
        target_2["masks"] = torch.as_tensor(mask_2, dtype=torch.uint8)

        return (image_1,image_2), (target_1, target_2) , (original_image_1, original_image_2)

class xray_dataset_mask(Dataset):
    def __init__(self, root_dir, df_dir, annotation, phase, size):
        self.root_dir = root_dir
        if phase == 'train':
            self.df = pd.read_csv(os.path.join(df_dir, phase+ '_' + size + '.csv'))
        else:
            self.df = pd.read_csv(os.path.join(df_dir, phase + '.csv'))

        with open(annotation) as json_file:
            self.img_labels = json.load(json_file)
        new_dict = {}

        for key in self.img_labels['_via_img_metadata'].keys():
            new_key = key.split('.png')[0] + '.png'
            new_dict[new_key] = self.img_labels['_via_img_metadata'][key]

        self.img_labels['_via_img_metadata'] = new_dict
        self.phase = phase
        self.filelist = self.df['image']

        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.resize = transforms.Resize((1024, 1024))
        self.normalize = transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                              std=[0.2422, 0.2422, 0.2422])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        identifier = self.filelist[idx]
        image_id = int(identifier.split('.png')[0])
        info_dict = self.img_labels['_via_img_metadata'][identifier]
        img_path = os.path.join(self.root_dir, info_dict['filename'])
        image = Image.open(img_path)
        data_df = self.df[self.df['image'] == identifier]
        self.mask = np.zeros(image.size)

        region_dict = info_dict['regions'][0]

        info_dict = info_dict['file_attributes']
        shape_attribute = region_dict['shape_attributes']
        polygon = np.array(list(zip(shape_attribute['all_points_y'], shape_attribute['all_points_x'])))
        cv2.fillPoly(self.mask, [polygon], 1)
        mask = Image.fromarray(self.mask.T)
        labels = data_df['type_code'].item()

        boxes = []

        image = self.grayscale(image)
        image = self.resize(image)
        original_image = image
        mask = self.resize(mask)

        if self.phase == 'train':
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor([labels], dtype=torch.int64)
        target["subtype"] = torch.as_tensor(data_df['subtype_code'].item(), dtype=torch.int64)
        target["fracture"] = torch.as_tensor(data_df['fracture_code'].item(), dtype=torch.int64)
        target["modality"] = torch.as_tensor(data_df['modality_code'].item(), dtype=torch.int64)
        target["Age"] = torch.as_tensor(data_df['Age'].item(), dtype = torch.float64)
        target["image_id"] = torch.as_tensor([image_id], dtype=torch.int64)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((1,), dtype=torch.uint8)
        target["masks"] = torch.as_tensor(mask, dtype=torch.uint8)

        return image, target, original_image

class xray_dataset_class(Dataset):
    def __init__(self, root_dir, phase, fold_dir, total_fold, current_fold):
        self.root_dir = root_dir
        if phase == 'train':
            self.df = pd.DataFrame()
            for index in range(total_fold):
                if index is not current_fold:
                    temp_df = pd.read_csv(os.path.join(fold_dir, 'fold' + str(index) + '.csv'))
                    if len(self.df) == 0 :
                        self.df = temp_df
                    else:
                        self.df = pd.concat([self.df,temp_df])
        else:
            self.df = pd.read_csv(os.path.join(fold_dir, 'fold' + str(current_fold) + '.csv'))

        self.filelist = list(self.df['image'])

        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                     std=[0.2422, 0.2422, 0.2422])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                     std=[0.2422, 0.2422, 0.2422])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.filelist[idx]
        info_dict = self.df[self.df['image'] == image_name]
        img_path = os.path.join(self.root_dir, image_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        target = {}
        target["type"] = torch.as_tensor(info_dict['type_code'].item()-1, dtype = torch.int64)
        target['img_path'] = img_path

        return image, target

class xray_dataset_external(Dataset):
    def __init__(self, task, multiview, external_dir, external_data):
        self.df = pd.read_excel(external_data)
        self.root_dir = external_dir
        self.filelist = os.listdir(self.root_dir)
        self.task = task
        self.multiview = multiview
        if self.task != 'class':
            self.grayscale = transforms.Grayscale(num_output_channels=3)
            self.resize = transforms.Resize((1024, 1024))
            self.normalize =transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                                std=[0.2422, 0.2422, 0.2422])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2915, 0.2915, 0.2915],
                                     std=[0.2422, 0.2422, 0.2422])
            ])


    def __len__(self):
        if self.multiview:
            return 46
        else:
            return len(self.filelist)

    def __getitem__(self, idx):
        if self.task == 'mask':
            if self.multiview:
                data_df = self.df[self.df['multiview_index'] == idx]
                identifier1 = list(data_df['Image'])[0]
                identifier2 = list(data_df['Image'])[1]
                image_id1 = identifier1.split('.tif')[0] + ('.png')
                image_id2 = identifier2.split('.tif')[0] + ('.png')
                data_df1 = data_df[data_df['Image'] == identifier1]
                data_df2 = data_df[data_df['Image'] == identifier2]
                image_path1 = os.path.join(self.root_dir, image_id1)
                image_path2 = os.path.join(self.root_dir, image_id2)
                image1 = Image.open(image_path1)
                image2 = Image.open(image_path2)

                image1 = self.grayscale(image1)
                image1 = self.resize(image1)
                image1 = transforms.functional.to_tensor(image1)

                image2 = self.grayscale(image2)
                image2 = self.resize(image2)
                image2 = transforms.functional.to_tensor(image2)

                target1 = {}
                target1['doctor'] = torch.as_tensor([data_df1['doctor_code'].item()], dtype=torch.int64)
                target1['labels'] = torch.as_tensor(data_df1['type_code'].item(), dtype=torch.int64)

                target2 = {}
                target2['doctor'] = torch.as_tensor([data_df2['doctor_code'].item()], dtype=torch.int64)
                target2['labels'] = torch.as_tensor(data_df2['type_code'].item(), dtype=torch.int64)

                return (image1, image2), (target1, target2)

            else:
                identifier = self.filelist[idx]
                img_path = os.path.join(self.root_dir, identifier)
                image = Image.open(img_path)
                identifier_tif = identifier.split('.png')[0] + '.tif'
                data_df = self.df[self.df['Image'] == identifier_tif]

                image = self.grayscale(image)
                image = self.resize(image)
                image = transforms.functional.to_tensor(image)

                target = {}
                target['doctor'] = torch.as_tensor([data_df['doctor_code'].item()], dtype=torch.int64)
                target['labels'] = torch.as_tensor(data_df['type_code'].item(), dtype = torch.int64)

                return image, target

        elif self.task == 'class':
            identifier = self.filelist[idx]
            img_path = os.path.join(self.root_dir, identifier)
            image = Image.open(img_path)
            identifier_tif = identifier.split('.png')[0] + '.tif'
            data_df = self.df[self.df['Image'] == identifier_tif]
            if self.transform:
                image = self.transform(image)
            target = {}
            target['type'] = torch.as_tensor(data_df['type_code'].item()-1, dtype = torch.int64)
            target['img_path'] = img_path

        return image, target
