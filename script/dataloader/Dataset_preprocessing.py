#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:20:49 2022

@author: seob
"""

import json
import os
import pandas as pd
import shutil
import random
from sklearn.preprocessing import LabelEncoder
import pickle

random.seed(1234567)

root_dir = "/home/seob/class_project/School_project/dataset/train_mask_final/"
small_dir = "/home/seob/class_project/School_project/dataset/train_mask_small/"
mid_dir = "/home/seob/class_project/School_project/dataset/train_mask_mid/"

data = os.listdir(root_dir)
small = random.sample(data, 200)
mid = random.sample(data, 500)

for image in small:
    shutil.copy(os.path.join(root_dir, image), os.path.join(small_dir, image))
for image in mid:
    shutil.copy(os.path.join(root_dir, image), os.path.join(mid_dir, image))
'''
root_dir = "/home/seob/class_project/School_project/dataset/"
df = pd.read_csv("/home/seob/class_project/School_project/dataset/total_dataset_additional.csv")

patient = list(set(list(df['patient'])))
random.shuffle(patient)
train_patient_len = int(len(patient) * 0.8)
train_patient = patient[0:train_patient_len]
test_patient = patient[train_patient_len:]

train_df = df[df['patient'].isin(train_patient)]
test_df = df[df['patient'].isin(test_patient)]

train_df = train_df[train_df['type'] != 'Normal']
test_df = test_df[test_df['type'] != 'Normal']

train_image = list(train_df['image'])
test_image = list(test_df['image'])

original_dir = os.path.join(root_dir, 'total_dataset')
train_dir = os.path.join(root_dir, 'train_mask_final')
test_dir = os.path.join(root_dir, 'test_mask_final')

if os.path.exists(train_dir) is False:
    os.mkdir(train_dir)
if os.path.exists(test_dir) is False:
    os.mkdir(test_dir)


root_dir = "/home/seob/class_project/School_project/dataset/"
image_dir = os.path.join(root_dir, "total_dataset")
train_diag = os.path.join(root_dir, "train_class")
test_diag = os.path.join(root_dir, "test_class")
train_mask = os.path.join(root_dir, 'train_mask')
test_mask = os.path.join(root_dir, 'test_mask')

annotation = "/home/seob/class_project/School_project/from_shin/Bone_tumor_20220522/via-2.0.11/Bone_Tumor/Bone_tumor_20220522.json"
#annotation = "/home/seob/class_project/School_project/from_shin/Bone_tumor_20220530/via-2.0.11/Bone_Tumor/Bone_tumor_20220530.json"
df
with open("/home/seob/class_project/School_project/dataset/age_dict.pickle", 'rb') as fr:
    age_dict = pickle.load(fr)

with open(annotation) as json_file:
    meta_data = json.load(json_file)
filelist = meta_data['_via_image_id_list']

diagnosis_dict = {}
diagnosis_dict['Normal'] = []
diagnosis_dict['Malignant'] = []
diagnosis_dict['Benign'] = []
diagnosis_dict['Intermediate'] = []

mask_dict = {}
mask_dict['Benign'] = []
mask_dict['Malignant'] = []
mask_dict['Normal'] = []
mask_dict['Intermediate'] = []


class_list = []
train_list = []
test_list = []

for idx in range(len(filelist)):
    image_dict = {}
    region_dict = {}

    identifier = filelist[idx]
    file_name = identifier.split('.png')[0] + '.png'
    file = os.path.join(root_dir, 'total_dataset', file_name)

    info_dict = meta_data['_via_img_metadata'][identifier]
    region_attribute = info_dict['regions']
    file_attribute = info_dict['file_attributes']

    if len(region_attribute) != 0:
        if file_name in train_image:
            train_list.append(file_name)
        elif file_name in test_image:
            test_list.append(file_name)

    image_dict['image'] = info_dict['filename']
    image_dict['patient'] = file_attribute['ID'].split('\n')[0]
    if image_dict['patient'] == '':
        image_dict['patient'] = image_dict['image'].split('.png')[0][0:5]
    if int(image_dict['patient']) in age_dict.keys():
        image_dict['Age'] = age_dict[int(image_dict['patient'])]
    else:
        image_dict['Age'] = file_attribute['Age']
    image_dict['type'] = file_attribute['Diagnosis_type']
    image_dict['subtype'] = file_attribute['Diagnosis_subtype']
    image_dict['fracture'] = file_attribute['Fracture']
    image_dict['modality'] = file_attribute['Modality']

    class_list.append(image_dict)

    if len(region_attribute) != 0 :
        mask_dict[file_attribute['Diagnosis_type']].append(file_name)
    
for image in train_list:
    shutil.copy(os.path.join(original_dir, image), os.path.join(train_dir, image))
for image in test_list:
    shutil.copy(os.path.join(original_dir, image), os.path.join(test_dir, image))


df = pd.DataFrame(class_list)

le = LabelEncoder()
df['type_code'] = le.fit_transform(df['type'])
df['subtype_code'] = le.fit_transform(df['subtype'])
df['fracture_code'] = le.fit_transform(df['fracture'])
df['modality_code'] = le.fit_transform(df['modality'])

patient_list = list(set(list(df['patient'])))
random.shuffle(patient_list)

df.to_csv("/home/seob/class_project/School_project/dataset/total_dataset.csv")



#diag_malignant = diagnosis_dict['Malignant']
#diag_benign = diagnosis_dict['Benign']
#diag_normal = diagnosis_dict['Normal']

mask_malignant = mask_dict['Malignant']
mask_benign = mask_dict['Benign']
mask_intermediate = mask_dict['Intermediate']

#diag_malignant_train = int(len(diag_malignant) * 0.8)
#diag_benign_train = int(len(diag_benign) * 0.8)
#diag_normal_train = int(len(diag_normal) * 0.8)

mask_malignant_train = int(len(mask_malignant) * 0.8)
mask_benign_train = int(len(mask_benign) * 0.8)
mask_intermediate_train = int(len(mask_intermediate) * 0.8)

#diag_malignant_train_list = random.sample(diag_malignant, diag_malignant_train)
#diag_benign_train_list = random.sample(diag_benign, diag_benign_train)
#diag_normal_train_list = random.sample(diag_normal, diag_normal_train)

mask_malignant_train_list = random.sample(mask_malignant, mask_malignant_train)
mask_benign_train_list = random.sample(mask_benign, mask_benign_train)
mask_intermediate_train_list = random.sample(mask_intermediate, mask_intermediate_train)

#diag_malignant_test_list = [item for item in diag_malignant if item not in diag_malignant_train_list]
#diag_benign_test_list = [item for item in diag_benign if item not in diag_benign_train_list]
#diag_normal_test_list = [item for item in diag_normal if item not in diag_normal_train_list]

mask_malignant_test_list = [item for item in mask_malignant if item not in mask_malignant_train_list]
mask_benign_test_list = [item for item in mask_benign if item not in mask_benign_train_list]
mask_intermediate_test_list = [item for item in mask_intermediate if item not in mask_intermediate_train_list]

#total_diagnosis_train = diag_malignant_train_list + diag_benign_train_list + diag_normal_train_list
#total_diagnosis_test = diag_malignant_test_list + diag_benign_test_list + diag_normal_test_list
total_mask_train = mask_malignant_train_list + mask_benign_train_list + mask_intermediate_train_list
total_mask_test = mask_malignant_test_list + mask_benign_test_list + mask_intermediate_test_list

#for file in total_diagnosis_train:
#    if os.path.exists(os.path.join(train_diag, file)) is False:
#        shutil.copy(os.path.join(image_dir, file),
#                    os.path.join(train_diag, file))
#for file in total_diagnosis_test:
#    if os.path.exists(os.path.join(test_diag, file)) is False:
#        shutil.copy(os.path.join(image_dir, file),
#                os.path.join(test_diag, file))

for file in total_mask_train:
    if os.path.exists(os.path.join(train_mask, file)) is False:
        shutil.copy(os.path.join(image_dir, file),
                    os.path.join(train_mask, file))
for file in total_mask_test:
    if os.path.exists(os.path.join(test_mask, file)) is False:
        shutil.copy(os.path.join(image_dir, file),
                    os.path.join(test_mask, file))

'''