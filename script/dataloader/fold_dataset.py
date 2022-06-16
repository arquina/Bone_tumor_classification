import pandas as pd
import random
import pickle
import os
import json

def fold_dataset(df):

    seed = 1234567
    random.seed(seed)
    type_dict = {}
    type_dict['Benign'] = []
    type_dict['Intermediate'] = []
    type_dict['Malignant'] = []

    for key in type_dict.keys():
        type_dict[key] = list(set(list(df[df['type'] == key]['patient'])))

    save_dir = "/home/seob/class_project/School_project/dataset/final_5_fold_dataset/"
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    random.shuffle(type_dict['Benign'])
    random.shuffle(type_dict['Intermediate'])
    random.shuffle(type_dict['Malignant'])

    fold = 5
    fold_dict = {}
    for key in type_dict.keys():
        fold_dict[key] = int(len(type_dict[key]) / 5)

    final_dict = {}
    for index in range(fold):
        fold_list = []
        index_list = []

        for key in type_dict.keys():
            if index != fold - 1:
                fold_list += type_dict[key][index * fold_dict[key]:(index + 1) * fold_dict[key]]
            else:
                fold_list += type_dict[key][index * fold_dict[key]:]
        for patient in fold_list:
            temp_df = df[df['patient'] == patient]
            index_list+= list(temp_df.index)
        final_dict[index] = index_list

    train_df = pd.DataFrame()
    train_small_df = pd.DataFrame()
    train_mid_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for key in final_dict.keys():
        fold_df = df.loc[final_dict[key],:].reset_index(drop=True)
        fold_df.to_csv(os.path.join(save_dir, 'fold' + str(key) + '.csv'), index = False)
        if key == 0:
            test_df = fold_df
        else:
            if key == 1:
                train_small_df = fold_df
                if len(train_mid_df) == 0:
                    train_mid_df = fold_df
                else:
                    train_mid_df = pd.concat([train_df, fold_df])
            elif key == 3:
                if len(train_mid_df) == 0:
                    train_mid_df = fold_df
                else:
                    train_mid_df = pd.concat([train_df, fold_df])
            if len(test_df) == 0:
                train_df = fold_df
            else:
                train_df = pd.concat([train_df, fold_df])

    train_df.to_csv(os.path.join(save_dir, 'train_big.csv'), index = False)
    train_small_df.to_csv(os.path.join(save_dir, 'train_small.csv'), index = False)
    train_mid_df.to_csv(os.path.join(save_dir, 'train_mid.csv'), index = False)
    test_df.to_csv(os.path.join(save_dir, 'test.csv'), index = False)

    for key in final_dict.keys():
        with open(os.path.join(save_dir, 'fold' + str(key)), 'wb') as f:
            pickle.dump(final_dict[key], f, pickle.HIGHEST_PROTOCOL)

def multiview_dataset(df):
    train_csv = pd.read_csv("/home/seob/class_project/School_project/dataset/final_5_fold_dataset/train_big.csv")
    test_csv = pd.read_csv("/home/seob/class_project/School_project/dataset/final_5_fold_dataset/test.csv")

    train_patient = list(set(list(train_csv['patient'])))
    test_patient = list(set(list(test_csv['patient'])))

    train_multiview = pd.DataFrame()
    for patient in train_patient:
        temp_df = train_csv[train_csv['patient'] == patient]
        if len(temp_df) != 1:
            if len(temp_df) %2 == 0:
                if len(train_multiview) != 0:
                    train_multiview = pd.concat([train_multiview, temp_df])
                else:
                    train_multiview = temp_df
            else:
                temp_df = temp_df.iloc[0:len(temp_df)-1,:]
                if len(train_multiview) != 0:
                    train_multiview = pd.concat([train_multiview, temp_df])
                else:
                    train_multiview = temp_df

    test_multiview = pd.DataFrame()
    for patient in test_patient:
        temp_df = test_csv[test_csv['patient'] == patient]
        if len(temp_df) != 1:
            if len(temp_df) % 2 == 0:
                if len(test_multiview) != 0:
                    test_multiview = pd.concat([test_multiview, temp_df])
                else:
                    test_multiview = temp_df
            else:
                temp_df = temp_df.iloc[0:len(temp_df) - 1, :]
                if len(test_multiview) != 0:
                    test_multiview = pd.concat([test_multiview, temp_df])
                else:
                    test_multiview = temp_df

    train_multiview = train_multiview.reset_index(drop = True)
    test_multiview = test_multiview.reset_index(drop = True)
    train_multiview_index_list = []
    test_multiview_index_list = []
    for idx, row in train_multiview.iterrows():
        train_multiview_index_list.append(int(idx/2))
    for idx, row in test_multiview.iterrows():
        test_multiview_index_list.append(int(idx/2))
    train_multiview['multiview_index'] = train_multiview_index_list
    test_multiview['multiview_index'] = test_multiview_index_list
    train_multiview.to_csv("/home/seob/class_project/School_project/dataset/final_5_fold_dataset/multiview_train.csv", index = False)
    test_multiview.to_csv("/home/seob/class_project/School_project/dataset/final_5_fold_dataset/mutiview_test.csv", index = False)

'''
df = pd.read_csv("/home/seob/class_project/School_project/dataset/total_dataset_additional.csv")
df = df[df['type'] != 'Normal']
df = df[df['patient'] != 74170962].reset_index(drop= True)
annotation = "/home/seob/class_project/School_project/from_shin/Bone_tumor_20220530/via-2.0.11/Bone_Tumor/Bone_tumor_20220530.json"
with open(annotation) as json_file:
    img_labels = json.load(json_file)
new_dict = {}
for key in img_labels['_via_img_metadata'].keys():
    new_key = key.split('.png')[0] + '.png'
    new_dict[new_key] = img_labels['_via_img_metadata'][key]

image_list = []
for data in new_dict.keys():
    if len(new_dict[data]['regions']) != 0:
        image_list.append(data)
df = df[df['image'].isin(image_list)].reset_index(drop = True)

with open("/home/seob/class_project/School_project/dataset/age_dict.pickle", 'rb') as f:
    data = pickle.load(f)

for patient in data.keys():
    for i in df[df['patient'] == patient].index:
        df.at[i, 'Age'] = data[patient]
'''
df = pd.read_csv("/home/seob/class_project/School_project/dataset/final_dataset.csv")
#fold_dataset(df)
multiview_dataset(df)