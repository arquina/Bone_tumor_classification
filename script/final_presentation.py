import cv2
import pandas as pd
import os, random, torch, sys
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

#from dataloader import xray_dataset_class_with_mask
import json
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from mask_r_cnn_model import mask_model, train_mask, mask_model_custom, multitask_predictor
from ResNet import classification_model
from sklearn.metrics import roc_curve, auc, confusion_matrix
from itertools import cycle
import seaborn as sn

def calculate_metric(save_dir, label_list, predict_list, save_name):

    cnf_matrix = confusion_matrix(label_list, predict_list)
    cnf_df = pd.DataFrame(cnf_matrix)
    plt.figure(figsize = (5,3.5))
    sn.heatmap(cnf_df, annot = True)
    plt.savefig(os.path.join(save_dir, save_name + '_confusion_matrix.png'))
    plt.clf()
    cnf_df.to_csv(os.path.join(save_dir, save_name  + '_confusion_matrix.csv'))

    metric_dic = {}
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    metric_dic['TPR'] = TP / (TP + FN)
    # Specificity or true negative rate
    metric_dic['TNR'] = TN / (TN + FP)
    # Precision or positive predictive value
    metric_dic['PPV'] = TP / (TP + FP)
    # Negative predictive value
    metric_dic['NPV'] = TN / (TN + FN)
    # Fall out or false positive rate
    metric_dic['FPR'] = FP / (FP + TN)
    # False negative rate
    metric_dic['FNR'] = FN / (TP + FN)
    # False discovery rate
    metric_dic['FDR'] = FP / (TP + FP)
    # Overall accuracy for each class
    metric_dic['ACC'] = (TP + TN) / (TP + FP + FN + TN)

    metric_df = pd.DataFrame.from_dict(metric_dic)
    metric_df.to_csv(os.path.join(save_dir, save_name + '_metric.csv'))

def final_evaluation(model, data_loader, save_dir, save_name, num_class, cuda):
    model.eval()
    mode = 'test'
    total_correct = 0
    total = 0
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))
    model.to(device)
    label_list = []
    predict_list = []
    image_path_list = []
    correct_path_list = []

    final_array = []
    output_array = []
    softmax = nn.Softmax(dim=1)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = data
            inputs = inputs.to(device)
            labels = target["type"].to(device)
#            paths = target['img_path']

            outputs = model(inputs)
            if len(output_array) != 0:
                output_array = np.concatenate((output_array, softmax(outputs).detach().cpu().numpy()), axis=0)
                final_array = np.concatenate((final_array, torch.max(softmax(outputs), 1)[0].detach().cpu().numpy()),
                                             axis=0)
            else:
                output_array = softmax(outputs).detach().cpu().numpy()
                final_array = torch.max(softmax(outputs), 1)[0].detach().cpu().numpy()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total += len(predicted)
            acc = correct / len(predicted)

            #correct_path = np.array(paths)[np.where((predicted == labels).detach().cpu().numpy())[0]]

            #image_path_list += paths
            #correct_path_list += list(correct_path)
            label_list += labels.detach().cpu().numpy().tolist()
            predict_list += predicted.detach().cpu().numpy().tolist()
    calculate_metric(save_dir, label_list=label_list,
                     predict_list=predict_list, save_name=save_name)

    class_label = {0: 'Benign', 1: 'Malignant', 2: 'Intermediate'}
    label_torch = torch.Tensor(label_list)
    label_torch = nn.functional.one_hot(label_torch.to(torch.int64), num_classes= num_class)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(np.array(label_torch)[:, i], output_array[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    lw = 2
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1: 0.2f})'.format(class_label[i],roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Bone tumor classification roc curve_' + save_name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, save_name + '_roc_curve.png'))
    plt.clf()

    return total_correct/ total

def collate_fn(batch):
    return tuple(zip(*batch))

class xray_dataset_mask(Dataset):
    def __init__(self, root_dir, df_dir, annotation, phase, size):
        self.root_dir = os.path.join(root_dir, 'total_dataset')
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

class xray_dataset_class(Dataset):
    def __init__(self, root_dir, phase, fold_dir, total_fold, current_fold):
        self.root_dir = os.path.join(root_dir, 'total_dataset')
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

def result(model, test_dataloader, save_dir, save_name, device):

    total_correct = 0
    total_correct_cur = 0
    total = 0
    label_list = []
    predict_list = []
    final_array = []
    model.to(device)
    correct_list = []
    prediction_list = []
    for i, data in enumerate(test_dataloader):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = torch.Tensor([t['labels'].item() for t in targets])
        model.eval()
        with torch.no_grad():
            prediction, _ = model(images)
            output_array = np.zeros((len(prediction), 4))
            for index, predict in enumerate(prediction):
                label = predict['labels']
                score = predict['scores']
                if len(label) != 0:
                    output_array[index, label[0]] = score[0]
                    if label[0] == 1:
                        output_array[index, 2] = (1 - score[0]) / 2
                        output_array[index, 3] = (1 - score[0]) / 2
                    if label[0] == 2:
                        output_array[index, 1] = (1 - score[0]) / 2
                        output_array[index, 3] = (1 - score[0]) / 2
                    if label[0] == 3:
                        output_array[index, 2] = (1 - score[0]) / 2
                        output_array[index, 1] = (1 - score[0]) / 2
            if len(final_array) != 0:
                final_array = np.concatenate((final_array, output_array), axis=0)
            else:
                final_array = output_array

            predict_label = torch.Tensor(
                [t['labels'][0] if len(t['labels']) != 0 else torch.tensor(0) for t in prediction])
            correct = (predict_label == targets).sum().item()
            total_correct += correct
            total += len(predict_label)
            label_list += targets.detach().cpu().numpy().tolist()
            predict_list += predict_label.detach().cpu().numpy().tolist()

    print(total_correct / total)

    calculate_metric(save_dir, label_list=label_list,
                     predict_list=predict_list, save_name= save_name)

    label_torch = torch.Tensor(label_list)
    label_torch = nn.functional.one_hot(label_torch.to(torch.int64), num_classes=4)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1, 4):
        fpr[i], tpr[i], _ = roc_curve(np.array(label_torch)[:, i], final_array[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    class_label = {1: 'Benign', 2: 'Malignant', 3: 'Intermediate'}
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    lw = 2
    for i, color in zip(range(1, 4), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1: 0.2f})'.format(class_label[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Bone tumor classification roc curve_' + save_name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, save_name + '_roc_curve.png'))
    plt.clf()

root_dir = "/home/seob/class_project/School_project/dataset/"
json_file = "/home/seob/class_project/School_project/from_shin/Bone_tumor_20220530/via-2.0.11/Bone_Tumor/Bone_tumor_20220530.json"
df_dir = "/home/seob/class_project/School_project/dataset/final_5_fold_dataset/"
batch_size = 16
save_dir = "/home/seob/class_project/School_project/result/final_for_paper/"
save_name = 'final'
multiview = False
external_dir = "/home/seob/class_project/School_project/from_shin/external_png"
external_data = "/home/seob/class_project/School_project/from_shin/external_validation_set1/external validation set1.xlsx"
size = 'big'

class_test_dataset = xray_dataset_class(root_dir, 'test', df_dir, 5, 0)
mask_test_dataset = xray_dataset_mask(root_dir, df_dir, json_file, 'test', size = size)
class_external_dataset = xray_dataset_external('class', multiview, external_dir, external_data)
mask_external_dataset = xray_dataset_external('mask', multiview, external_dir, external_data)

class_test_dataloader = DataLoader(class_test_dataset, batch_size = batch_size)
mask_test_dataloader = DataLoader(mask_test_dataset, batch_size = batch_size, collate_fn = collate_fn)
class_external_dataloader = DataLoader(class_external_dataset, batch_size = batch_size)
mask_external_dataloader = DataLoader(mask_external_dataset, batch_size = batch_size, collate_fn = collate_fn)

'''
class_model = classification_model(pretrained = False, num_classes = 3)
class_final_model = "/home/seob/class_project/School_project/result/final_for_paper/60_cnn_model.pt"
class_model.load_state_dict(torch.load(class_final_model))
'''

model = mask_model_custom(pretrained = False, num_classes = 4)
save_model = "/home/seob/class_project/School_project/result/mask_result/2022-06-12_17:55:25_epoch_100_classnum_3_big_modality_subtype/90/mask_r_cnn_model.pt"
model.load_state_dict(torch.load(save_model, map_location = 'cuda:1'))
device = torch.device('cuda:1')
multitask_model = []


#_ = final_evaluation(class_model, class_external_dataloader, save_dir, num_class = 3, cuda = 'cuda:1', save_name = 'ResNet')
# result(model, mask_test_dataloader, save_dir, 'Mask_R_CNN_subtype', device)
result(model, mask_external_dataloader, save_dir, 'Mask_R_CNN_subtype_modality', device)
