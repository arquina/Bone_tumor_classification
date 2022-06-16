import argparse

import torch.utils.data
from torch.utils.data import DataLoader
from time import strftime, localtime, time
import os
from dataloader import xray_dataset_class, xray_dataset_mask, collate_fn, xray_dataset_mask_multiview, xray_dataset_mask_external
from ResNet import classification_model, train_class
import pickle
import pandas as pd


def Parser_main():
    parser = argparse.ArgumentParser(description="Class_project_Xray_classification")
    parser.add_argument("--rootdir" , default = "/home/seob/class_project/School_project/dataset/" , help="dataset rootdir", type = str)
    parser.add_argument("--json_file", default = "/home/seob/class_project/School_project/from_shin/Bone_tumor_20220522/via-2.0.11/Bone_Tumor/Bone_tumor_20220522.json", help = "dataset metadata", type = str)
    parser.add_argument("--pretrained", action= "store_true")
    parser.add_argument("--batch_size", default=16, help="train, test batch_size", type=int)
    parser.add_argument("--epoch", default=0, help="epoch number", type=int)
    parser.add_argument("--task", default='mask', help="Classification type", type=str)
    parser.add_argument("--lr", default = 0.001, help = "Learning rate", type = float)
    parser.add_argument("--weight_decay", default = 0, help = "Weight decay", type = float)
    parser.add_argument("--name", default='Resnet_realdata_medium_with_Weight_decay', type = str)
    parser.add_argument("--fold", default = 5, type = int)
    parser.add_argument("--model", default = 'resnet', type = str)
    parser.add_argument("--classnum", default = 4, type = int)
    parser.add_argument("--cuda", default = 'cpu', type = str)
    parser.add_argument("--custom", action= "store_true")
    parser.add_argument("--checkpoint", default = None , type = str)
    parser.add_argument("--subtype_checkpoint", default=None, type=str)
    parser.add_argument("--modality_checkpoint", default=None, type=str)
    parser.add_argument("--fracture_checkpoint", default=None, type=str)
    parser.add_argument("--subtype", action= "store_true")
    parser.add_argument("--modality", action= "store_true")
    parser.add_argument("--fracture", action= "store_true")
    parser.add_argument("--multiview", action = "store_true")
    return parser.parse_args()

def main():
    tm = localtime(time())
    cur_time = strftime('%Y-%m-%d_%H:%M:%S', tm)

    Argument = Parser_main()
    Argument.custom = True
    Argument.multiview = True
    Argument.checkpoint = "/home/seob/class_project/School_project/result/mask_result/multiview_best_60/60/mask_r_cnn_model.pt"
    if Argument.multiview:
        print('MultiView model')
        from mask_r_cnn_model_multiview import mask_model, train_mask, mask_model_custom, multitask_predictor, multiview_predictor
    else:
        from mask_r_cnn_model import mask_model, train_mask, mask_model_custom, multitask_predictor

    if Argument.task == 'class':
        cur_time += '_epoch_' + str(Argument.epoch) + '_lr_' + str(Argument.lr) + '_decay_' + str(Argument.weight_decay) + '_'+str(Argument.classnum) + '_' + Argument.model
        root_dir = os.path.join(Argument.rootdir, 'total_dataset')
        result_dir = "/home/seob/class_project/School_project/result/resnet_result/"
        fold_dir = "/home/seob/class_project/School_project/dataset/classification/4_class/"
        result_dir = os.path.join(result_dir, cur_time)

        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

        print('Load classification dataset')
        dataset = xray_dataset_class(root_dir, Argument.json_file, 'train', Argument.task, Argument.classnum)

        print(str(Argument.fold) + ' fold validation')
        fold_dict = {}
        total_fold = []
        for index in range(Argument.fold):
            with open(os.path.join(fold_dir, 'fold' + str(index)), 'rb') as f:
                fold_dict[index] = pickle.load(f)
            total_fold += fold_dict[index]
        results = {}

        for fold  in range(Argument.fold):
            test_ids = fold_dict[fold]
            train_ids = [index for index in total_fold if index not in test_ids]

            print(str(fold) + '_fold')
            save_dir = os.path.join(result_dir, str(fold))
            if os.path.exists(save_dir) is False:
                os.mkdir(save_dir)

            print("Data subsample")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            print('Create Dataloader')
            train_loader = DataLoader(dataset, batch_size = Argument.batch_size, sampler = train_subsampler)
            test_loader = DataLoader(dataset, batch_size = Argument.batch_size, sampler = test_subsampler)
            dataloader = {'train': train_loader, 'test': test_loader}
            print("Data loader complete")

            print("Classification model")
            print("Load model")
            if Argument.classnum == 'normal':
                model = classification_model(pretrained = Argument.pretrained, num_classes = 2, model_name = Argument.model)
                num_class = 2
            elif Argument.classnum == '3class':
                model = classification_model(pretrained = Argument.pretrained, num_classes =3, model_name = Argument.model)
                num_class = 3
            elif Argument.classnum == '4class':
                model = classification_model(pretrained = Argument.pretrained, num_classes = 4, model_name = Argument.model)
                num_class = 4
            print("Model load complete")

            print("train start!")
            results[fold] = train_class(model, dataloader, Argument.epoch, Argument.lr, Argument.weight_decay, save_dir, num_class, Argument.cuda)
            print("Train Finish!")

        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum*100 / len(results.items()) } %')

        df = pd.DataFrame(results.values())
        df.to_csv(os.path.join(result_dir, '5_fold_cross_validation.csv'))

    elif Argument.task == 'mask':
        cur_time += '_epoch_' + str(Argument.epoch) +'_classnum_' + str(Argument.classnum)
        result_dir = "/home/seob/class_project/School_project/result/mask_result"
        result_dir = os.path.join(result_dir, cur_time)
        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

        if Argument.multiview:
            print('Load multiview train dataset')
            train_dataset = xray_dataset_mask_multiview(Argument.rootdir, Argument.json_file, 'train', Argument.classnum)
        else:
            print('Load mask train dataset')
            train_dataset = xray_dataset_mask(Argument.rootdir, Argument.json_file, 'train', Argument.classnum)
        data = train_dataset[0]

        if Argument.multiview:
            print('Load multiview test dataset')
            #test_dataset = xray_dataset_mask_multiview(Argument.rootdir, Argument.json_file, 'test', Argument.classnum)
            test_dataset = xray_dataset_mask_external(Argument.multiview)
        else:
            print('Load mask test dataset')
            test_dataset = xray_dataset_mask(Argument.rootdir, Argument.json_file, 'test', Argument.classnum)

        print('Create Dataloader')
        train_dataloader = DataLoader(train_dataset, batch_size=Argument.batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=Argument.batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader = {'train': train_dataloader, 'test': test_dataloader}

        if Argument.custom:
            print("Custom Mask RCNN model")
            model = mask_model_custom(pretrained=Argument.pretrained, num_classes=Argument.classnum)
        else:
            print("Public Mask RCNN model")
            model = mask_model(pretrained = Argument.pretrained, num_classes = Argument.classnum)

        if Argument.checkpoint is not None:
            print('Pretrained model')
            model.load_state_dict(torch.load(Argument.checkpoint))

        multitask_model = {}
        if Argument.subtype or Argument.modality or Argument.fracture:
            print("Multitask model")
            multitask_model = {}
            if Argument.subtype:
                print('subtype')
                multitask_model['subtype'] = multitask_predictor(num_classes = 14)
                if Argument.subtype_checkpoint is not None:
                    print('pretrained_subtype')
                    multitask_model['subtype'].load_state_dict(torch.load(Argument.subtype_checkpoint))
            if Argument.fracture:
                print('fracture')
                multitask_model['fracture'] = multitask_predictor(num_classes = 2)
                if Argument.fracture_checkpoint is not None:
                    print('pretrained_fracture')
                    multitask_model['fracture'].load_state_dict(torch.load(Argument.fracture_checkpoint))
            if Argument.modality:
                print('modality')
                multitask_model['modality'] = multitask_predictor(num_classes = 15)
                if Argument.modality_checkpoint is not None:
                    print('pretrained_modality')
                    multitask_model['modality'].load_state_dict(torch.load(Argument.modality_checkpoint))
        multiview_model = None
        if Argument.multiview:
            multiview_model = multiview_predictor(num_classes = 3)
            multiview_model.load_state_dict(torch.load("/home/seob/class_project/School_project/result/mask_result/multiview_best_60/60/multiview_model.pt"))

        print("train")
        train_mask(model, multitask_model, multiview_model, Argument.epoch, dataloader, result_dir, Argument.cuda, Argument.custom)
        print('Train Finish')

    else:
        print("Wrong task")

if __name__ == "__main__":
    main()