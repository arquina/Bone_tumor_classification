import argparse

import torch.utils.data
from torch.utils.data import DataLoader
from time import strftime, localtime, time
import os
from .dataloader import xray_dataset_class, xray_dataset_mask, collate_fn, xray_dataset_external, xray_dataset_mask_multiview
from .ResNet import classification_model, train_class, final_evaluation
import pandas as pd


def Parser_main():
    parser = argparse.ArgumentParser(description="Class_project_Xray_classification")
    parser.add_argument("--rootdir" , default = "/home/seob/class_project/School_project/dataset/" , help="dataset rootdir", type = str)
    parser.add_argument("--df_dir", default = "/home/seob/class_project/School_project/dataset/final_5_fold_dataset/", help = "directory which have dataset_csv per fold", type = str)
    parser.add_argument("--json_file", default = "/home/seob/class_project/School_project/dataset/Bone_tumor_20220530.json", help = "dataset metadata", type = str)
    parser.add_argument("--pretrained", action= "store_true", help= "using the pretrained back bone model")
    parser.add_argument("--batch_size", default=8, help="train, test batch_size", type=int)
    parser.add_argument("--epoch", default= 20, help="epoch number", type=int)
    parser.add_argument("--task", default='mask', help="class/mask", type=str)
    parser.add_argument("--lr", default = 0.0001, help = "Learning rate", type = float)
    parser.add_argument("--weight_decay", default = 0, help = "Weight decay", type = float)
    parser.add_argument("--fold", default = 5, type = int, help = 'fold number')
    parser.add_argument("--classnum", default = 3, type = int, help = 'classification type number')
    parser.add_argument("--custom", action = "store_true", help = "torchvision model or customized model")
    parser.add_argument("--cuda", default = 'cuda:1', type = str, help = 'cuda device or cpu')
    parser.add_argument("--checkpoint", default = None , type = str, help = 'existing trained model')
    parser.add_argument("--subtype_checkpoint", default=None, type=str, help = 'subtype model checkpoint')
    parser.add_argument("--modality_checkpoint", default=None, type=str, help = 'modality model checkpoint')
    parser.add_argument("--fracture_checkpoint", default=None, type=str, help = 'fracture model checkpoint')
    parser.add_argument("--multiview_checkpoint", default = None, type = str, help = 'multiview model checkpoint')
    parser.add_argument("--size", default = 'big', type = str, help = 'size of the dataset')
    parser.add_argument("--subtype", action= "store_true", help = 'subtype multitask model')
    parser.add_argument("--modality", action= "store_true", help = 'modality multitask model')
    parser.add_argument("--fracture", action= "store_true", help = 'fracture multitask model')
    parser.add_argument("--multiview", action = "store_true", help = 'multiview model')
    parser.add_argument("--result", default = "/home/seob/class_project/School_project/result/mask_result/", help = "result save directory", type = str)
    parser.add_argument("--external", action = "store_true", help = "test with external_dataset")
    parser.add_argument("--external_dir", default = "/home/seob/class_project/School_project/dataset/external_png/", help = "external_data_image_dir", type = str)
    parser.add_argument("--external_data", default = "/home/seob/class_project/School_project/dataset/external validation set1.xlsx", help = "external_metadata", type = str)
    return parser.parse_args()

def main():
    tm = localtime(time())
    cur_time = strftime('%Y-%m-%d_%H:%M:%S', tm)
    Argument = Parser_main()
    root_dir = os.path.join(Argument.rootdir, 'total_dataset')
    if Argument.multiview:
        print('MultiView model')
        from .mask_r_cnn_model_multiview import train_mask, mask_model_custom, multitask_predictor, multiview_predictor, external_evaluation
    else:
        from .mask_r_cnn_model import train_mask, mask_model_custom, multitask_predictor, external_evaluation

    if Argument.task == 'class':
        cur_time += '_epoch_' + str(Argument.epoch) + '_lr_' + str(Argument.lr) + '_decay_' + str(Argument.weight_decay) + '_'+str(Argument.classnum)
        result_dir = Argument.result
        result_dir = os.path.join(result_dir, cur_time)
        if Argument.external:
            print("Load external dataset")
            test_dataset = xray_dataset_external(Argument.task, Argument.multiview, Argument.external_dir, Argument.external_data)
            print("Complete dataset load")

            print("Load Dataloader")
            test_dataloader = DataLoader(test_dataset, batch_size = Argument.batch_size, shuffle = True)
            dataloader = {'test': test_dataloader}
            print("Data loader complete")

            print("Classification model")
            model = classification_model(pretrained=Argument.pretrained, num_classes=Argument.classnum)
            if Argument.checkpoint is not None:
                print("Load model")
                model.load_state_dict(torch.load(Argument.checkpoint))
                print("Model load complete")
            else:
                print("Error")
                return 0

            print("Test for external dataset")
            final_evaluation(model, dataloader, result_dir, Argument.classnum, Argument.cuda)

        else:
            if os.path.exists(result_dir) is False:
                os.mkdir(result_dir)
            #### For train the model with best parameter
            if Argument.fold == 0:
                print("Load classification dataset")
                train_dataset = xray_dataset_class(root_dir, 'train', Argument.df_dir, 5, 0)
                test_dataset = xray_dataset_class(root_dir, 'test', Argument.df_dir, 5, 0)

                print("Load Dataloader")
                train_dataloader = DataLoader(train_dataset, batch_size=Argument.batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=Argument.batch_size, shuffle=True)
                dataloader = {'train': train_dataloader, 'test': test_dataloader}
                print("Data loader complete")

                print("Classification model")
                model = classification_model(pretrained=Argument.pretrained, num_classes=Argument.classnum)
                if Argument.checkpoint is not None:
                    print("Load model")
                    model.load_state_dict(torch.load(Argument.checkpoint))
                    print("Model load complete")
                print("Model call complete")

                print("train start!")
                train_class(model, dataloader, Argument.epoch, Argument.lr, Argument.weight_decay, result_dir, Argument.classnum, Argument.cuda)
                print("Train Finish!")

            #### For 5 fold cross validation
            else:
                results = {}
                print(str(Argument.fold) + ' fold validation')
                for fold in range(Argument.fold):
                    print('Load classification dataset')
                    train_dataset = xray_dataset_class(root_dir, 'train',  Argument.df_dir, Argument.fold, fold)
                    test_dataset = xray_dataset_class(root_dir, 'test', Argument.df_dir, Argument.fold, fold)
                    print(str(fold) + '_fold')
                    save_dir = os.path.join(result_dir, str(fold))
                    if os.path.exists(save_dir) is False:
                        os.mkdir(save_dir)

                    print('Create Dataloader')
                    train_loader = DataLoader(train_dataset, batch_size = Argument.batch_size)
                    test_loader = DataLoader(test_dataset, batch_size = Argument.batch_size)
                    dataloader = {'train': train_loader, 'test': test_loader}
                    print("Data loader complete")

                    print("Classification model")
                    model = classification_model(pretrained = Argument.pretrained, num_classes =Argument.classnum)
                    if Argument.checkpoint is not None:
                        print("Load model")
                        model.load_state_dict(torch.load(Argument.checkpoint))
                        print("Model load complete")
                    print("Model call complete")

                    print("train start!")
                    results[fold] = train_class(model, dataloader, Argument.epoch, Argument.lr, Argument.weight_decay, save_dir, Argument.classnum, Argument.cuda)
                    print("Train Finish!")

                sum = 0.0
                for key, value in results.items():
                    print(f'Fold {key}: {value} %')
                    sum += value
                print(f'Average: {sum*100 / len(results.items()) } %')

                df = pd.DataFrame(results.values())
                df.to_csv(os.path.join(result_dir, '5_fold_cross_validation.csv'))

    elif Argument.task == 'mask':
        cur_time += '_epoch_' + str(Argument.epoch) +'_classnum_' + str(Argument.classnum) + '_' + Argument.size
        result_dir = Argument.result
        result_dir = os.path.join(result_dir, cur_time)
        class_num = Argument.classnum + 1

        if Argument.external:
            print("external validation")
            result_dir += '_external'
            if os.path.exists(result_dir) is False:
                os.mkdir(result_dir)

            print("Load external dataset")
            test_dataset = xray_dataset_external(Argument.task, Argument.multiview, Argument.external_dir, Argument.external_data)
            print("Complete dataset load")

            print("Load Dataloader")
            test_dataloader = DataLoader(test_dataset, batch_size=Argument.batch_size, shuffle=True, collate_fn = collate_fn)
            print("Data loader complete")

            print("Mask RCNN model")
            model = mask_model_custom(pretrained=Argument.pretrained, num_classes=class_num)

            multiview_model = None
            if Argument.multiview:
                multiview_model = multiview_predictor(num_classes=3)
                multiview_model.state_dict(torch.load(Argument.multiview_checkpoint))
            if Argument.checkpoint is not None:
                print('Load latest trained model')
                model.load_state_dict(torch.load(Argument.checkpoint))

            external_evaluation(model, multiview_model, test_dataloader, Argument.cuda)


        else:
            if os.path.exists(result_dir) is False:
                os.mkdir(result_dir)

            if Argument.multiview:
                print('Load multiview train dataset')
                train_dataset = xray_dataset_mask_multiview(root_dir, Argument.df_dir, Argument.json_file, 'train')
                print('Load multiview test dataset')
                test_dataset = xray_dataset_mask_multiview(root_dir, Argument.df_dir, Argument.json_file, 'test')
            else:
                print('Load mask train dataset')
                train_dataset = xray_dataset_mask(root_dir, Argument.df_dir, Argument.json_file, 'train', size = Argument.size)
                print('Load mask test dataset')
                test_dataset = xray_dataset_mask(root_dir, Argument.df_dir, Argument.json_file, 'test', size = Argument.size)


            print('Create Dataloader')
            train_dataloader = DataLoader(train_dataset, batch_size=Argument.batch_size, shuffle=True, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=Argument.batch_size, shuffle=True, collate_fn=collate_fn)
            dataloader = {'train': train_dataloader, 'test': test_dataloader}

            print("Mask RCNN model")
            model = mask_model_custom(pretrained=Argument.pretrained, num_classes=class_num)


            if Argument.checkpoint is not None:
                print('Load latest trained model')
                model.load_state_dict(torch.load(Argument.checkpoint))
            else:
                if Argument.multiview:
                    print('Multiview learning need pretrained model')
                    return 0

            multitask_model = {}
            if Argument.subtype or Argument.modality or Argument.fracture:
                print("Multitask model")
                multitask_model = {}
                if Argument.subtype:
                    print('subtype')
                    multitask_model['subtype'] = multitask_predictor(num_classes = 15)
                    if Argument.subtype_checkpoint is not None:
                        print('latest_subtype')
                        multitask_model['subtype'].load_state_dict(torch.load(Argument.subtype_checkpoint))
                if Argument.fracture:
                    print('fracture')
                    multitask_model['fracture'] = multitask_predictor(num_classes = 3)
                    if Argument.fracture_checkpoint is not None:
                        print('latest_fracture')
                        multitask_model['fracture'].load_state_dict(torch.load(Argument.fracture_checkpoint))
                if Argument.modality:
                    print('modality')
                    multitask_model['modality'] = multitask_predictor(num_classes = 15)
                    if Argument.modality_checkpoint is not None:
                        print('latest_modality')
                        multitask_model['modality'].load_state_dict(torch.load(Argument.modality_checkpoint))

            multiview_model = None
            if Argument.multiview:
                print("multiview model")
                multiview_model = multiview_predictor(num_classes=3)

            print("train")
            train_mask(model, multitask_model, multiview_model, Argument.epoch, dataloader, result_dir, Argument.cuda, Argument.custom)
            print('Train Finish')
    else:
        print("Wrong task")

if __name__ == "__main__":
    main()
