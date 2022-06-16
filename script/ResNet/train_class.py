import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import os
import shutil

def final_evaluation(model, data_loader, save_dir, num_class, cuda):
    model.eval()
    mode = 'test'
    total_correct = 0
    total = 0
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))

    label_list = []
    predict_list = []
    image_path_list = []
    correct_path_list = []

    final_array = []
    output_array = []
    softmax = nn.Softmax(dim=1)
    for i, data in enumerate(data_loader[mode]):
        with torch.no_grad():
            inputs, target = data
            inputs = inputs.to(device)
            labels = target["type"].to(device)
            paths = target['img_path']
            outputs = model(inputs)
            if len(output_array) != 0:
                output_array = np.concatenate((output_array, softmax(outputs).detach().cpu().numpy()), axis=0)
                final_array = np.concatenate((final_array, torch.max(softmax(outputs), 1)[0].detach().cpu().numpy()), axis=0)
            else:
                output_array = softmax(outputs).detach().cpu().numpy()
                final_array = torch.max(softmax(outputs), 1)[0].detach().cpu().numpy()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total += len(predicted)
            acc = correct / len(predicted)

            correct_path = np.array(paths)[np.where((predicted == labels).detach().cpu().numpy())[0]]

            image_path_list += paths
            correct_path_list += list(correct_path)
            label_list += labels.detach().cpu().numpy().tolist()
            predict_list += predicted.detach().cpu().numpy().tolist()

    non_correct_path_list = [path for path in image_path_list if path not in correct_path_list]
    correct_file_dir = os.path.join(save_dir, 'correct_image')
    if os.path.exists(correct_file_dir) is False:
        os.mkdir(correct_file_dir)

    non_correct_file_dir = os.path.join(save_dir, 'non_correct_image')
    if os.path.exists(non_correct_file_dir) is False:
        os.mkdir(non_correct_file_dir)

    for image in correct_path_list:
        shutil.copy(image, os.path.join(correct_file_dir, image.split('/')[-1]))
    for image in non_correct_path_list:
        shutil.copy(image, os.path.join(non_correct_file_dir, image.split('/')[-1]))

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
                 label='ROC curve of class {0} (area = {1: 0.2f})'.format(i,roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Bone tumor classification roc curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.clf()

    print(total_correct/total)
    return total_correct/ total



def calculate_metric(epoch, save_dir, label_list, predict_list):

    cnf_matrix = confusion_matrix(label_list, predict_list)
    cnf_df = pd.DataFrame(cnf_matrix)
    cnf_df.to_csv(os.path.join(save_dir, str(epoch) + '_confusion_matrix.csv'))

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
    metric_df.to_csv(os.path.join(save_dir, str(epoch) + '_metric.csv'))


def train_class(model, data_loader, epoch_num, lr, weight_decay, save_dir, num_class, cuda):

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr, weight_decay= weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    phase_list = ['train', 'test']
    for epoch in range(epoch_num):
        for mode in phase_list:
            if mode == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            total_correct = 0
            total = 0
            label_list = []
            predict_list = []
            with tqdm(total=len(data_loader[mode])) as pbar:
                for i, data in enumerate(data_loader[mode]):
                    inputs, target = data
                    inputs = inputs.to(device)
                    labels = target["type"].to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    total_correct += correct
                    acc = correct / len(predicted)

                    total += labels.size(0)
                    loss = criterion(outputs, labels)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        label_list += labels.detach().cpu().numpy().tolist()
                        predict_list += predicted.detach().cpu().numpy().tolist()

                    pbar.set_description(
                        "Loss: %0.5f, acc:%0.5f, lr: %0.8f " % (loss, acc, optimizer.param_groups[0]['lr']))
                    running_loss += loss.item()
                    pbar.update()

            scheduler.step()

            if mode == "train":
                print(epoch)
                epoch_loss = running_loss / len(data_loader[mode])
                epoch_acc = total_correct / total
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
                print("train loss: " + str(epoch_loss) + " train_acc:" + str(epoch_acc))

            if mode == "test":
                print(epoch)
                epoch_loss = running_loss / len(data_loader[mode])
                epoch_acc = total_correct / total
                test_loss_list.append(epoch_loss)
                test_acc_list.append(epoch_acc)
                print("test loss: " + str(epoch_loss) + " test_acc:" + str(epoch_acc))
                if epoch % 5 == 0 and epoch != 0:
                    calculate_metric(epoch, save_dir, label_list, predict_list)
                    torch.save(model.state_dict(), os.path.join(save_dir, str(epoch) + '_cnn_model.pt'))
                if epoch == epoch_num-1:
                    calculate_metric(epoch, save_dir, label_list, predict_list)
                    torch.save(model.state_dict(), os.path.join(save_dir, str(epoch) + '_cnn_model.pt'))

    plt.plot(range(epoch_num), train_loss_list, label = 'train_loss')
    plt.plot(range(epoch_num), test_loss_list, label = 'test_loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.clf()

    plt.plot(range(epoch_num), train_acc_list, label='train_acc')
    plt.plot(range(epoch_num), test_acc_list, label='test_acc')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'acc_plot.png'))
    plt.clf()

    fold_acc = final_evaluation(model, data_loader, save_dir, num_class = num_class, cuda = cuda)

    return fold_acc
