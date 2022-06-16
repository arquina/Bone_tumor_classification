import torch
from .engine import train_one_epoch, evaluate
from PIL import ImageDraw, Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def pil_draw_rect(image, original_mask,predicted_mask, original_point1, original_point2, predicted_point1, predicted_point2,
                  target_class, predicted_class, score, save_dir, image_id):
    original_image = image.copy()

    predicted_mask.putalpha(128)
    original_mask.putalpha(128)

    original_image.paste(im = original_mask, box = (0,0), mask = original_mask)
    image.paste(im=predicted_mask, box = (0,0), mask = predicted_mask)

    original_draw = ImageDraw.Draw(original_image)
    predicted_draw = ImageDraw.Draw(image)

    original_draw.rectangle((original_point1, original_point2), outline = (0,0,255), width = 3)
    predicted_draw.rectangle((predicted_point1, predicted_point2), outline=(0, 0, 255), width=3)

    #original_image = original_image.resize((1024, 1024))
    #image = image.resize((1024, 1024))

    fig = plt.figure(figsize=(40, 20))
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(original_image)
    ax1.set_title('original:' + target_class , fontsize = 50)
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(image)
    ax2.set_title('predicted:' + predicted_class + '(' + str(score*100) + '%)', fontsize = 50)
    ax2.axis("off")

    plt.savefig(os.path.join(save_dir, image_id))
    plt.clf()
    plt.close()
    plt.cla()

def external_evaluation(model, multiview_model, test_dataloader, cuda):

    total_correct = 0
    total = 0
    label_list = []
    predict_list = []
    final_array = []
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    model.to(device)

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
            acc = correct / len(predict_label)
            label_list += targets.detach().cpu().numpy().tolist()
            predict_list += predict_label.detach().cpu().numpy().tolist()

    print(total_correct / total)

def self_evaluation(data_loader, cuda, model, epoch, save_dir):

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    predict_list = []
    correct_list = []
    original_image_list = []

    image_list = []
    mask_list = []
    prediction_list = []
    target_list = []

    for i, data in enumerate(data_loader['test']):
        images, targets, original_images = data
        images = list(image.to(device) for image in images)
        image_list += list(image for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'original'} for t in targets]
        mask_list += list(target['masks'] for target in targets)
        original_image_list += original_images
        target_list += targets
        for image, target in zip(images, targets):
            model.eval()
            with torch.no_grad():
                prediction, _ = model([image])
            if prediction[0]['labels'].shape[0] != 0:
                predict_label = prediction[0]['labels'][torch.argmax(prediction[0]['scores'])]
                correct = target['labels']
                predict_list.append(predict_label.detach().cpu().numpy().item())
                correct_list.append(correct.detach().cpu().numpy().item())
                prediction_list.append(prediction)

    total_correct = 0
    for index, data in enumerate(predict_list):
        total_correct += predict_list[index] == correct_list[index]
    print(total_correct / len(correct_list))
    label_dict = {1: 'Benign', 3: 'Intermediate', 2: 'Malignant'}

    for c, prediction in enumerate(prediction_list):
        length = len(prediction[0]['scores'])
        if length > 2:
            length = 2
        for index in range(length):
            predicted_box = prediction[0]['boxes'][index].detach().cpu().numpy()
            target_box = target_list[c]['boxes'].detach().cpu().numpy()[0]
            predict_mask = prediction[0]['masks'][index]
            target_mask = target_list[c]['masks']
            target_class = target_list[c]['labels'][0].item()
            predict_class = prediction[0]['labels'][index].item()
            score = prediction[0]['scores'][index].item()
            pil_draw_rect(original_image_list[c].copy(),
                          Image.fromarray(target_mask[0].mul(255).byte().detach().cpu().numpy()).convert("RGBA").copy(),
                          Image.fromarray(predict_mask[0].mul(255).byte().detach().cpu().numpy()).convert(
                              "RGBA").copy(),
                          (target_box[0], target_box[1]), (target_box[2], target_box[3]),
                          (predicted_box[0], predicted_box[1]), (predicted_box[2], predicted_box[3]),
                          label_dict[target_class], label_dict[predict_class],
                          round(score, 3),
                          save_dir,
                          str(target_list[c]['image_id'].detach().cpu().numpy().item()) + '_' + str(index) + '.png')

    cnf_matrix = confusion_matrix(correct_list, predict_list)
    cnf_df = pd.DataFrame(cnf_matrix)
    cnf_df.to_csv(os.path.join(save_dir, str(epoch) + '_confusion_matrix.csv'))


def train_mask(model, multitask_model, multiview_model, epoch_num, data_loader, save_dir, cuda, custom):

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))
    model.to(device)
    if len(multitask_model) is not 0:
        for key in multitask_model.keys():
            multitask_model[key].to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.99)

    # let's train it for 10 epochs
    for epoch in range(epoch_num):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, multitask_model, optimizer, data_loader['train'], device, epoch, custom, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, multitask_model, data_loader['test'], custom, device=device)
        if epoch != 0 and epoch % 10 == 0:
            final_dir = os.path.join(save_dir, str(epoch))
            if os.path.exists(final_dir) is False:
                os.mkdir(final_dir)
            self_evaluation(data_loader, device, model, epoch, final_dir)
            torch.save(model.state_dict(),
                       os.path.join(final_dir, 'mask_r_cnn_model.pt'))
            if len(multitask_model) is not 0:
                for key in multitask_model.keys():
                    torch.save(multitask_model[key].state_dict(),
                               os.path.join(final_dir, key + '_model.pt'))

    self_evaluation(data_loader, cuda, model, epoch, save_dir)
    torch.save(model.state_dict(),
               os.path.join(save_dir, 'mask_r_cnn_model.pt'))

    if len(multitask_model) is not 0:
        for key in multitask_model.keys():
            torch.save(multitask_model[key].state_dict(),
                       os.path.join(save_dir, key + '_model.pt'))

    print("That's it!")
