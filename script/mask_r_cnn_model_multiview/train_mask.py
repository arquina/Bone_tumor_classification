import torch
from .engine import train_one_epoch, evaluate
from PIL import ImageDraw, Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

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

def external_evaluation(model, multiview_model, data_loader, cuda):

    total_correct = 0
    total = 0
    total_correct_score = 0
    total_score = 0
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    model.to(device)
    multiview_model.to(device)

    for i, data in enumerate(data_loader):
        images, targets = data
        images_1 = list(image1.to(device) for image1, _ in images)
        images_2 = list(image2.to(device) for _, image2 in images)
        targets_1 = [{k: v.to(device) for k, v in t1.items()} for t1, _ in targets]
        targets_2 = [{k: v.to(device) for k, v in t2.items()} for _, t2 in targets]
        #image_list += list(image for image in images)
        #targets = [{k: v.to(device) for k, v in t.items() if k != 'original'} for t in targets]
        #mask_list += list(target['masks'] for target in targets)
        #original_image_list += original_images
        #target_list += targets
        for image_1, image_2, target_1, target_2 in zip(images_1, images_2, targets_1, targets_2):
            model.eval()
            with torch.no_grad():
                prediction_1, _, selected_feature_1 = model([image_1])
                prediction_2, _, selected_feature_2 = model([image_2])

                if len(prediction_1[0]['scores']) != 0 and len(prediction_2[0]['scores']) != 0:
                    score_1 = prediction_1[0]['scores'][0].item()
                    score_2 = prediction_2[0]['scores'][0].item()
                    label1 = prediction_1[0]['labels'][0]
                    label2 = prediction_2[0]['labels'][0]
                    if score_1 > score_2:
                        predicted_label = label1
                    else:
                        predicted_label = label2
                    target_label = target_1['labels']
                    if target_label.item() == predicted_label:
                        total_correct_score += 1
                total_score += 1

                selected_features = torch.cat([selected_feature_1, selected_feature_2], dim=0)
                output = multiview_model(selected_features.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == (target_1['labels']-1)).sum().item()
                total_correct += correct
                total += 1
    print(total_correct/total)
    print(total_correct_score/total_score)

def self_evaluation(data_loader, device, model, multiview_model):

    predict_list = []
    correct_list = []
    original_image_list = []

    image_list = []
    mask_list = []
    prediction_list = []
    target_list = []
    total_correct = 0
    total = 0
    for i, data in enumerate(data_loader['test']):
        images, targets, _ = data
        images_1 = list(image1.to(device) for image1, _ in images)
        images_2 = list(image2.to(device) for _, image2 in images)
        targets_1 = [{k: v.to(device) for k, v in t1.items()} for t1, _ in targets]
        targets_2 = [{k: v.to(device) for k, v in t2.items()} for _, t2 in targets]
        #image_list += list(image for image in images)
        #targets = [{k: v.to(device) for k, v in t.items() if k != 'original'} for t in targets]
        #mask_list += list(target['masks'] for target in targets)
        #original_image_list += original_images
        #target_list += targets
        for image_1, image_2, target_1, target_2 in zip(images_1, images_2, targets_1, targets_2):
            model.eval()
            with torch.no_grad():
                prediction_1, _, selected_feature_1 = model([image_1])
                prediction_2, _, selected_feature_2 = model([image_2])

                selected_features = torch.cat([selected_feature_1, selected_feature_2], dim=0)
                output = multiview_model(selected_features.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == (target_1['labels']-1)).sum().item()
                total_correct += correct
                total += 1
    print(total_correct/total)


def train_mask(model, multitask_model, multiview_model, epoch_num, data_loader, save_dir, cuda, custom):

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))

    #model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)
    if len(multitask_model) is not 0:
        for key in multitask_model.keys():
            multitask_model[key].to(device)
    if multiview_model is not None:
        multiview_model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.99)

    # let's train it for 10 epochs

    if epoch_num == 0:
        self_evaluation(data_loader, device, model, multiview_model)

    for epoch in range(epoch_num):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, multitask_model, multiview_model, optimizer, data_loader['train'], device, epoch, custom, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, multitask_model, multiview_model, data_loader['test'], custom, device=device)
        if epoch != 0 and epoch % 10 == 0:
            final_dir = os.path.join(save_dir, str(epoch))
            if os.path.exists(final_dir) is False:
                os.mkdir(final_dir)
            self_evaluation(data_loader, device, model,multiview_model)
            torch.save(model.state_dict(),
                       os.path.join(final_dir, 'mask_r_cnn_model.pt'))
            if len(multitask_model) is not 0:
                for key in multitask_model.keys():
                    torch.save(multitask_model[key].state_dict(),
                               os.path.join(final_dir, key + '_model.pt'))
            if multiview_model is not None:
                torch.save(multiview_model.state_dict(),
                           os.path.join(final_dir, 'multiview_model.pt'))

    self_evaluation(data_loader, device, model, multiview_model)
    torch.save(model.state_dict(),
               os.path.join(save_dir, 'mask_r_cnn_model.pt'))
    if len(multitask_model) is not 0:
        for key in multitask_model.keys():
            torch.save(multitask_model[key].state_dict(),
                       os.path.join(save_dir, key + '_model.pt'))
    if multiview_model is not None:
        torch.save(multiview_model.state_dict(),
                   os.path.join(save_dir, 'multiview_model.pt'))
    print("That's it!")
