import math
import sys
import time
import torch
from torch import nn

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .utils import warmup_lr_scheduler, SmoothedValue, MetricLogger, reduce_dict


def train_one_epoch(model,multitask_model, optimizer, data_loader, device, epoch,custom,print_freq):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict, features = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        if len(multitask_model) is not 0:
            multitask_loss_dict = {}
            for key in multitask_model.keys():
                output = multitask_model[key](features['pool'])
                target_label = []
                for index,data in enumerate(targets):
                    target_label.append(data[key])
                target_label = torch.as_tensor(target_label, dtype=torch.int64)
                target_label = target_label.to(device)
                multitask_loss_dict[key] = criterion(output, target_label)
                losses += multitask_loss_dict[key]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if len(multitask_model) is not 0:
            metric_logger.update(multitask = sum(multitask_loss_dict.values()))


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, multitask_model, data_loader, custom, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    total_correct = {}
    total_correct['fracture'] = 0
    total_correct['modality'] = 0
    total_correct['subtype'] = 0
    total_correct['Age'] = 0
    total = {}
    total['fracture'] = 0
    total['modality'] = 0
    total['subtype'] = 0
    total['Age'] = 0
    for image, targets, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #torch.cuda.synchronize()
        model_time = time.time()
        outputs, features = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        if len(multitask_model) is not 0:
            for key in multitask_model.keys():
                output = multitask_model[key](features['pool'])
                target_label = []
                for index,data in enumerate(targets):
                    target_label.append(data[key])
                target_label = torch.as_tensor(target_label, dtype=torch.int64)
                target_label = target_label.to(device)
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target_label).sum().item()
                total[key] += len(targets)
                total_correct[key] += correct

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    if len(multitask_model) is not 0:
        multitask_loss_dict = {}
        for key in multitask_model.keys():
            print(key)
            print(total_correct[key]/ total[key])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
