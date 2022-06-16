from .custommodel.faster_rcnn import FastRCNNPredictor
from .custommodel.mask_rcnn import MaskRCNNPredictor
from .custommodel import maskrcnn_resnet50_fpn
from torch import nn

def mask_model_custom(pretrained, num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=pretrained,
                                                   image_mean = [0.2915, 0.2915, 0.2915],
                                                   image_std = [0.2422, 0.2422, 0.2422])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

class multitask_predictor(nn.Module):
    def __init__(self, num_classes):
        super(multitask_predictor, self).__init__()
        self.intermediate = nn.Linear(256*13*13, 512)
        self.cls_scores = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim = 1)
        x = self.intermediate(x)
        x = self.cls_scores(x)

        return x

class multiview_predictor(nn.Module):
    def __init__(self, num_classes):
        super(multiview_predictor, self).__init__()
        self.cls_score = nn.Linear(2*1024, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)

        return scores

