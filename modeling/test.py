import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"

sys.path.append(ROOT)

from baseline1_model import GroupActivity
from train import validate_model
from utils import load_config, GroupActivityRecognitionDataset

def test_model(config, model_path, loss_func, test_transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_point = torch.load(model_path, map_location= device)

    dataset_path = os.path.join(ROOT, config.data['videos_path'])
    annot_path = os.path.join(ROOT, config.data['annot_path'])

    model = GroupActivity(out_features=config.model['num_classes'])
    model.load_state_dict(check_point['model'])
    model.to(device)

    test_dataset = GroupActivityRecognitionDataset(dataset_path, annot_path, split=config.data['video_splits']['test'], transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory=True)

    save_file=os.path.join(ROOT, config.data['output_path'], config.experiment['name'], 'test_conf_matrix.png')
    test_avg_loss, test_acc, f1_score = validate_model(test_loader, device, model, loss_func, config.model['class_labels'], save_file)

    return test_avg_loss, test_acc, f1_score


if __name__ == "__main__":
    config_path = os.path.join(ROOT, "modeling/configs/baseline1.yaml")
    config = load_config(config_path)
    model_path = '/kaggle/working/Group-Activity-Recognition/modeling/outputs/Baseline 1/epoch15_model.pth'
    
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    loss_func = nn.CrossEntropyLoss()
    
    test_avg_loss, test_acc, f1_score = test_model(config, model_path, loss_func, test_transform)
    print(f" Test Loss {test_avg_loss:.4f} | Accuracy {test_acc:.2f}% | F1 Score {f1_score:.4}")