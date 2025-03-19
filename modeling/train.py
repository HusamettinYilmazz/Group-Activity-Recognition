import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"
sys.path.append(ROOT)


from baseline1_model import GroupActivity
from utils import load_config, GroupActivityRecognitionDataset
from utils import plot_conf_matrix, get_f1_score
from utils import lr_vs_epoch


def train_an_epoch(data_loader, device, model, optimizer, loss_func):
    total_loss, total_trues, total_examples = 0, 0, 0
    model.train()
    for batch_idx, (frames, targets) in enumerate(data_loader):
        frames, targets = frames.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(frames)
        loss = loss_func(outputs, targets)
        
        loss.backward()
        optimizer.step()

        pred_outputs = outputs.argmax(1)
        target_classes = targets.argmax(1)
        total_trues += target_classes.eq(pred_outputs).sum().item()
        total_examples += targets.size(0)

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"TRAIN: Batch:{batch_idx}/{len(data_loader)} Loss:{loss.item():.3f} & Accuracy:{(total_trues / total_examples) * 100:.2f}%")
    
    epoch_avg_loss = total_loss / len(data_loader)
    epoch_acc = (total_trues / total_examples) * 100
    
    return epoch_avg_loss, epoch_acc


def validate_model(data_loader, device, model, loss_func, class_names, save_dir=None):
    total_loss, total_trues, total_examples = 0, 0, 0
    y_true, y_pred = [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (frames, targets) in enumerate(data_loader):
            frames, targets = frames.to(device), targets.to(device)

            outputs = model(frames)
            loss = loss_func(outputs, targets)
            total_loss += loss.item()
            
            pred_outputs = outputs.argmax(1)
            target_classes = targets.argmax(1)
            total_trues += target_classes.eq(pred_outputs).sum().item()
            total_examples += targets.size(0)

            y_true.extend(target_classes.cpu().numpy())
            y_pred.extend(pred_outputs.cpu().numpy())
            
    f1_score = get_f1_score(y_true, y_pred)
    
    fig = plot_conf_matrix(y_true, y_pred, class_names)
    if save_dir:
        plt.savefig(save_dir, dpi=300)
        print(f"Confusion matrix saved at: {save_dir}")
            
    epoch_avg_loss = total_loss / len(data_loader)
    epoch_acc = (total_trues/total_examples) * 100

    return epoch_avg_loss, epoch_acc, f1_score


def train_model(config):
    dataset_path = os.path.join(ROOT, config.data['videos_path'])
    annot_path = os.path.join(ROOT, config.data['annot_path'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.9),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    train_dataset = GroupActivityRecognitionDataset(dataset_path, annot_path, split=config.data['video_splits']['train'], transform=train_transform)
    val_dataset = GroupActivityRecognitionDataset(dataset_path, annot_path, split=config.data['video_splits']['validation'], transform=val_transform)
    test_dataset = GroupActivityRecognitionDataset(dataset_path, annot_path, split=config.data['video_splits']['test'], transform=val_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory=True)

    model = GroupActivity(out_features=config.model['num_classes'])
    model = model.to(device)

    optimizer = AdamW(params= model.parameters(), lr=float(config.training['learning_rate']), weight_decay=float(config.training['weight_decay']))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    loss_func = nn.CrossEntropyLoss()

    save_dir = os.path.join(ROOT, config.data['output_path'], config.experiment['name'])
    os.makedirs(save_dir, exist_ok=True)
    lrs = []
    for epoch in range(1, config.training['num_epochs']+1):
        save_file = os.path.join(save_dir, f'epoch{epoch}_conf_matrix.png')
        
        print(f"Epoch:{epoch}")
        train_avg_loss, train_acc =train_an_epoch(train_loader, device, model, optimizer, loss_func)
        print(f"FULL_EPOCH TRAIN: Loss:{train_avg_loss:.3f} & Accuracy:{train_acc:.2f}%")
        
        val_avg_loss, val_acc, f1_score = validate_model(val_loader, device, model, loss_func, config.model['class_labels'], save_file)
        print(f" Validation Loss {val_avg_loss:.4f} | Accuracy {val_acc:.2f}% | F1 Score {f1_score:.4}")
        
        scheduler.step(val_avg_loss)
        
        cur_lr = optimizer.param_groups[0]['lr']
        lrs.append(cur_lr)
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning rate': cur_lr,
            'val_accuracy': val_acc,
            'config': config
        }, os.path.join(save_dir, f'epoch{epoch}_model.pth'))

    ## Validate test dataset with last model
    val_avg_loss, val_acc, f1_score = validate_model(test_loader, device, model, loss_func, config.model['class_labels'], save_file)
    lr_vs_epoch(config.training['num_epochs'], lrs, save_dir)



if __name__ == "__main__":
    config_path = os.path.join(ROOT, "modeling/configs/baseline1.yaml")
    config = load_config(config_path)
    
    train_model(config)
    