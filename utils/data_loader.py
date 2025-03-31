import os 
import sys

import pickle
from PIL import Image
from albumentations.augmentations import crops
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"
# sys.path.append(ROOT)

from .helper import load_config
from .boxinfo import BoxInfo
from modeling.baseline3B_model import GroupActivity3B ## reput the 2 dots ..modeling


group_activity_categories = ["r_set", "r_spike" , "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {categ:idx for idx, categ in enumerate(group_activity_categories)}

person_activity_categories = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_categories = [categ.lower() for categ in person_activity_categories]
person_activity_labels = {categ:idx for idx, categ in enumerate(person_activity_categories)}

class GroupActivityRecognitionDataset(Dataset):
    def __init__(self, videos_path, annot_path, split, crop=False, all_players_once=False, transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GroupActivity3B(out_features=1).to(self.device)
        
        self.load_frame_label_pair(videos_path, annot_path, split, crop, all_players_once)


    def load_frame_label_pair(self, videos_path, annot_path, split, crop, all_players_once):
        split = [str(i) for i in split]
        
        
        with open(annot_path, "rb") as annot_file:
            annot_file = pickle.load(annot_file)
        
        for video in split:
            video_dir = os.path.join(videos_path, video)
            for clip in annot_file[video]:
                frame_path = os.path.join(video_dir, clip, f"{clip}.jpg")
                
                if crop:                    
                    if all_players_once:
                        
                        player_boxes = []
                        for player in annot_file[video][clip]['frame_boxes_dct'][int(clip)]:
                            player_boxes.append(player.box)
                            
                        category = annot_file[video][clip]['category']
                        # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                        label = torch.zeros(len(group_activity_categories)) ## One hot encoding
                        label[group_activity_labels[category]] = 1
                        
                        self.data.append((frame_path, player_boxes, label))
                        
                    else:
                        for player in annot_file[video][clip]['frame_boxes_dct'][int(clip)]:
                            box_info = [player.box]
                            category = player.category
                            label = torch.zeros(len(person_activity_categories)) ## One hot encoding
                            label[person_activity_labels[category]] = 1
    
                            self.data.append((frame_path, box_info, label))
                    
                else:
                    category = annot_file[video][clip]['category']
                    # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                    label = torch.zeros(len(group_activity_categories)) ## One hot encoding
                    label[group_activity_labels[category]] = 1
                    box_info = None

                    self.data.append((frame_path, box_info, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_path, box_info, label = self.data[idx]
        image = cv2.imread(frame_path)

        if len(box_info) > 1:
            crops = []            
            for box in box_info:
                x1, y1, x2, y2= box[0], box[1], box[2], box[3]
                crop = image[y1:y2, x1:x2]
                crop = self.transform(image=crop)['image'] if self.transform else crop
                crops.append(crop)
                
            stacked_crops = np.stack(crops, axis=0)
            crops = torch.tensor(stacked_crops, dtype=torch.float32).to(self.device)
            
            image = self.model.feat_extraction(crops)  ## both of feature extraction and max pooling applied
            
        elif len(box_info) == 1:
            box_info = box_info[0]
            x1, y1, x2, y2= box_info[0], box_info[1], box_info[2], box_info[3]
            crop = image[y1:y2, x1:x2]
            image = self.transform(image=crop)['image'] if self.transform else crop
            
        else:
            image = self.transform(image=image)["image"] if self.transform else image
            
        return (image, label)


def print_one_branch(data, indent=0):
    """ Recursively print one branch from the root to a leaf. """
    prefix = " " * (indent * 4)

    if isinstance(data, dict):  # If it's a dictionary, take the first key
        key = next(iter(data))  # Get first key
        print(f"{prefix}video: {key}")
        print_one_branch(data[key], indent + 1)
    
    elif isinstance(data, list):  # If it's a list, take the first element
        if data:
            print(f"{prefix}image: 0")  # Since it's a list, we assume first item
            print_one_branch(data[0], indent + 1)
    
    elif isinstance(data, tuple):
        if data:
            print(f"{prefix}Tuple: (first item)")
            print_one_branch(data[0], indent + 1)
    
    elif isinstance(data, str):  # If it's a string, assume it's a class name
        print(f"{prefix}class: {data}")
    
    elif isinstance(data, BoxInfo):
        print(f"{prefix}class: {data}")

# Load and print the structure of a .pkl file
def load_and_print_pkl_tree(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print_one_branch(data)

if __name__ == "__main__":
    config_path = os.path.join(ROOT, "modeling/configs/baseline3B.yaml")
    config = load_config(config_path)
    # print(config.model)
    annot_path = os.path.join(ROOT, config.data["annot_path"])
    videos_path = os.path.join(ROOT, config.data["videos_path"])
    split = config.data['video_splits']['train']

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

    group_act = GroupActivityRecognitionDataset(videos_path, annot_path, split, crop=True, all_players_once=False, transform=train_transform)
    print(group_act.__getitem__(idx=0))

    # load_and_print_pkl_tree(os.path.join(ROOT, config.data['annot_path']))

