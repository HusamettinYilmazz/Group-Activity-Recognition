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


group_activity_categories = ["r_set", "r_spike" , "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {categ:idx for idx, categ in enumerate(group_activity_categories)}

person_activity_categories = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_categories = [categ.lower() for categ in person_activity_categories]
person_activity_labels = {categ:idx for idx, categ in enumerate(person_activity_categories)}

class GroupActivityRecognitionDataset(Dataset):
    def __init__(self, videos_path, annot_path, split, seq=False, crop=False, all_players_once=False, transform=None):
        super().__init__()
        self.transform = transform
        self.seq = seq
        self.crop = crop
        self.all_players_once = all_players_once
        self.data = []
        
        
        self.load_frame_label_pair(videos_path, annot_path, split)


    def load_frame_label_pair(self, videos_path, annot_path, split):
        split = [str(i) for i in split]
        
        
        with open(annot_path, "rb") as annot_file:
            annot_file = pickle.load(annot_file)
        
        for video in split:
            video_dir = os.path.join(videos_path, video)
            
            for clip in annot_file[video]:
                frame_path = os.path.join(video_dir, clip, f"{clip}.jpg")
                frame_paths = [frame_path]

                if not self.crop and not self.seq:

                    category = annot_file[video][clip]['category']
                    # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                    label = torch.zeros(len(group_activity_categories)) ## One hot encoding
                    label[group_activity_labels[category]] = 1
                    box_info = [None]

                    self.data.append((frame_paths, box_info, label))

                elif self.crop and not self.seq:
                
                    if self.all_players_once:
                        
                        player_boxes = []
                        for player in annot_file[video][clip]['frame_boxes_dct'][int(clip)]:
                            player_boxes.append(player.box)

                        category = annot_file[video][clip]['category']
                        # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                        label = torch.zeros(len(group_activity_categories)) ## One hot encoding
                        label[group_activity_labels[category]] = 1
                        
                        self.data.append((frame_paths, player_boxes, label))
                        
                    else:
                        for player in annot_file[video][clip]['frame_boxes_dct'][int(clip)]:
                            box_info = [player.box]
                            category = player.category
                            label = torch.zeros(len(person_activity_categories)) ## One hot encoding
                            label[person_activity_labels[category]] = 1
    
                            self.data.append((frame_paths, box_info, label))

                elif not self.crop and self.seq:
                    
                    frame_paths = [] 
                    for i in range(-4, 5):
                        frame_path = os.path.join(video_dir, clip, f"{int(clip)+i}.jpg")
                        frame_paths.append(frame_path)

                    category = annot_file[video][clip]['category']
                    # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                    label = torch.zeros(len(group_activity_categories)) ## One hot encoding
                    label[group_activity_labels[category]] = 1
                    box_info = [None]
                    
                    self.data.append((frame_paths, box_info, label))

                else:   ## if self.crop and self.seq: write the logic
                    ## padding the lost players (12 player) in total for each frame
                    pass
                    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, box_info, label = self.data[idx]

        if not self.crop and not self.seq:
            frame_path = frame_paths[0]
            image = cv2.imread(frame_path)
            image = self.transform(image=image)["image"] if self.transform else image

        elif self.crop and not self.seq:
            frame_path = frame_paths[0]
            image = cv2.imread(frame_path)

            if self.all_players_once:

                crops = []            
                for box in box_info:
                    x1, y1, x2, y2= box[0], box[1], box[2], box[3]
                    crop = image[y1:y2, x1:x2]
                    crop = self.transform(image=crop)['image'] if self.transform else crop
                    crops.append(crop)
                crops = torch.stack(crops)
                image = complete_sequence(crops)

            else:
                box_info = box_info[0]
                x1, y1, x2, y2= box_info[0], box_info[1], box_info[2], box_info[3]
                crop = image[y1:y2, x1:x2]
                image = self.transform(image=crop)['image'] if self.transform else crop

        elif not self.crop and self.seq:

            images = []
            for frame_path in frame_paths:
                image = cv2.imread(frame_path)
                image = self.transform(image=image)["image"] if self.transform else image
                images.append(image)

            image = torch.stack(images)

        else:   ## if self.crop and self.seq: Write the logic
            pass
            
        return (image, label)


def complete_sequence(sequence):
        seq_length, c, h, w = sequence.shape
        if seq_length < 12:
            new_frame = torch.zeros(12 - seq_length, c, h, w)
            sequence = torch.cat((sequence, new_frame), dim=0)

        elif seq_length > 12:
            sequence = sequence[:12]

        return sequence



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
    config_path = os.path.join(ROOT, "configs/baseline3B.yaml")
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

    group_act = GroupActivityRecognitionDataset(videos_path, annot_path, split, seq=config.experiment['sequential'], crop=config.experiment['crop'], all_players_once=config.experiment['all_players_once'], transform=train_transform)
    print(group_act.__getitem__(idx=0))

    # load_and_print_pkl_tree(os.path.join(ROOT, config.data['annot_path']))

