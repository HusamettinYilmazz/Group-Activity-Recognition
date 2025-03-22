import os 
import pickle
from PIL import Image
from albumentations.augmentations import crops
import cv2

import torch
from torch.utils.data import Dataset


ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"

from .helper import load_config
from .boxinfo import BoxInfo

group_activity_categories = ["r_set", "r_spike" , "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {categ:idx for idx, categ in enumerate(group_activity_categories)}

person_activity_categories = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_categories = [categ.lower() for categ in person_activity_categories]
person_activity_labels = {categ:idx for idx, categ in enumerate(person_activity_categories)}

class GroupActivityRecognitionDataset(Dataset):
    def __init__(self, videos_path, annot_path, split, crop=False, transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.load_frame_label_pair(videos_path, annot_path, split, crop, transform=self.transform)


    def load_frame_label_pair(self, videos_path, annot_path, split, crop, transform):
        split = [str(i) for i in split]
        
        
        with open(annot_path, "rb") as annot_file:
            annot_file = pickle.load(annot_file)
        
        for video in split:
            video_dir = os.path.join(videos_path, video)
            for clip in annot_file[video]:
                frame_path = os.path.join(video_dir, clip, f"{clip}.jpg")
                
                if crop:
                    for player in annot_file[video][clip]['frame_boxes_dct'][int(clip)]:
                        box_info = player.box
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
        if box_info:
            image = cv2.imread(frame_path)
            x1, y1, x2, y2= box_info[0], box_info[1], box_info[2], box_info[3]
            crop = image[y1:y2, x1:x2]
            image = self.transform(image=crop)['image'] if self.transform else crop
            
            # return (crop, label)
        else:
            image = cv2.imread(frame_path)
            # frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
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
    config_path = os.path.join(ROOT, "modeling/configs/baseline3.yaml")
    config = load_config(config_path)
    # print(config.model)
    annot_path = os.path.join(ROOT, config.data["annot_path"])
    videos_path = os.path.join(ROOT, config.data["videos_path"])
    split = config.data['video_splits']['train']

    group_act = GroupActivityRecognitionDataset(videos_path, annot_path, split, crop=True)
    print(group_act.__getitem__(idx=0))

    # load_and_print_pkl_tree(os.path.join(ROOT, config.data['annot_path']))

