import os 
import pickle
import cv2

import torch
from torch.utils.data import Dataset


ROOT = "/teamspace/studios/this_studio/Group-Activity-Recognition"

from .helper import load_config
from .boxinfo import BoxInfo

group_activity_categories = ["r_set", "r_spike" , "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {categ:idx for idx, categ in enumerate(group_activity_categories)}

class GroupActivityRecognitionDataset(Dataset):
    def __init__(self, videos_path, annot_path, split, transform=None):
        # super(Dataset, GroupActivityRecognition)
        self.data = []
        self.load_frame_label_pair(videos_path, annot_path, split, transform)


    def load_frame_label_pair(self, videos_path, annot_path, split, transform):
        split = [str(i) for i in split]
        label = torch.zeros(len(group_activity_categories)) ## One hot encoding
        
        with open(annot_path, "rb") as annot_file:
            annot_file = pickle.load(annot_file)
        
        for video in split:
            video_dir = os.path.join(videos_path, video)
            for clip in annot_file[video]:
                frame_path = os.path.join(video_dir, clip, f"{clip}.jpg")
                
                category = annot_file[video][clip]['category']
                # label = [1 if cur_label == label else 0 for cur_label in config.model['class_labels']] ## One hot encoding
                
                label[group_activity_labels[category]] = 1
                
                frame = cv2.imread(frame_path)
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                frame = transform(frame) if transform else frame

                self.data.append((frame, label))

                print(frame_path)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def print_dict_hierarchy(d, indent=0):
    for key in d:
        print(" " * indent + str(key))
        if isinstance(d[key], dict):  # If value is a nested dictionary, recurse
            print_dict_hierarchy(d[key], indent + 4)
        else:
            continue
        # elif isinstance(d[key], list):
            # print_dict_hierarchy(d[0], indent + 4)

if __name__ == "__main__":
    config_path = os.path.join(ROOT, "modeling/configs/baseline1.yaml")
    config = load_config(config_path)
    # print(config.model)
    annot_path = os.path.join(ROOT, config.data["annot_path"])
    videos_path = os.path.join(ROOT, config.data["videos_path"])
    split = config.data['video_splits']['train']

    group_act = GroupActivityRecognitionDataset(videos_path, annot_path, split)
    print(group_act.__len__())

