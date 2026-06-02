import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer
import json
from tqdm import tqdm

#from utils import transform


def create_image_inputs(karpathy_json_path, image_dir, transform):
    """
    This function preprocesses the image and saves it in the image_dir.
    I's faster for the training process to load the images from the torch file.
    """
    karpathy = json.load(open(karpathy_json_path, "r"))
    bar = tqdm(karpathy["images"])
    for image in bar:
        image_path = os.path.join(image_dir, image["filepath"], image["filename"])
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        torch.save(image, image_path.replace(".jpg", ".pt"))


class ImageCaptionDataset(Dataset):
    def __init__(self, karpathy_json_path, image_dir, tokenizer, max_seq_len=256, transform=None, phase="train"):
        self.transform = transform
        self.tokenizer = tokenizer
        self.karpathy_json_path = karpathy_json_path
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.train="data/Flickr8k_text/Flickr_8k.trainImages.txt"
        self.val="data/Flickr8k_text/Flickr_8k.devImages.txt"
        self.test = "data/Flickr8k_text/Flickr_8k.testImages.txt"
        self.df = self.create_inputs()
    
    def create_inputs(self):
        df = []
        count=0
        captions_file_text = self.load_file_text(self.karpathy_json_path)  # read caption file
        img_cpts= self.get_captions(captions_file_text)  # get image-names,caption
        for image in img_cpts:
            image_path = os.path.join(self.image_dir, "Flicker8k_Dataset", image+".jpg")
            captions = img_cpts[image]
            for caption in captions:
                row = {
                    "image_id": count,
                    "image_path": image_path, 
                    "caption": caption, 
                    "all_captions": captions+[""]*(10-len(captions))
                    }
                if self.phase == "train" and image in self.get_phase(self.train):
                    df.append(row)
                elif self.phase == "val" and image in self.get_phase(self.test):
                    df.append(row)
                elif self.phase == "test" and image in self.get_phase(self.test):
                    df.append(row)
            count=count+1
        return pd.DataFrame(df)#.sample(frac=0.0001).reset_index(drop=True)


    def load_file_text(self,file_path):
        """reads arabic text and returns text in captions file """
        file = open(file_path, 'r', encoding='utf-8')
        all_text = file.read()
        file.close()
        return all_text

    def get_captions(self,file_text):

        cpts = {}
        # loop through lines
        for line in file_text.split('\n'):  # each line contains image name & its caption separated by tab
            # split by tabs
            img_cpt = line.split('\t')
            if len(img_cpt) < 2: continue
            img, cpt = img_cpt
            # remove image extension & index (remove everything befor the dot)
            img_name = img.split('.')[0]
            # add to dictionary
            if img_name not in cpts:
                cpts[img_name] = [cpt]
            else:
                cpts[img_name].append(cpt)
        return cpts

    def get_phase(self, file):
        imgs = self.load_file_text(file)
        img_names=[img.split('.')[0]  for img in imgs.split('\n')]
        return img_names


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.df.iloc[index]["image_path"]
        image_torch_path = image_path.replace(".jpg", ".pt")
        if os.path.exists(image_torch_path):
            image = torch.load(image_torch_path)
        else:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            torch.save(image, image_torch_path)

        caption = self.df.loc[index, "caption"]
        caption_tokens = self.tokenizer(caption, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
        all_captions = self.df.loc[index, "all_captions"]
        all_captions_tokens = self.tokenizer(all_captions, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        return {
            "image_id": self.df.loc[index, "image_id"],
            "image_path": image_path,
            "image": image,
            "caption_seq": caption,
            "caption": caption_tokens,
            "all_captions_seq": all_captions,
            "all_captions": all_captions_tokens
        }

# Test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased", help="Bert tokenizer")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128, help="Maximum sequence length for caption generation")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size")
    # Data parameters
    parser.add_argument("--image_dir", "-id", type=str, default="../coco/", help="Path to image directory, this contains train2014, val2014")
    parser.add_argument("--karpathy_json_path", "-kap", type=str, default="../coco/karpathy/dataset_coco.json", help="Path to karpathy json file")
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    dataset = ImageCaptionDataset(
        karpathy_json_path=args.karpathy_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform, 
        phase="train"
    )
    print(dataset[0])

    create_image_inputs(args.karpathy_json_path, args.image_dir, transform)