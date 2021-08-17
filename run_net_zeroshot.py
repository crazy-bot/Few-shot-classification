import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import sys
from pathlib import Path
import os
import torch
import argparse
import numpy as np
from train_util import Net, load_imgs, getTransform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
clip_dir = Path(".").absolute() / "CLIP"
sys.path.append(str(clip_dir))
print(f"CLIP dir is: {clip_dir}")
import clip

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='models/model_180.ckpt')
args = parser.parse_args()

DATA_FEWSHOT_TRAIN =  Path(".").absolute() / "data/coco_crops_few_shot/train"
DATA_ZEROSHOT =  Path(".").absolute() / "data/coco_crops_zero_shot/test"
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, clip_transform = clip.load("ViT-B/32", device=device)
print(f"CLIP Model dir: {os.path.expanduser('~/.cache/clip')}")

net = Net()
net = net.to(device)
net.load_state_dict(torch.load(args.ckpt))
net.eval()

trained_class_names = os.listdir(DATA_FEWSHOT_TRAIN)

# Load test images
dataset_test = ImageFolder(root=DATA_ZEROSHOT, transform = clip_transform)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
dataset_idx_to_class = {dataset_test.class_to_idx[k]:k for k in dataset_test.class_to_idx}

f = open('coco_classes.txt')
class_names = []
for line in f:
    line = line.strip()
    _, value = line.split(':')
    class_names.append(value.strip()[2:-2])
    class_captions = [f"a photo of a {x}" for x in class_names]
text_input = clip.tokenize(class_captions).to(device)
print(f"Tokens shape: {text_input.shape}")

with torch.no_grad():
    text_features = CLIP_model.encode_text(text_input).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
print(f"Text features shape: {text_features.shape}")

pred_class_scores, pred_class_idxs = [], []
unseen_files, pred_class_names = [],[]
for i, data in enumerate(dataloader_test):
    image, label = data
    image, label = image.to(device), label.to(device)
    with torch.no_grad():
        image_features = CLIP_model.encode_image(image).float()
        output = net(image_features)

        pred_class_score, pred_class_idx = torch.max(output, dim=1)
        pred_class_score, pred_class_idx = pred_class_score.detach().cpu().numpy(), pred_class_idx.detach().cpu().numpy()
        # consider the image as unlnown class if max score is less than 0.5
        if pred_class_score[0] < 0.5:
            fname = dataloader_test.dataset.samples[i]
            # run CLIP for unknown images 
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_class_idx = torch.argmax(similarity, dim=1)
            unseen_files.append(fname)
            pred_class_names.append(class_names[pred_class_idx[0]])

fp = open('unseen_class.txt','w')
for img, cls in zip(unseen_files, pred_class_names ):
    print('fname: {} class:{}'.format(img,cls))
    fp.write('fname: {} class:{}\n'.format(img,cls))

