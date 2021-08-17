# Standard library imports
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# third party imports
clip_dir = Path(".").absolute() / "CLIP"
sys.path.append(str(clip_dir))
print(f"CLIP dir is: {clip_dir}")
import clip

# set constants
DATA_FEWSHOT =  Path(".").absolute() / "data/coco_crops_few_shot/train"
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = os.listdir(DATA_FEWSHOT)

# Load CLIP pre-trained model
model, transform = clip.load("ViT-B/32", device=device)
print(f"Model dir: {os.path.expanduser('~/.cache/clip')}")

# Get the encoded text features of possible labels
class_captions = [f"a photo of a {x}" for x in class_names]
text_input = clip.tokenize(class_captions).to(device)
print(f"Tokens shape: {text_input.shape}")
with torch.no_grad():
    text_features = model.encode_text(text_input).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
print(f"Text features shape: {text_features.shape}")


# prepare test images dataset
dataset = ImageFolder(root=DATA_FEWSHOT, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
dataset_idx_to_class = {dataset.class_to_idx[k]:k for k in dataset.class_to_idx}

# CLIP prediction for test images
pred_class_names, gt_class_names = [], []
with torch.no_grad():
    for data in dataloader:
        image, label = data
        image = image.to(device)
        image_features = model.encode_image(image).float()
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred_class_idx = torch.argmax(similarity, dim=1)
        pred_class_names_batch = [class_names[i] for i in pred_class_idx]
        gt_class_names_batch = [dataset_idx_to_class[k] for k in label.tolist()]
        pred_class_names.extend(pred_class_names_batch)
        gt_class_names.extend(gt_class_names_batch)

# eval metrics
fig, ax = plt.subplots(figsize=(10,10))
cf_mat = confusion_matrix(gt_class_names, pred_class_names, labels=class_names)
print(cf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=class_names)
disp.plot(ax=ax)
plt.savefig('clip_fewshot_train.png')
with open('clip_fewshot_train.txt', 'w') as fp:
    fp.write(classification_report(gt_class_names, pred_class_names, labels=class_names))