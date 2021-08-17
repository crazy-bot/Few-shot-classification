# Standard library imports
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# third party imports
clip_dir = Path(".").absolute() / "CLIP"
sys.path.append(str(clip_dir))
print(f"CLIP dir is: {clip_dir}")
import clip

# local imports
from train_util import load_imgs

# set constants
DATA_FEWSHOT_TRAIN =  Path(".").absolute() / "data/coco_crops_few_shot/train"
DATA_FEWSHOT_TEST =  Path(".").absolute() / "data/coco_crops_few_shot/test"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP pre-trained model
model, transform = clip.load("ViT-B/32", device=device)
print(f"Model dir: {os.path.expanduser('~/.cache/clip')}")

# get all the class names
class_names = os.listdir(DATA_FEWSHOT_TRAIN)

# store mean feature (CLIP) of all train images per class
class_mean_feat = []
for class_ in class_names:
    # Load train images
    images = load_imgs(Path.joinpath(DATA_FEWSHOT_TRAIN, class_, '*'), transform)
    images = images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features_mean = torch.mean(image_features, dim=0)
        class_mean_feat.append(image_features_mean)
class_mean_feat = torch.stack(class_mean_feat).unsqueeze(0)

# Testing starts: classify based on nearest class mean from training data (NCM)
pred_class_names, gt_class_names = [], []
for class_ in class_names:
    print(class_)
    # Load test images
    images = load_imgs(Path.joinpath(DATA_FEWSHOT_TEST, class_, '*'), transform)
    images = images.to(device)
    with torch.no_grad():
        test_image_features = model.encode_image(images)
        class_dist_mat = torch.cdist(test_image_features.unsqueeze(0), class_mean_feat, p=2.0)
        class_dist_mat = class_dist_mat[0]
        pred_class = torch.argmin(class_dist_mat, dim=1)
        gt_class_names.extend([class_]*len(pred_class))
        pred_class_names.extend([class_names[k] for k in pred_class.tolist()])

# eval metrics
fig, ax = plt.subplots(figsize=(10,10))
cf_mat = confusion_matrix(gt_class_names, pred_class_names, labels=class_names)
print(cf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=class_names)
disp.plot(ax=ax)
plt.savefig('NCM_fewshot.png')
with open('NCM_fewshot.txt', 'w') as fp:
    fp.write(classification_report(gt_class_names, pred_class_names, labels=class_names))