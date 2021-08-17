# Standard library imports
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score

# third party imports
clip_dir = Path(".").absolute() / "CLIP"
sys.path.append(str(clip_dir))
print(f"CLIP dir is: {clip_dir}")
import clip
# local imports
from train_util import Net, getTransform

# command line arguments setup
parser = argparse.ArgumentParser()
parser.add_argument('--isTest', action='store_true')
parser.add_argument('--ckpt', default='models/model_180.ckpt')
args = parser.parse_args()

if args.isTest:
    print('Test mode')
else:
    print('Train mode')

# training configuration
num_epochs = 1000
MODEL_DIR = 'models'
DATA_FEWSHOT_TRAIN =  Path(".").absolute() / "data/coco_crops_few_shot/train"
DATA_FEWSHOT_TEST =  Path(".").absolute() / "data/coco_crops_few_shot/test"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP pre-trained model
CLIP_model, clip_transform = clip.load("ViT-B/32", device=device)
print(f"CLIP Model dir: {os.path.expanduser('~/.cache/clip')}")

# initialize/load network
net = Net()
net = net.to(device)
if args.isTest and args.ckpt != '':
    net.load_state_dict(torch.load(args.ckpt))
else:
    net.train()
    nllloss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.0005)

# prepare dataset
dataset = ImageFolder(root=DATA_FEWSHOT_TRAIN, transform = getTransform(clip_transform))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset_idx_to_class = {dataset.class_to_idx[k]:k for k in dataset.class_to_idx}
class_names = os.listdir(DATA_FEWSHOT_TRAIN)

dataset_test = ImageFolder(root=DATA_FEWSHOT_TEST, transform = clip_transform)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

# training loop
_f1score = 0.0
for epoch in range(num_epochs):
    # train if not in test mode
    if not args.isTest:
        loss_epoch = 0
        for data in dataloader:
            image, label = data
            #image = clip_transform(image)
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                image_features = CLIP_model.encode_image(image).float()
            optimizer.zero_grad()
            output = net(image_features)
            loss = nllloss(output, label)
            loss.backward()
            optimizer.step()
            loss_epoch = loss_epoch + loss.item()

        print('epoch: {} loss: {}'.format(epoch, loss_epoch))

    # do validation or test
    if (epoch%10 == 0 and epoch!=0) or args.isTest:
        print('-----start of evaluation------')
        net.eval()
        pred_class_names, gt_class_names = [], []
        for data in dataloader_test:
            image, label = data
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                image_features = CLIP_model.encode_image(image).float()
            output = net(image_features)

            pred_class_idx = torch.argmax(output, dim=1)
            pred_class_names_batch = [dataset_idx_to_class[k] for k in pred_class_idx.tolist()]
            gt_class_names_batch = [dataset_idx_to_class[k] for k in label.tolist()]
            pred_class_names.extend(pred_class_names_batch)
            gt_class_names.extend(gt_class_names_batch)

        f1score = f1_score(gt_class_names, pred_class_names, average='micro')
        cf_mat = confusion_matrix(gt_class_names, pred_class_names, labels=class_names)
        print('f1score: ',f1score)
        if args.isTest:
           fig, ax = plt.subplots(figsize=(10,10))
           disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat, display_labels=class_names)
           disp.plot(ax=ax)
           plt.savefig('net_fewshot.png')
           with open('net_fewshot.txt', 'w') as fp:
               fp.write(classification_report(gt_class_names, pred_class_names, labels=class_names))
           break
        else:
            print(classification_report(gt_class_names, pred_class_names, labels=class_names))
            # save best validation model
            if f1score > _f1score:
                # saving model checkpoint
                torch.save(net.state_dict(), MODEL_DIR+'/model_{}.ckpt'.format(epoch))
                _f1score = f1score

