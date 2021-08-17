import torch
from PIL import Image
import glob
from torchvision.transforms import Compose, ColorJitter, RandomRotation

class Net(torch.nn.Module):

    def __init__(self, in_feat=512, out_feat=8):
        super(Net, self).__init__()
        #self.linear1 = torch.nn.Linear(in_feat, in_feat)
        self.linear2 = torch.nn.Linear(in_feat, out_feat)
        self.m = torch.nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = self.m(self.linear2(X))
        return X


def load_imgs(path_to_folder, transform):
    imgs = []
    paths = glob.glob(str(path_to_folder))
    for p in paths:
        img = Image.open(p)
        img = transform(img)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs

def getTransform(clip_transform):
    return Compose([
       ColorJitter(brightness=0.5, contrast=0.4, saturation=0.5, hue=0.3),
       RandomRotation(degrees=(0,90)),
       clip_transform
    ])