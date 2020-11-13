import torch
import os
from PIL import Image
import glob
import torchvision
import torchvision.transforms as transforms
import numpy as np

class DataLoaderSegmentation(torch.utils.data.Dataset):
    def __init__(self, folder_path,  transform=None, train = False):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'*.png'))
        self.mask_files = []
        self.transform = transform
        self.train = train
        self.crop_size = (200,200)
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path))) 

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = Image.open(img_path)
            label = Image.open(mask_path)
            if self.train == True:
                pass
                #data, label = RandomFlip2im(data,label)
                #data, label = RandomRotate2im(data,label)
            if self.transform:
                data = self.transform(data)
                label = torch.from_numpy(np.asarray(label))
                label = label.permute(2,0,1)
            return data, label

    def __len__(self):
        return len(self.img_files)

def collate_wrapper(batch):
    size = self.crop_size 
    trans = transforms.ToTensor()
    norm = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    databatch = []
    labelbatch = []
    for item in batch:
        data, label = item
        data, label = RandomCrop2im(data, label, size)
        data = trans(data)
        data = norm(data)
        label = torch.from_numpy(np.asarray(label))
        label = label.permute(2,0,1)
        databatch.append(data)
        labelbatch.append(label)
    return torch.stack(databatch), torch.stack(labelbatch)

class Collator(object):
    def __init__(self, crop_size = (200,200)):
        self.crop_size = crop_size
        
    def __call__(self, batch):
        size = self.crop_size 
        trans = transforms.ToTensor()
        norm = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        databatch = []
        labelbatch = []
        for item in batch:
            data, label = item
            data, label = RandomCrop2im(data, label, size)
            data = trans(data)
            data = norm(data)
            label = torch.from_numpy(np.asarray(label))
            label = label.permute(2,0,1)
            databatch.append(data)
            labelbatch.append(label)
        return torch.stack(databatch), torch.stack(labelbatch)
    
def RandomCrop2im(image, image2, size):
    #i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(image, scale=(0.08, 1.0), 
     #                                                           ratio=(0.75, 1.3333333333333333))
    i,j,h,w = torchvision.transforms.RandomCrop.get_params(image, output_size = size)
    #image = torchvision.transforms.functional.resized_crop(image, i, j, h, w, size)
    #image2 = torchvision.transforms.functional.resized_crop(image2, i, j, h, w, size)
    image = torchvision.transforms.functional.crop(image, i, j, h, w)
    image2 = torchvision.transforms.functional.crop(image2, i, j, h, w)
    return image, image2

def RandomFlip2im(image,image2):
    p = np.random.choice([0,1])
    q = np.random.choice([0,1])
    if p == 1:
        image = torchvision.transforms.functional.vflip(image)
        image2 = torchvision.transforms.functional.vflip(image2)
    if q == 1:
        image = torchvision.transforms.functional.hflip(image)
        image2 = torchvision.transforms.functional.hflip(image2)
    return image, image2

def RandomRotate2im(image,image2):
    angle = np.random.choice([0,90,180,270])
    image = torchvision.transforms.functional.rotate(image, angle)
    image2 = torchvision.transforms.functional.rotate(image2, angle)
    return image,image2