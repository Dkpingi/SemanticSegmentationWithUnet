import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda')
dtype = torch.float32 
def compute_accuracy(dataloader, model):
    model.eval()
    accuracy = 0
    for i,data in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device=device, dtype=dtype)  
            labels = labels.to(device=device, dtype=torch.long)
            labels = labels[:,0,:,:]
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim = 1)
            mask = torch.eq(predictions,labels).float()
            size = inputs.size()
            size = size[0]*size[2]*size[3]
            accuracy += torch.sum(mask) / size
    #del inputs, labels, data
    return float(accuracy/(i+1))

def compute_IoU(dataloader, model, reduce = True):
    """
    Intersection over Union score for background-foreground prediction.
    The shape of both `predictions` and `targets` should be (batch_size, 1, x_size_image, y_size_image)
    """
    model.eval()
    nb_classes = 3
    IoU_ = torch.zeros(nb_classes)
    for i,data in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device=device, dtype=dtype)  
            labels = labels.to(device=device, dtype=torch.long)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim = 1)
            for cl in range(nb_classes):
                intersection = ((predictions == cl) & (labels == cl)).sum().float()
                union = ((predictions == cl) | (labels == cl)).sum().float()
                if union == 0:
                    IoU_[cl] += 1
                else:
                    IoU_[cl] += intersection/union
    IoU_ /= (i+1)
    if reduce == True:
        return torch.mean(IoU_)
    else:
        return IoU_

def compute_loss(dataloader, model, criterion):
    model.eval()
    running_loss = 0
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
            labels = labels[:,0,:,:]
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item()
    return float(running_loss/i)

def print_loader(dataloader):
    counts = torch.zeros(3)
    for i,data in enumerate(dataloader,0):
        a,b = data
        plt.hist(a.flatten())
        plt.show()
        if i < 10:
            plt.imshow(np.moveaxis(np.asarray(a[0]), 0, 2),vmin=0, vmax=1)
            plt.show()
            plt.imshow(np.asarray(b[0][0]))
            plt.show()
            
def plot_and_eval(dataloader, model):
    device = torch.device('cuda')
    for a,b in dataloader:
        with torch.no_grad():
            a = a.to(device=device, dtype=torch.float32)  
            output = model(a)
            output = torch.argmax(output,dim=1)
            output = output.to(device = 'cpu')
            a = a.to(device='cpu')
            plt.imshow(np.moveaxis(np.asarray(a[0]), 0, 2))
            plt.show()
            plt.imshow(np.asarray(output[0].detach()))
            plt.show()
            plt.imshow(np.asarray(b[0][0]))
            plt.show()