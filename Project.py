#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.transforms as transforms
import os
import time
from metrics import *
from loss import *
from dataloader import * 
from model import *


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = DataLoaderSegmentation('./data/images/train', transform = None, train = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0, 
                                          collate_fn = Collator(crop_size = (200,200)), pin_memory = True)

valset = DataLoaderSegmentation('./data/images/val', transform = transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                         shuffle=False, num_workers=0)

testset = DataLoaderSegmentation('./data/images/test', transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)
    
#print_loader(valloader)
#compute_loss(valloader, net, criterion)


# In[3]:
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

dtype = torch.float32
device = torch.device('cuda')
exp_name = "exp_name"

cold = True
net = Net()
net = net.to(device=device)
if cold == True:
    IoU = 0.0
else:
    net.load_state_dict(torch.load(f"./saved_models/{exp_name}"))
    IoU = compute_IoU(valloader, net)


net.apply(weights_init)
params = net.parameters()
optimizer = torch.optim.SGD(params, momentum = 0.9, lr = 1e-3, weight_decay = 1e-4)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)
criterion = FocalLoss(gamma = 2.0)


# In[4]:


for epoch in range(30):
    t0 = time.time()
    net.train() # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device=device, dtype=dtype)  
        labels = labels.to(device=device, dtype=torch.long)

        labels = labels[:,0,:,:]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print(epoch)
    lr = [ group['lr'] for group in optimizer.param_groups ]
    print(f'Learning Rate: {lr}')
    print(f'train loss: {running_loss/i}')
    val_loss = compute_loss(valloader, net, criterion)
    print(f'val loss: {val_loss}')
    #scheduler.step(val_loss)
    print(f'val accuracy: {compute_accuracy(valloader, net)}')
    val_IoU = compute_IoU(valloader, net, reduce = False)
    print(f'val IoU per class: {val_IoU}')
    val_IoU = torch.mean(val_IoU)
    print(f'val IoU: {val_IoU}')
    if val_IoU > IoU:
        torch.save(net.state_dict(),f"./saved_models/{exp_name}")
        IoU = val_IoU
    t = time.time() - t0
    print(f'Time for Epoch: {t}')

print('Finished Training')


# In[ ]:


net = Net()
net = net.to(device=device)
net.load_state_dict(torch.load(f"./saved_models/{exp_name}"))
plot_and_eval(valloader, net)
print(compute_IoU(valloader, net))


# In[ ]:


headline = ['ExperimentName', 'train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc', 'train_IoU', 'val_IoU', 'test_IoU']
train_loss = compute_loss(trainloader, net, criterion)
val_loss = compute_loss(valloader, net, criterion)
test_loss = compute_loss(testloader, net, criterion)
train_acc = compute_accuracy(trainloader, net)
val_acc = compute_accuracy(valloader, net)
test_acc = compute_accuracy(testloader, net)
train_IoU = compute_IoU(trainloader, net)
val_IoU = compute_IoU(valloader, net)
test_IoU = compute_IoU(testloader, net)
result = [exp_name, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_IoU, val_IoU, test_IoU]
result = [x if type(x) is str else format(x,'.4f') for x in result]
print(result)
print(compute_IoU(valloader, net, reduce = False))


# In[ ]:


import csv
import os
with open('experiments.csv', mode='a', newline = '') as result_file:
    result_writer = csv.writer(result_file, delimiter=',')
    if os.stat("experiments.csv").st_size == 0:
        result_writer.writerow(headline)
    result_writer.writerow(result)

