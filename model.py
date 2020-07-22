import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1a = nn.Conv2d(3, 32, 3, 1, padding = 1)
      self.norm1a = nn.GroupNorm(2,32)#nn.BatchNorm2d(32)
      self.conv1b = nn.Conv2d(32, 32, 3, 1, padding = 1)
      self.norm1b = nn.GroupNorm(2,32)#nn.BatchNorm2d(32)
      self.conv1c = nn.Conv2d(32, 32, 3, 1, padding = 1)
    
      self.pool1 = nn.MaxPool2d(2, return_indices = True)
      self.poolnorm1 = nn.GroupNorm(2,32)
      self.conv2a = nn.Conv2d(32, 128, 3, 1, padding = 1)
      self.norm2a = nn.GroupNorm(8,128)#nn.BatchNorm2d(64)
      self.conv2b = nn.Conv2d(128, 128, 3, 1, padding = 1)
      self.norm2b = nn.GroupNorm(8,128)#nn.BatchNorm2d(64)
      self.conv2c = nn.Conv2d(128, 128, 3, 1, padding = 1)
        
      self.pool2 = nn.MaxPool2d(2, return_indices = True)
      self.poolnorm2 = nn.GroupNorm(4,128)
      self.conv3a = nn.Conv2d(128, 256, 3, 1, padding = 1)
      self.norm3a = nn.GroupNorm(16,256)#nn.BatchNorm2d(64)
      self.conv3b = nn.Conv2d(256, 256, 3, 1, padding = 1)
      self.norm3b = nn.GroupNorm(16,256)#nn.BatchNorm2d(64)
      self.conv3c = nn.Conv2d(256, 512, 3, 1, padding = 1)
        
      self.pool3 = nn.MaxPool2d(2, return_indices = True)
      self.poolnorm3 = nn.GroupNorm(32,512)
      self.conv4a = nn.Conv2d(512, 512, 3, 1, padding = 1)#, padding = 2, dilation = 2)
      self.norm4a = nn.GroupNorm(32,512)#nn.BatchNorm2d(64)
      self.conv4b = nn.Conv2d(512, 512, 3, 1, padding = 1)#, padding = 5, dilation = 5)
      self.norm4b = nn.GroupNorm(32,512)#nn.BatchNorm2d(64)
      self.conv4e = nn.Conv2d(512, 512, 3, 1, padding = 1)#, padding = 20, dilation = 20)
      self.norm4e = nn.GroupNorm(32,512)#nn.BatchNorm2d(64)
      self.conv4f = nn.Conv2d(512, 512, 3, 1, padding = 1)#, padding = 32, dilation = 32)

      self.upsample1 = nn.MaxUnpool2d(2)
      self.upsamplenorm1 = nn.GroupNorm(64,1024)
      self.conv5a = nn.Conv2d(1024, 256, 3, 1, padding = 1)
      self.norm5a = nn.GroupNorm(16,256)#nn.BatchNorm2d(64)
      self.conv5b = nn.Conv2d(256, 256, 3, 1, padding = 1)
      self.norm5b = nn.GroupNorm(16,256)#nn.BatchNorm2d(64)
      self.conv5c = nn.Conv2d(256, 128 , 3, 1, padding = 1)
    
      self.upsample2 = nn.MaxUnpool2d(2)
      self.upsamplenorm2 = nn.GroupNorm(16,256)
      self.conv6a = nn.Conv2d(256, 128, 3, 1, padding = 1)
      self.norm6a = nn.GroupNorm(8,128)#nn.BatchNorm2d(64)
      self.conv6b = nn.Conv2d(128, 128, 3, 1, padding = 1)
      self.norm6b = nn.GroupNorm(8,128)#nn.BatchNorm2d(64)
      self.conv6c = nn.Conv2d(128, 32 , 3, 1, padding = 1)
        
      self.upsample3 = nn.MaxUnpool2d(2)
      self.upsamplenorm3 = nn.GroupNorm(4,64)
      self.conv7a = nn.Conv2d(64, 32, 3, 1, padding = 1)
      self.norm7a = nn.GroupNorm(2,32)#nn.BatchNorm2d(16)
      self.conv7b = nn.Conv2d(32, 32, 3, 1, padding = 1)
      self.norm7b = nn.GroupNorm(2,32)#nn.BatchNorm2d(16)
      self.conv7c = nn.Conv2d(32, 3, 3, 1, padding = 1)

    # x represents our data
    def forward(self, x):
      x = F.relu(self.conv1a(x))
      x = self.norm1a(x)
      x = F.relu(self.conv1b(x))
      x = self.norm1b(x)
      x1 = F.relu(self.conv1c(x))

      x,p1 = self.pool1(x1)
      x = self.poolnorm1(x)
      x = F.relu(self.conv2a(x)) 
      x = self.norm2a(x)
      x = F.relu(self.conv2b(x))
      x = self.norm2a(x)
      x2 = F.relu(self.conv2c(x))
        
      x,p2 = self.pool2(x2)
      x = self.poolnorm2(x)
      x = F.relu(self.conv3a(x)) 
      x = self.norm3a(x)
      x = F.relu(self.conv3b(x))
      x = self.norm3b(x)
      x3 = F.relu(self.conv3c(x))


      x,p3 = self.pool3(x3)
      x = self.poolnorm3(x)
      x = F.relu(self.conv4a(x))
      x = self.norm4a(x)
      x = F.relu(self.conv4b(x))
      x = self.norm4b(x)
      x = F.relu(self.conv4e(x))
      x = self.norm4e(x)
      x = F.relu(self.conv4f(x))
    
      x = self.upsample1(x, p3, output_size=x3.size())
        
      x = torch.cat((x,x3),dim = 1)
      x = self.upsamplenorm1(x)
      x = F.relu(self.conv5a(x))
      x = self.norm5a(x)
      x = F.relu(self.conv5b(x))
      x = self.norm5b(x)
      x = F.relu(self.conv5c(x))
    
      
      x = self.upsample2(x, p2, output_size=x2.size())

      x = torch.cat((x,x2),dim = 1)
      x = self.upsamplenorm2(x)
      x = F.relu(self.conv6a(x))
      x = self.norm6a(x)
      x = F.relu(self.conv6b(x))
      x = self.norm6b(x)
      x = F.relu(self.conv6c(x))
    
      x = self.upsample3(x, p1, output_size=x1.size())
        
      x = torch.cat((x,x1),dim = 1)
      x = self.upsamplenorm3(x)
      x = F.relu(self.conv7a(x))
      x = self.norm7a(x)
      x = F.relu(self.conv7b(x))
      x = self.norm7b(x)
      x = self.conv7c(x)
      return x