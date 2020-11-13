import torch
import torch.nn as nn
import torch.nn.functional as F

def NormConv2d(f_in,f_out,k):
    return nn.Sequential(nn.GroupNorm(max(int(f_in/16),1),f_in),nn.Conv2d(f_in, f_out, k, padding = int((k-1)/2)),nn.ReLU())

class ResBlock(nn.Module):
    def __init__(self, f_in, f_out, nlayers):
      super(ResBlock, self).__init__()
      self.nlayers = nlayers
      layerlist = nn.ModuleList()
      layerlist.append(NormConv2d(f_in, f_out, 3))
      for i in range(1,nlayers):
          layerlist.append(NormConv2d(f_out, f_out, 3))
      self.layerlist = layerlist
          
    def forward(self, x):
      N = len(self.layerlist)
      
      x = self.layerlist[0](x)
      
      v = torch.zeros((N,*x.size())).cuda()
      
      for i in range(1,N):
          v[i] = x
          v[0] = self.layerlist[i](x)
          x = torch.mean(v[:i+1], dim = 0)
      return x
        
      
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      f_low = 32
      self.block1 = ResBlock(3,f_low,5)
    
      self.pool1 = nn.MaxPool2d(2, return_indices = True)
      self.block2 = ResBlock(f_low,2*f_low,5)
        
      self.pool2 = nn.MaxPool2d(2, return_indices = True)
      self.block3 = ResBlock(2*f_low,4*f_low,5)
        
      self.pool3 = nn.MaxPool2d(2, return_indices = True)
      self.block4 = ResBlock(4*f_low,4*f_low,5)

      self.upsample1 = nn.MaxUnpool2d(2)
      self.block5 = ResBlock(4*f_low,2*f_low,5)
    
      self.upsample2 = nn.MaxUnpool2d(2)
      self.block6 = ResBlock(2*f_low,f_low,5)
        
      self.upsample3 = nn.MaxUnpool2d(2)
      self.block7 = ResBlock(f_low,f_low,5)
      
      self.convlast = nn.Conv2d(f_low,3,3,padding = 1)
          
        

    # x represents our data
    def forward(self, x):
      x1 = self.block1(x)

      x,p1 = self.pool1(x1)
      x2 = self.block2(x)
        
      x,p2 = self.pool2(x2)
      x3 = self.block3(x)

      x,p3 = self.pool3(x3)
      x = self.block4(x)
    
      x = self.upsample1(x, p3, output_size=x3.size())
      x = x + x3 #torch.cat((x,x3),dim = 1)
      x = self.block5(x)
    
      
      x = self.upsample2(x, p2, output_size=x2.size())
      x = x + x2 #torch.cat((x,x2),dim = 1)
      x = self.block6(x)
    
      x = self.upsample3(x, p1, output_size=x1.size())
      x = x + x1 # torch.cat((x,x1),dim = 1)
      x = self.block7(x)
      
      x = self.convlast(x)
      
      return x