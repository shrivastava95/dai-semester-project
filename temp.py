import torch
from torchsummary import summary 
from model import *

config, net = get_model()
net = net.net
net = net.cuda()

print(summary(net, (3, 32, 32)))