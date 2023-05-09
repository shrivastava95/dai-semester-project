import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as tf
from importlib import import_module
import collections


tt =  tf.Compose([
    tf.ToPILImage()
])

n = 2
data =  torch.load(f"noise/noise_cifar10_resnet_pgd_testCE_L{n}.pt")

oks = data['orig']
advs = data['adv']
pred_nos = data['pred_noise']
act_nos = data['actual_noise']
labels = data['label']

img1 = oks[0][0]
img2 = advs[0][0]
ps = pred_nos[0][0]
an = act_nos[0][0]

# print(img1)
img1 = tt(img1)
img2 = tt(img2)
ps = tt(ps)
an = tt(an)

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[16, 4])
axs[0].imshow(img1), axs[0].set_title('clean sample')
axs[1].imshow(img2), axs[1].set_title('attacked sample')
axs[2].imshow(ps), axs[2].set_title('predicted noise')
axs[3].imshow(an), axs[3].set_title('actual noise')
fig.suptitle(f'RESULTS')
plt.show()


# img = Image.fromarray(img1).save("./ok.png")
# img = Image.fromarray(img2).save("./adv.png")
# img = Image.fromarray(ps).save("./ps.png")
# img = Image.fromarray(an).save("./an.png")

import os
import sys
from importlib import import_module
##################
modelpath = os.path.join(os.path.abspath('../Exps'),'sample')
# train_data = np.load(os.path.join(modelpath,'train_split.npy'))
# val_data = np.load(os.path.join(modelpath,'val_split.npy'))
# with open(os.path.join(modelpath,'train_attack.txt'),'r') as f:
#     train_attack = f.readlines()
# train_attack = [attack.split(' ')[0].split(',')[0].split('\n')[0] for attack in train_attack]
sys.path.append(modelpath)
model = import_module('model')
config, net = model.get_model()
############
# config, net = model.get_model()
net = net.net#!
state_dict = torch.load(f'noise/noise_cifar10_resnet_pgd_trainCE_L{n}.pt')['net.state_dict()']
new_state_dict = collections.OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net = net.cuda()


orig, adv, label = oks[0], advs[0], labels[0]
print(orig.shape, adv.shape, label.shape)
adv = adv.cuda()
label = label.cuda()

####################################
denoised = net.denoise_fn(adv)
_, features, scores = net(denoised)
features.retain_grad()

for class_id in range(10):
    loss = torch.Tensor([0])[0]
    loss.requires_grad = True
    loss = loss + scores[0][class_id]
    loss.backward(retain_graph=True)


    alphas = torch.mean(features.grad, dim=(2, 3)).unsqueeze(2).unsqueeze(2)
    print(features.shape, alphas.shape)
    maps = torch.sum(tf.Resize([224, 224])(features * alphas), dim=1)
    maps = torch.nn.ReLU()(maps)

    
    plt.imshow(maps[0].clone().cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.title(f'{scores[0][class_id]}')
    plt.show()