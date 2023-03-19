import os

import torchvision
import torchvision.transforms as transforms

from torchvision import datasets

train_dataset = datasets.cifar.CIFAR100(root='../../data/cifar100', train=True, transform=None, download=True)
test_dataset = datasets.cifar.CIFAR100(root='../../data/cifar100', train=False, transform=None, download=True)


# f = os.listdir('../../../wangbo/data/imagenet/')
# f = os.listdir('../../data/Imagenet/')
# print(f)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# augmentation=transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])
# train_dataset=torchvision.datasets.ImageFolder('../../../wangbo/data/imagenet/train/',transform=augmentation)

# trainset = torchvision.datasets.ImageNet('../../data/ImageNet/train/', split='train',
#                                      target_transform=None, download=True)
# valset = torchvision.datasets.ImageNet('../../data/ImageNet/val/', split='val',
#                                    target_transform=None, download=True)