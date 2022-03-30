from torchvision import datasets, transforms
from PIL import Image
from .gaussian_blur import GaussianBlur

# train_transform,input 224*224
train_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# test_transform,input 224*224
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
