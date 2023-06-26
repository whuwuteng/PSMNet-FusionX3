import torch
import torchvision.transforms as transforms
import random

__imagenet_stats_kitti = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

__imagenet_stats_gray_kitti = {'mean': [0.485],
                         'std': [0.229]}

#use kitti for all
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

__imagenet_stats_gray = {'mean': [0.354957],
                         'std': [0.169372]}

#IARPA
#__imagenet_stats = {'mean': [0.229213, 0.229213, 0.229213],
#                    'std': [0.109629, 0.109629, 0.109629]}

#__imagenet_stats_all = {'mean': [0.352490, 0.349178, 0.339951],
#                    'std': [0.222547, 0.212384, 0.206278]}

# UMBRA
#__imagenet_stats = {'mean': [0.162272, 0.162284, 0.159792],
#                   'std': [0.125941, 0.121415, 0.117070]}

#JAX
#__imagenet_stats = {'mean': [0.516369, 0.521606, 0.518110],
#                    'std': [0.220612, 0.206167, 0.203684]}

# Dublin
#__imagenet_stats= {'mean': [0.427924, 0.457238, 0.479585],
#                    'std': [0.202187, 0.203649, 0.192346]}

#Enschede
#__imagenet_stats= {'mean': [0.437332, 0.432745, 0.407168],
#                    'std': [0.237859, 0.214953, 0.197412]}

# keep B/H ratio the origin value
__imagenet_stats_bh = {'mean': [0.352490, 0.349178, 0.339951, 0],
                    'std': [0.222547, 0.212384, 0.206278, 1]}                   
__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True, channle=3):
    if normalize is None :
        normalize = __imagenet_stats

        if channle == 1:
            normalize = __imagenet_stats_gray

    if input_size is None:
        input_size = 256
    
    #print('[***] normal= ', normalize)

    if augment:
        return inception_color_preproccess(input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
