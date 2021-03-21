import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_GROUP_PATHS = {
    'train': 'train',
    'test': 'test',
    'val': 'val',
}
DEFAULT_CLASS_PATHS = {
    'covid':     (os.path.join('infected', 'covid'), ),
    'non-covid': (os.path.join('infected', 'non-covid'), ),
    'normal':    ('normal', ),
    'infected':  (os.path.join('infected', 'covid'),
                  os.path.join('infected', 'non-covid'))
}


class LungDataset(Dataset):
    def __init__(self,
                 dataset_path='./dataset', group='train', classes=('covid', 'non-covid', 'normal'),
                 group_paths=DEFAULT_GROUP_PATHS, class_paths=DEFAULT_CLASS_PATHS):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        super().__init__()

        # Three classes considered here depend on parameters
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # The dataset consists only of training/test/val images
        if group not in group_paths:
            raise ValueError(f'unknown dataset group `{group}`: '
                             f'group should be one of {group_paths.keys()}')
        self.group = group

        # Data preprocessing and augmentation
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        if self.group == 'train':
            self.transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomRotation(5, resample=Image.BILINEAR),
                transforms.RandomResizedCrop((150, 150), (0.7, 1.0)),
                transforms.ToTensor(),
            ])

        # Index images
        if not os.path.exists(dataset_path):
            raise ValueError(f'given dataset path `{dataset_path}` does not exist')

        self.image_paths = []
        self.image_classes = []
        self.class_paths = {}
        self.class_sizes = {}

        for data_class in self.classes:
            if data_class not in class_paths:
                raise ValueError(
                    f'unknown dataset class `{data_class}`: '
                    f'class should be one of {class_paths.keys()}')

            self.class_paths[data_class] = class_paths[data_class]
            self.class_sizes[data_class] = 0

            for class_path in class_paths[data_class]:
                class_path = os.path.join(
                    dataset_path, group_paths[group], class_path)
                if not os.path.exists(class_path):
                    raise ValueError(f'dataset class path `{class_path}` does not exist')

                for filename in os.listdir(class_path):
                    if not str(filename).endswith('.jpg'):
                        continue
                    self.image_paths.append(os.path.join(class_path, filename))
                    self.image_classes.append(self.class_to_idx[data_class])
                    self.class_sizes[data_class] += 1

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        # Generate description
        msg = "This is the " + self.group + " dataset of the Lung Dataset"
        msg += " used for the Small Project in 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(len(self))
        msg += "of size 150 by 150.\n"
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.class_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.class_sizes[key])
        print(msg)

    def open_img(self, image_path):
        """
        Opens image with specified parameters.
        Parameters:
        - image_path path of the image.

        Returns loaded image as a preprocessed tensor.
        """
        im = Image.open(image_path)
        im_t = self.transforms(im)
        return im_t

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image in torch tensor and its label as class id
        """
        return self.open_img(self.image_paths[index]), self.image_classes[index]
