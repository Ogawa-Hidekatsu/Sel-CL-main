import random
import torchvision as tv
from typing import Callable, Optional
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img

class CIFAR10Poison(tv.datasets.CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        sample_indexes=None,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

        # self.root = os.path.expanduser(args.train_root)
        # self.transform = transform
        # self.target_transform = target_transform
        # self.train = train  # Training set or validation set
        #
        # self.args = args
        # if sample_indexes is not None:
        #     self.data = self.data[sample_indexes]
        #     self.targets = list(np.asarray(self.targets)[sample_indexes])
        #
        # self.num_classes = self.args.num_classes
        # self.in_index = []
        # self.out_index = []
        # self.noisy_indexes = []
        # self.clean_indexes = []
        # self.clean_labels = []
        self.noisy_labels = []
        # self.out_data = []
        # self.out_labels = []
        # self.soft_labels = []
        # self.labelsNoisyOriginal = []
        # self._num = []
        # self._count = 1
        # self.prediction = []
        # self.confusion_matrix_in = np.array([])
        # self.confusion_matrix_out = np.array([])
        # self.labeled_idx = []
        # self.unlabeled_idx = []
        #
        #
        # # From in ou split function:
        # self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        # self._num = int(len(self.targets) * self.args.noise_ratio)

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                img = self.trigger_handler.put_trigger(img)

            if self.transform is not None:
                img = self.transform(img)

            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return img, target, index
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                img = self.trigger_handler.put_trigger(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

