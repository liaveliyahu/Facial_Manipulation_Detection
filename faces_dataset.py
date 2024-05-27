"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        real_image_len = len(self.real_image_names)

        if index < real_image_len:
            label = 0
            img_name = self.real_image_names[index]
            img_path = os.path.join(self.root_path, 'real', img_name)
        else:
            label = 1
            img_name = self.fake_image_names[index - real_image_len]
            img_path = os.path.join(self.root_path, 'fake', img_name)

        image = Image.open(img_path)
        
        if self.transform != None:
            image = self.transform(image)
        
        return (image, label)
        #return torch.rand((3, 256, 256)), int(torch.randint(0, 2, size=(1, )))

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        real_len = len(self.real_image_names)
        fake_len = len(self.fake_image_names)

        return real_len + fake_len
        #return 100
