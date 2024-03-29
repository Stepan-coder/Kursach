import torchvision.transforms as transforms
from typing import Dict, Any
from torch.utils.data import Dataset, sampler


class ImagePrepare(Dataset):
    """
    This class is necessary for the preliminary preparation of images for the operation of the neural network.
    """
    def __init__(self, images: Dict[str, Any]):
        """
        :param images: A dictionary in which the key is the id or the name of the image, the value is the image itself.
        """
        super(ImagePrepare, self).__init__()
        self.images = images
        self.images_ids = list(images.keys())
        self.transform = transforms.Compose([transforms.Grayscale(1),
                                             transforms.Resize((64, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        assert index <= len(self), 'Index out of range'
        return {'idx': self.images_ids[index],
                'img': self.transform(self.images[self.images_ids[index]])}