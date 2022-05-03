import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from label_coder import *
from torch.utils.data import Dataset, sampler


class OCRdataset(Dataset):
    def __init__(self, path_to_imgdir: str, path_to_labels: str, transform_list=None):
        super(OCRdataset, self).__init__()
        self.imgdir = path_to_imgdir
        df = pd.read_csv(path_to_labels, sep='\t', names=['image_name', 'label'])
        self.image2label = [(self.imgdir + image, label) for image, label in zip(df['image_name'], df['label'])]
        if transform_list == None:
            transform_list = [transforms.Grayscale(1),
                              transforms.Resize((64, 256)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)
        self.collate_fn = Collator()

    def __len__(self):
        return len(self.image2label)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_path, label = self.image2label[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return {'idx': index, 'img': img, 'label': label}