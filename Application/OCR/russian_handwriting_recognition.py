import torch
from PIL import Image
from OCR.model import *
from OCR.collator import *
from OCR.label_coder import *
from OCR.image_prepare import *
from typing import Dict, Any


class RussianHandwritingRecognition:
    """
    This is the main class of the module.
    It implements loading a model from a file and recognizing Russian handwritten text from a 256 x 64 image.
    """
    def __init__(self, path_to_model: str) -> None:
        """
        :param path_to_model: Path to model weights config file
        """
        self.__ALPHABET = " %(),-./0123456789:;?[]«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё-"
        self.__DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model = Model3(256, len(self.__ALPHABET))
        self.__model.load_state_dict(torch.load(path_to_model, map_location=torch.device(self.__DEVICE)))
        self.__model.to(self.__DEVICE)
        self.__coder = LabelCoder(self.__ALPHABET)
        self.__collator = Collator()

    def predict(self, images: Dict[str, Image.Image]) -> Dict[str, str]:
        """
        This method implements text recognition from an image
        :param images: A dictionary in which the key is the id or name of the image, the value is the image itself
        return: A dictionary in which the key is the id or name of the image,
                the value is the text recognized from the image
        """
        result = {}
        data = ImagePrepare(images=images)
        for batch in torch.utils.data.DataLoader(data, collate_fn=self.__collator):
            result[batch['idx'][0]] = self.__predict(batch['img'])
        return result

    def __predict(self, tensor: torch.Tensor) -> str:
        """
        This is an internal method that implements the transformation of input images into tensors and
        performs text recognition from them.
        """
        logits = self.__model(tensor.to(self.__DEVICE))
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        return self.__coder.decode(pos.data, pred_sizes.data, raw=False)



