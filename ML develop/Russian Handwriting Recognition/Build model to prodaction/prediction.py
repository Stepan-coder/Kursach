import torch
import torchvision
import torchvision.transforms as transforms
from main import *
from PIL import Image
from model import *
from OCR import *
from collator import *
from label_coder import *
from typing import Dict, Any


class RussianHandwritingRecognition:
    def __init__(self, model_path):
        self.__ALPHABET = " %(),-./0123456789:;?[]«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё-"
        self.__DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model = Model3(256, len(self.__ALPHABET))
        self.__model.load_state_dict(torch.load(model_path, map_location=torch.device(self.__DEVICE)))
        self.__model.to(self.__DEVICE)
        self.__coder = LabelCoder(self.__ALPHABET)
        self.__collator = Collator()

    def predict(self, images: Dict[str, Any]):
        result = {}
        data = OCRdataset(images=images)
        for batch in torch.utils.data.DataLoader(data, collate_fn=self.__collator):
            result[batch['idx'][0]] = \
                self.__predict(batch['img'])
        return result

    def __predict(self, img):
        logits = self.__model(img.to(self.__DEVICE))
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.__coder.decode(pos.data, pred_sizes.data, raw=False)
        return sim_preds


rhr = RussianHandwritingRecognition(model_path="checkpoints\checkpoint_epoch_8.pt")
print(rhr.predict(images={"со слов": Image.open('123.png')}))


