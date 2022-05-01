import torch
from main import *
from model import *
import matplotlib.pyplot as plt

model1 = Model3(256, len(ALPHABET))
model1.load_state_dict(torch.load("checkpoints/checkpoint_epoch_8.pt", map_location=torch.device('cpu')))
model1.to(DEVICE)

test_dataset = OCRdataset(PATH_TO_TEST_IMGDIR, PATH_TO_TEST_LABELS)
collator = Collator()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collator)
print(evaluate(model1, test_loader))