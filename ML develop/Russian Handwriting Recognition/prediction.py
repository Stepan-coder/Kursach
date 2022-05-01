import torch
from main import *
from model import *
import matplotlib.pyplot as plt


def predict(model, img):
    logits = model(img.to(DEVICE))
    logits = logits.contiguous().cpu()
    T, B, H = logits.size()
    pred_sizes = torch.LongTensor([T for i in range(B)])
    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)
    sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)
    return sim_preds



model1 = Model3(256, len(ALPHABET))
model1.load_state_dict(torch.load("checkpoints/checkpoint_epoch_8.pt", map_location=torch.device('cpu')))
model1.to(DEVICE)

test_dataset = OCRdataset(PATH_TO_TEST_IMGDIR, PATH_TO_TEST_LABELS)
collator = Collator()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collator)

examples = []
idx = 0
coder = LabelCoder(ALPHABET)
for batch in test_loader:
    img, true_label = batch['img'], batch['label']
    pred_label = predict(model1, img)
    examples.append([img, true_label, pred_label])
    idx += 1
    if idx == 99:
        break

# fig = plt.figure(figsize=(10, 10))
rows = int(9 / 4) + 2
columns = int(9 / 8) + 2
for j, exp in enumerate(examples):
    plt.imshow(exp[0][0].permute(2, 1, 0).permute(1, 0, 2))
    print(j, 'true:' + exp[1][0] + 'pred:' + exp[2][0])