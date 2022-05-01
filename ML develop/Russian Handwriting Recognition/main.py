import os
import cv2
import time
import math
import torch
import random
import Augmentor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import *
from torch import nn
from PIL import Image
from tqdm import tqdm
from OCR_dataset import *
from label_coder import *
from customCTCLoss import *
from torchvision import models
from textdistance import levenshtein as lev
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU, LeakyReLU
from torch.utils.data import Dataset, sampler
from torch.nn.utils.clip_grad import clip_grad_norm_


def print_epoch_data(epoch, mean_loss, char_error, word_error, time_elapsed, zero_out_losses):
    if epoch == 0:
        print('epoch | mean loss | mean cer | mean wer | time elapsed | warnings')
    epoch_str = str(epoch)
    zero_out_losses_str = str(zero_out_losses)
    if len(epoch_str) < 2:
        epoch_str = '0' + epoch_str
    if len(zero_out_losses_str) < 2:
        zero_out_losses_str = '0' + zero_out_losses_str
    report_line = epoch_str + ' ' * 7 + "%.3f" % mean_loss + ' ' * 7 + "%.3f" % char_error + ' ' * 7 + \
                  "%.3f" % word_error + ' ' * 7 + "%.1f" % float(time_elapsed)
    if zero_out_losses != 0:
        report_line += f'       {zero_out_losses} batch losses skipped due to nan value'
    print(report_line)


def fit(model, optimizer, loss_fn, loader, epochs=64):
    report = []
    coder = LabelCoder(ALPHABET)
    for epoch in range(epochs):
        zero_out_losses = 0
        start_time = time.time()
        model.train()
        outputs = []
        for batch_nb, batch in enumerate(loader):
            optimizer.zero_grad()
            input_, targets = batch['img'], batch['label']
            targets, lengths = coder.encode(targets)
            logits = model(input_.to(DEVICE))
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            targets = targets.view(-1).contiguous()
            loss = loss_fn(logits, targets, pred_sizes, lengths)
            if (torch.zeros(loss.size()) == loss).all():
                zero_out_losses += 1
                continue
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = coder.decode(preds.data, pred_sizes.data, raw=False)
            char_error = sum(
                [lev(batch['label'][i], sim_preds[i]) / max(len(batch['label'][i]), len(sim_preds[i])) for i in
                 range(len(batch['label']))]) / len(batch['label'])
            word_error = 1 - sum([batch['label'][i] == sim_preds[i] for i in range(len(batch['label']))]) / len(
                batch['label'])
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            output = {'loss': abs(loss.item()), 'cer': char_error, 'wer': word_error}
            outputs.append(output)

        if len(outputs) == 0:
            print('ERROR: bad loss, try to decrease learning rate and batch size')
            return None
        end_time = time.time()
        mean_loss = sum([outputs[i]['loss'] for i in range(len(outputs))]) / len(outputs)
        char_error = sum([outputs[i]['cer'] for i in range(len(outputs))]) / len(outputs)
        word_error = sum([outputs[i]['wer'] for i in range(len(outputs))]) / len(outputs)
        report.append({'mean_loss': mean_loss, 'mean_cer': char_error, 'mean_wer': word_error})
        print_epoch_data(epoch, mean_loss, char_error, word_error, end_time - start_time, zero_out_losses)
        if epoch % 4 == 0:
            torch.save(model.state_dict(), PATH_TO_CHECKPOINT + 'checkpoint_epoch_' + str(epoch) + '.pt')
    return report


def evaluate(model, loader):
    coder = LabelCoder(ALPHABET)
    labels, predictions = [], []
    for iteration, batch in enumerate(tqdm(loader)):
        input_, targets = batch['img'].to(DEVICE), batch['label']
        labels.extend(targets)
        targets, _ = coder.encode(targets)
        logits = model(input_)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)
        predictions.extend(sim_preds)
    char_error = sum([lev(labels[i], predictions[i])/max(len(labels[i]), len(predictions[i])) for i in range(len(labels))])/len(labels)
    word_error = 1 - sum([labels[i] == predictions[i] for i in range(len(labels))])/len(labels)
    return {'char_error' : char_error, 'word_error' : word_error}


ALPHABET = " %(),-./0123456789:;?[]«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё-"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
REPORT_ACCURACY = True
PATH_TO_TRAIN_IMGDIR = "Dataset/train/"
PATH_TO_TRAIN_LABELS = "Dataset/train.tsv"
PATH_TO_TEST_IMGDIR = "Dataset/test/"
PATH_TO_TEST_LABELS = "Dataset/test.tsv"
PATH_TO_CHECKPOINT = "checkpoints/"
BATCH_SIZE = 2
APPLY_AUGS = True  # is augmentation applied?
SEED = 41

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


p = Augmentor.Pipeline()
p.random_distortion(probability=1.0, grid_width=8, grid_height=5, magnitude=5)

if APPLY_AUGS:
    transform_list = [
            transforms.Grayscale(1),
            transforms.Resize((64, 256)),
            #transforms.RandomRotation(degrees=(-9, 9), fill=255),
            #transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            p.torch_transform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
else:
    transform_list = None


dataset = OCRdataset(PATH_TO_TRAIN_IMGDIR,
                     PATH_TO_TRAIN_LABELS,
                     transform_list=transform_list)
collator = Collator()
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=4,
                                           collate_fn=collator,
                                           shuffle=True)


examples = []
idx = 0
for batch in train_loader:
    img, true_label = batch['img'], batch['label']
    examples.append([img, true_label])
    idx += 1
    if idx == BATCH_SIZE:
        break
fig = plt.figure(figsize=(10, 10))
rows = int(BATCH_SIZE / 4) + 2
columns = int(BATCH_SIZE / 8) + 2
for j, exp in enumerate(examples):
    fig.add_subplot(rows, columns, j + 1)
    plt.imshow(exp[0][0].permute(2, 1, 0).permute(1, 0, 2))
    plt.title(exp[1][0])


model = Model3(256, len(ALPHABET))

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
loss_fn = CustomCTCLoss()

# print("Start learning..")
# report = fit(model, optimizer, loss_fn, train_loader, epochs=12)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collator, shuffle=True)
# report = fit(model, optimizer, loss_fn, train_loader, epochs=12)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
# report = fit(model, optimizer, loss_fn, train_loader, epochs=12)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=12, collate_fn=collator, shuffle=True)

