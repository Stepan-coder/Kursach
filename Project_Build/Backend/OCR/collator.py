import torch


class Collator(object):
    def __call__(self, batch):
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch),
                           batch[0]['img'].shape[0],
                           batch[0]['img'].shape[1],
                           max(width)],
                           dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                pass
        item = {'img': imgs, 'idx': indexes}
        if 'label' in batch[0].keys():
            item['label'] = [item['label'] for item in batch]
        return item