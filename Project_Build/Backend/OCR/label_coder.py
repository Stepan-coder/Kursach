import torch


class LabelCoder(object):
    def __init__(self, alphabet, ignore_case=False):
        self.alphabet = alphabet
        self.char2idx = {}
        for i, char in enumerate(alphabet):
            self.char2idx[char] = i + 1
        self.char2idx[''] = 0

    def encode(self, text: str) -> tuple:
        length = []
        result = []
        for item in text:
            length.append(len(item))
            for char in item:
                result.append(self.char2idx[char] if char in self.char2idx else 0)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                texts.append(self.decode(t[index:index + length[i]], torch.IntTensor([length[i]]), raw=raw))
                index += length[i]
            return texts