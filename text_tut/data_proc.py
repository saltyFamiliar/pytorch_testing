import torch
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader, random_split


tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = AG_NEWS()
train_dataset, test_dataset = to_map_style_dataset(train_iter), to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])


def yield_tokens(data_iter):
    for _, text in data_iter: yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    from main import device
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
    text_list = torch.cat(text_list).to(device)
    return label_list, text_list, offsets


batch_size = 64
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
train_dataloader = DataLoader(split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)