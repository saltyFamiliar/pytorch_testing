import time
import torch

device = torch.device("cpu")
epochs = 1
lr = 5

from data_proc import train_iter, vocab
from brain import TextClassificationModel
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

from data_proc import train_dataloader, valid_dataloader, test_dataloader
from learning import train, evaluate
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader, model, optimizer, epoch)
    accu_val = evaluate(valid_dataloader, model)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)


ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "The latest information from the U.S. retail giant put the figure at\
     a massive 2.30 million. Not even the behemoth that is Amazon comes close, \
     despite being in second place with a 1.61 million-strong workforce. As \
     Statista's Martin Armstrong shows in the inforgraphic below though, there's \
     one sector that apparently needs even more manpower than retail, and that's defense."
from data_proc import text_pipeline
print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])