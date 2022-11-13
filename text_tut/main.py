import torch
import time

from brain import TextClassificationModel
from configs import device
from learning import train, evaluate
import data_proc


epochs = 1
lr = 5
emsize = 64
model = TextClassificationModel(data_proc.vocab_size, emsize, data_proc.num_class).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(data_proc.train_dataloader, model, optimizer, epoch)
    accu_val = evaluate(data_proc.valid_dataloader, model)
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

ex_text_str = "The latest information from the U.S. retail giant put the figure at\
     a massive 2.30 million. Not even the behemoth that is Amazon comes close, \
     despite being in second place with a 1.61 million-strong workforce. As \
     Statista's Martin Armstrong shows in the inforgraphic below though, there's \
     one sector that apparently needs even more manpower than retail, and that's defense."

category = data_proc.ag_news_label[predict(ex_text_str, data_proc.text_pipeline)]
print(f"This is a {category} news")