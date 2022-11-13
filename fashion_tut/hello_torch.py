import torch
from torch import nn

import fashion_tut.brain as brain
from fashion_tut.brain import train, test

import fashion_tut.data as data


device = torch.device("mps")

# Create the NN
model = brain.NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data.train_dataloader, model, loss_fn, optimizer, device)
    test(data.test_dataloader, model, loss_fn, device)

x, y = data.test_data[0][0].to(device), data.test_data[0][1]
with torch.no_grad():
    model.eval()
    pred = model(x)
    predicted, actual = data.classes[pred[0].argmax(0)], data.classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
