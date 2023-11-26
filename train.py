import torch
import models.independent_gru as gru
import data.dataset as ds


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = ds.PollutionDataset('data/jingjinji.csv', get_device())
total = len(dataset)
train_size = int(0.8 * total)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, total - train_size])
dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
model = gru.IndependentGru(dataset[0][0].shape[1], 128, 2, 0.2, get_device())
optim = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(200):
    running_loss = 0.0
    iter = 0
    for i, (data, target) in enumerate(dataloader):
        optim.zero_grad()
        output, hidden = model(data)
        loss = criterion(output, target[:, -1].unsqueeze(1))
        running_loss += loss.item()
        iter += 1
        loss.backward()
        optim.step()
    print('Epoch: {}, Loss: {}'.format(epoch, running_loss / iter))
    running_loss = 0.0

model.eval()
mae = 0.0
mse = 0.0
with torch.no_grad():
    for i, (data, target) in enumerate(testloader):
        output, hidden = model(data)
        target = target[:, -1].unsqueeze(1)
        mae += torch.sum(torch.abs(output - target))
        mse += torch.sum(torch.pow(output - target, 2))

print('VAL MAE: {}'.format(mae / len(val_set)))
print('VAL MSE: {}'.format(mse / len(val_set)))
mae = 0.0
mse = 0.0
for i, (data, target) in enumerate(dataloader):
    output, hidden = model(data)
    target = target[:, -1].unsqueeze(1)
    mae += torch.sum(torch.abs(output - target))
    mse += torch.sum(torch.pow(output - target, 2))

print('==================')
print('TRAIN MAE: {}'.format(mae / len(train_set)))
print('TRAIN MSE: {}'.format(mse / len(train_set)))
