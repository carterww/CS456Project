import torch
import models.independent_gru as gru
import data.dataset as ds
import plotting as train_plot
from time import sleep


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, model, optimizer, criterion, trainloader, valloader, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.trained_model = None
        self.train_plot = train_plot.TrainPlot('Training Loss', 'Epoch', 'MSE Loss')

    def train(self, epoch):
        best_val_mae = int(1e9)
        for step in range(epoch):
            self.model.train()
            running_loss = 0.0
            iter = 0
            for i, (data, target) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target[:, -1].unsqueeze(1))
                running_loss += loss.item()
                iter += 1
                loss.backward()
                self.optimizer.step()
            print('Epoch: {}, Loss: {}'.format(step + 1, running_loss / iter))
            _, _, loss = self.valuate(True)
            if loss < best_val_mae:
                best_val_mae = loss
                self.trained_model = self.model.state_dict()
            self.train_plot.update(step + 1, running_loss / iter, 'train')
            self.train_plot.update(step + 1, loss, 'test')

    def valuate(self, quiet=False, plot=False):
        if plot:
            self.train_plot.close()
        mae = 0.0
        mse = 0.0
        loss = 0.0
        iter = 0
        if plot:
            y_pred = []
            y_real = []
        with torch.no_grad():
            self.model.eval()
            for i, (data, target) in enumerate(self.valloader):
                output = self.model(data)
                target = target[:, -1].unsqueeze(1)
                if plot:
                    y_pred += output.squeeze(1).to('cpu').tolist()
                    y_real += target.squeeze(1).to('cpu').tolist()
                loss += self.criterion(output, target).item()
                iter += 1
                mae += torch.sum(torch.abs(output - target))
                mse += torch.sum(torch.pow(output - target, 2))
        if not quiet:
            print('VAL MAE: {}'.format(mae / len(self.valloader.dataset)))
            print('VAL MSE: {}'.format(mse / len(self.valloader.dataset)))
        if plot:
            self.val_plot = train_plot.ValuationPlot('Valuation', 'Sample Index', 'PM2.5 Residual', y_pred, y_real, False)
        return mae / len(self.valloader.dataset), mse / len(self.valloader.dataset), loss / iter

    def best_model(self):
        return self.trained_model


def split_dataset(dataset, train_size):
    total = len(dataset)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, total - train_size])
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    return train_set, test_set, dataloader, testloader


def init_model(dict_path=None):
    model = gru.IndependentGru(dataset[0][0].shape[1], 128, 2, 0.2, get_device())
    if dict_path is not None:
        model.load_state_dict(torch.load(dict_path))
    return model


dataset = ds.PollutionDataset('data/jingjinji.csv', get_device())
train_set, test_set, dataloader, testloader = split_dataset(dataset, int(0.8 * len(dataset)))
model = init_model()
optim = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

trainer = Trainer(model, optim, criterion, dataloader, testloader, get_device())
trainer.train(200)
if trainer.best_model() is not None:
    torch.save(trainer.best_model(), 'models/trained_model.pth')
# Wait on user input to close the plot
trainer.train_plot.show()
# trainer.train_plot.save('plots/loss.png')
trainer.valuate(False, True)
