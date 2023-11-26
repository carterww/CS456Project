import torch
import torch.nn as nn
import torch.nn.functional as F


class IndependentGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0,
                 device=None):
        super(IndependentGru, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout,
                          bidirectional=False, device=device, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.add_module('gru', self.gru)
        self.add_module('fc', self.fc)
        self.fc = self.fc.to(device)

    def forward(self, input, hidden=None):
        # input: (seq_len, batch, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(input, hidden)
        output = self.fc(output[:, -1, :])
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def eval(self):
        self.gru.eval()
        self.fc.eval()
        return self

    def train(self):
        self.gru.train()
        self.fc.train()
        return self

    def children(self):
        return self.gru, self.fc
