from torch_geometric.nn import GCNConv, LayerNorm
from torch_geometric.data import Data
import torch.nn.functional as F
import torch


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=None):
        super(GNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.conv1 = GCNConv(input_size, hidden_size, device=device)
        self.conv2 = GCNConv(hidden_size, output_size, device=device)
        self.layer_norm = LayerNorm(output_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.layer_norm(x)
        print(x)
        return x

    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())


class GRUDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device=None):
        super(GRUDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, device=device)
        self.fc = torch.nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x):
        # x is shape (sequence_length, city_count, hidden_size)
        x, _ = self.gru(x)
        x = self.fc(x[-1])
        return x

    def parameters(self):
        return list(self.gru.parameters()) + list(self.fc.parameters())


class PollutionGNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device=None):
        super(PollutionGNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.encoder = GNNEncoder(input_size, hidden_size, hidden_size, device=device)
        self.decoder = GRUDecoder(hidden_size, hidden_size, output_size, num_layers=num_layers, device=device)

    def forward(self, graph_list):
        x = torch.zeros((len(graph_list), graph_list[0].x.shape[0], self.hidden_size), device=self.device)
        for i in range(len(graph_list)):
            x[i] = self.encoder(graph_list[i])
        x = self.decoder(x)
        return x

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def eval(self):
        return self.train(False)
