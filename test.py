import data.dataset as ds
import torch
import models.gnn as gnn
from torch_geometric.data import Data

NOTEWORTHY_DISTANCE = 200


def get_device():
    return torch.device('cpu')
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = ds.PollutionDatasetGNN('data/jingjinji.csv', 3, 15, True, get_device())
model = gnn.PollutionGNN(12, 15, 1, device=get_device())

graphs = []
code_to_node_index_list = []
for i in range(len(dataset)):
    graphs.append([])
    x, y, codes = dataset[i]
    codes = codes[0]
    edge_index = []
    edge_weight = []
    nodes = torch.zeros((len(codes), x[0].shape[1]))
    city_code_to_index = {}
    for j in range(len(codes)):
        city_code_to_index[codes[j].item()] = j
    code_to_node_index_list.append(city_code_to_index)
    # Build Edges
    for j in range(len(codes)):
        for k in range(len(codes)):
            one, two = codes[j].item(), codes[k].item()
            one_index, two_index = dataset.city_code_dict[one], dataset.city_code_dict[two]
            same = one == two
            distance_ok = dataset.distance_matrix[one_index][two_index] < NOTEWORTHY_DISTANCE
            if not same and distance_ok:
                edge_index.append([j, k])
                edge_weight.append(dataset.distance_matrix[one_index][two_index])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    # For each day
    for j in range(len(x)):
        for k in range(len(codes)):
            curr_code = codes[k].item()
            node_index = city_code_to_index[curr_code]
            nodes[node_index] = x[j][k]
        graphs[-1].append(Data(x=nodes.detach().clone().to(get_device()), edge_index=edge_index, y=y[j].to(get_device()), edge_weight=edge_weight))
        nodes = torch.zeros((len(codes), x[0].shape[1]))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for i in range(20):
    running_loss = 0.0
    last_output = None
    for i in range(len(graphs)):
        optimizer.zero_grad()
        outputs = model(graphs[i])
        loss = criterion(outputs.squeeze(), graphs[i][-1].y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        last_output = outputs.squeeze()
    print(last_output)
    print(running_loss / len(graphs))
