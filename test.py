import data.data_utils as du
import data.data_utils_gnn as du_gnn
import data.dataset as ds

dataset = ds.PollutionDatasetGNN('data/jingjinji.csv', 3, 15, True, 'cpu')
for i in range(len(dataset)):
    print(dataset[i][0].shape)
