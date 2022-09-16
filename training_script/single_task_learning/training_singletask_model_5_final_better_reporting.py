#**********************************************************************************
# Copyright (c) 2020 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt.svt/private/graph_neural_network_for_fuel_ignition_quality.git
#
#*********************************************************************************

import os.path as osp
import os
import sys
sys.path.insert(0,'../..')

import argparse
import torch
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from smiles_to_molecular_graphs.read_in_singletask import FUELNUMBERS
from k_gnn import GraphConv, DataLoader, avg_pool
from k_gnn import TwoLocal
from src import EarlyStopping

from datetime import datetime
import matplotlib.pyplot as plt
import csv


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 2 nodes.
class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        print(x)
        data.x = data.x[:, :3]   # only consider atom types (H,C,O) of atom features vectors for determining isomorphic type in kgnn
        data = TwoLocal()(data)   # create higher-dimensional graph (2)
        data.x = x
        return data

# Hyperparameter
data_scaling = 'standard'   # only standardization to mean = 0 and standard deviation = 1 is implemented yet
parser = argparse.ArgumentParser()
parser.add_argument('--target', default='TB_simple')   # define target property for single task learning (DCN, MON, or RON)
parser.add_argument('--epochs', default=300)   # number of epochs
parser.add_argument('--early_stopping_patience', default=50)   # number of epochs until early stopping
parser.add_argument('--pooling', default='add')   # pooling function
parser.add_argument('--conv', default=3)   # number of graph convolutional layers in 1-GNN
parser.add_argument('--conv2', default=2)   # number of graph convolutional layers in 2-GNN
parser.add_argument('--dim', default=64)   # size of hidden node states
parser.add_argument('--lrate', default=0.001)   # initial learning rate
parser.add_argument('--lrfactor', default=0.8)   # decreasing factor for learning rate
parser.add_argument('--lrpatience', default=3)   # number of consecutive epochs without model improvement after which learning rate is decreased
parser.add_argument('--model_type', default=0)   # choose model
parser.add_argument('--num_layers', default=1)   # choose number of layers
parser.add_argument('--num_nodes', default=16)   # choose number of nodes in list per layer. 
parser.add_argument('--gnn', default='GCN')   # choose number of nodes in list per layer. 


args = parser.parse_args()
target_name = str(args.target)
epochs = int(args.epochs)
early_stopping_patience = int(args.early_stopping_patience)
pool_type = str(args.pooling)
conv_type = int(args.conv)
conv_type2 = int(args.conv2)
dim = int(args.dim)
lrate = float(args.lrate)
lrfactor = float(args.lrfactor)
lrpatience = int(args.lrpatience)
model_type_int = int(args.model_type)
num_layers = int(args.num_layers)
num_nodes = int(args.num_nodes)
gnn = str(args.gnn)


print(num_layers)
print(num_nodes)

# Dataset splitting
val_percent = 0.15

## Define whether an external test set (predefined test data) is used or whether test data should be randomly selected from the training dataset (excluded from training data)
ext_test = True
if ext_test is False:
    test_percent = 0.15
else:
    test_percent = 0   # external test data
    val_percent = 0.15/0.85   # it is assumed that the size of the external test dataset is 15% of size of training + test dataset, so that validation set size equals external test set size


print('---- Target: Training of singletask model for ' + target_name + ', Pooling: ADD ----')

## Adjust path for dataset
dataset_folder_name = target_name + '/Default/'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../Data/' + dataset_folder_name)

dataset = FUELNUMBERS(
    path + 'Train/',
    pre_transform=MyPreTransform(),
    pre_filter=MyFilter())

if ext_test is True:
    ext_test_dataset = FUELNUMBERS(
        path + 'Test/',
        pre_transform=MyPreTransform(),
        pre_filter=MyFilter())

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = int(dataset.data.iso_type_2.max().item() + 1)
dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

if ext_test is True:
    ext_test_dataset.data.iso_type_2 = torch.unique(ext_test_dataset.data.iso_type_2, True, True)[1]
    ## make sure num_i_2 equals ext_num_i_2, otherwise add isomorphism types (see predict_DCN_MON_RON_single_mol.py)
    #ext_num_i_2 = int(ext_test_dataset.data.iso_type_2.max().item() + 1)
    ext_test_dataset.data.iso_type_2 = F.one_hot(ext_test_dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Operating on following hardware: ' + str(device))

## Shuffle training data
dataset = dataset.shuffle()

## Calculate mean for target data points
norm_percent = test_percent
norm_length = int(len(dataset) * norm_percent)
mean = torch.as_tensor(dataset.data.y[norm_length:], dtype=torch.float).mean()
std = torch.as_tensor(dataset.data.y[norm_length:], dtype=torch.float).std()

## Normalize targets to mean = 0 and std = 1.
dataset.data.y = (dataset.data.y - mean) / std
if ext_test is True:
    ext_test_dataset.data.y = (ext_test_dataset.data.y - mean) / std

print('Target data mean: ' + str(mean.tolist()))
print('Target data standard deviation: ' + str(std.tolist()))
print('Training is based on ' + str(dataset.num_features) + ' atom features and ' + str(dataset.num_edge_features) + ' edge features for a molecule.')


## Split dataset into train, validation, test set
test_length = int(len(dataset) * test_percent)
val_length = int(len(dataset) * val_percent)

val_dataset = dataset[test_length:test_length+val_length]
train_dataset = dataset[test_length+val_length:]
if ext_test is True:
    test_dataset = ext_test_dataset[:]
else:
    test_dataset = dataset[:test_length]

test_loader = DataLoader(test_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


print('Length of train dataset: ' + str(len(train_dataset)))
print('Length of validation dataset: ' + str(len(val_dataset)))
print('Length of test dataset: ' + str(len(test_dataset)))

model_types = ['GCN_single', 'GCN_multiple', 'graphconv_single', 'graphconv_multiple']
model_type_name = model_types[model_type_int]

gnn_layer_by_name = {
    "GCN": GCNConv,
    "GraphConv": GraphConv,
    "GraphSAGE": SAGEConv
}


# Model structure
class Net(torch.nn.Module):
    def __init__(self, layer_name = gnn, c_in = 14, c_hidden = num_nodes, c_out = 32, num_layers = num_layers):
        super(Net, self).__init__()

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden

        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, out_channels, bias = False, aggr = 'add')
            ]
            in_channels = c_hidden

        layers += [
                gnn_layer(in_channels, c_out, bias = False, aggr = 'add')
            ]

        self.layers = torch.nn.ModuleList(layers)
        
        self.fc11 = torch.nn.Linear(32, 16)
        self.fc12 = torch.nn.Linear(16, 1)

    def forward(self, data, print_intermediaries = False, model_type = model_type_name):
        
        for i, l in enumerate(self.layers):
            if i == 0:
                output_GNN = F.relu(l(data.x, data.edge_index))
            else:
                output_GNN = F.relu(l(output_GNN, data.edge_index))
            

        
        # print(f'>this is what we are looking for?: {self.training}')

        out_scatter = scatter_add(output_GNN, data.batch, dim=0)
     
        out_ANN1 = F.relu(self.fc11(out_scatter))
        
        out_final = self.fc12(out_ANN1)
        
        if print_intermediaries == True:
            if model_type == 'graphconv_single' or 'GCN_single':
                intermediate_values = [data.x, data.edge_index, out_GNN, out_scatter, out_ANN1, out_final]
            elif model_type == 'graphconv_multiple':
                intermediate_values = [data.x, data.edge_index, GCN_1, GCN_2, out_GNN, out_scatter, out_ANN1, out_final]
            elif model_type == 'GCN_multiple':
                intermediate_values = [data.x, data.edge_index, out_GCN_1, out_GCN_2, out_GNN, out_scatter, out_ANN1, out_final]
            else: 
                raise Exception('sum ting wong')            
            
            print(f'this is the value that we are looking for: {out_final}')
        
        return out_final if (print_intermediaries == False) else intermediate_values

model = Net().to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=lrfactor, patience=lrpatience, min_lr=0.0000001)


# Training
def train(epoch):
    model.train()
    loss_all = 0
    std_train_error = 0
    rel_train_error = 0
    train_mae, train_norm_mae, train_mre = 0, 0, 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        tmp_pred_data = model(data)
        tmp_real_data = data.y
        loss = F.mse_loss(tmp_pred_data, tmp_real_data)
        loss.backward()
        loss_all += loss * data.num_graphs
        optimizer.step()

        # Error calculations
        std_train_error += (tmp_pred_data - tmp_real_data).abs().sum(0).item()
        rel_train_error += ((tmp_pred_data - tmp_real_data)/(tmp_real_data+mean/std)).abs().sum(0).item()

    train_norm_mae = std_train_error
    train_mae = std_train_error*std
    train_mre = rel_train_error
   	
    return loss_all / len(train_loader.dataset), train_norm_mae / len(train_loader.dataset), train_mae / len(train_loader.dataset), train_mre / len(train_loader.dataset)


# Testing 
def test(loader, verbose):
    model.eval()
    std_pred_error = 0
    rel_pred_error = 0
    mae, norm_mae, mre = 0, 0, 0

    for data in loader:
        data = data.to(device)
        tmp_pred_data = model(data)
        tmp_real_data = data.y
        loss = F.mse_loss(tmp_pred_data, tmp_real_data)

        # Error calculations
        std_pred_error += (tmp_pred_data - tmp_real_data).abs().sum(0).item()
        rel_pred_error += ((tmp_pred_data - tmp_real_data)/(tmp_real_data+mean/std)).abs().sum(0).item()

    norm_mae = std_pred_error
    mae = std_pred_error*std
    mre = rel_pred_error

    return norm_mae / len(loader.dataset), mae / len(loader.dataset), mre / len(loader.dataset)


# Write model predictions to csv file
def write_predictions(loader, save_path, dataset_type):
    model.eval()
    mol_id, pred, real_value, mol_names, pred_list, mae_errors, mre_errors = None, None, None, [], [], [], []
    with open(save_path + "/predictions.csv","a+") as pred_file:
        pred_file.write('\n Predictions for ' + dataset_type + ' dataset')
        pred_file.write('\n ' + 'SMILES' + ',' + 'Predicted ' + target_name + ',' + 'Measured ' + target_name + ',' + 'Absolute Error ' + target_name + ',' + 'Relative Error ' + target_name) 
        for data in loader:
            mol_id = data.mol_id.tolist()
            for mol in mol_id:
                tmp_mol_name = ''
                for i in mol: 
                    if int(i) != 0:
                        tmp_mol_name += chr(int(i))
                mol_names.append(tmp_mol_name)
                print(f'this is the mol name: {tmp_mol_name}')
                if tmp_mol_name == 'CCCCl':
                    CCCCl_data = []
                    print('this one!')
                    data_2 = data.to(device)
                    intermediate_values = model(data)
                    torch.save(intermediate_values, save_path + "/intermediate_values_1.tar")

                    print(f'are we training?: {model.training}')
                    print(f'this is the other pred: {intermediate_values}')
                    
            real_value = data.y.tolist()
            data = data.to(device)
            pred = model(data).tolist()
            print(f'this is the pred: {pred}')

        
            for c, k in enumerate(pred):
                if data_scaling == 'standard':
                    pred_list.append([mol_names[c],(pred[c][0]*std+mean).item(), (real_value[c][0]*std+mean).item(), ((pred[c][0]-real_value[c][0])*std).abs().item(), abs((pred[c][0]-real_value[c][0])/(real_value[c][0]+mean/std)).item()])
            mol_names = []
        
        pred_list.sort()
        for k in pred_list:
            pred_file.write('\n ' + str(k[0]) + ',' + str(k[1]) + ',' + str(k[2]) + ',' + str(k[3]) + ',' + str(k[4]))
            if dataset_type == 'train':
               mae_errors.append(k[3])
               mre_errors.append(k[4])
        if dataset_type == 'train':
            mae = torch.tensor(mae_errors, dtype=torch.float).mean()
            mre = torch.tensor(mre_errors, dtype=torch.float).mean()
            return mae, mre

# Define save path and make directories
def preprocess_save(dataset_folder_name):
    model_type = 'base_models'
    save_dir_model_type = str(osp.dirname(osp.realpath(__file__))) + '/Training/' + model_type
    save_dir_target = save_dir_model_type + '/' + target_name
    save_dir = save_dir_model_type + '/' + dataset_folder_name   # directory based on target
    train_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # training id (date and time) 
    save_path = save_dir  + train_id + '-' + str(gnn) + "_" + str(num_nodes) + "_" + str(num_layers)

    try:
        os.mkdir(save_dir_model_type)
    except:
        print('Model type directory already exists.')
    try:
        os.mkdir(save_dir_target)
    except:
        print('Target directory already exists.')
    try:
        os.mkdir(save_dir)
    except:
        print('Base model for this dataset and target already exits.')
    try:
        os.mkdir(save_path)
    except:
        print('Training directory already exists - older model is replaced by model from this training run.')
    return save_path


# Specify and create saving directory and initialize model performance variables for training
save_path = preprocess_save(dataset_folder_name)

early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, save_path=save_path)

best_epoch, best_val_mae, best_epoch_test_mae = None, None, None
best_epoch_val_mre, best_epoch_test_mre = None, None
train_losses, val_errors, test_errors = [], [], []

print('\n')
print(model)

print('\n-- Start Training --')

for epoch in range(1, epochs+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss, train_norm_mae, train_mae, train_mre = train(epoch)
    val_norm_mae, val_mae, val_mre = test(val_loader, verbose=False)
    test_norm_mae, test_mae, test_mre = test(test_loader, verbose=False)

    scheduler.step(val_norm_mae)

    if best_val_mae is None:
        best_epoch = epoch
        best_val_mae, best_epoch_test_mae = val_mae, test_mae
        best_epoch_val_mre, best_epoch_test_mre = val_mre, test_mre
    elif val_mae < best_val_mae:
        best_epoch = epoch
        best_val_mae, best_epoch_test_mae = val_mae, test_mae
        best_epoch_val_mre, best_epoch_test_mre = val_mre, test_mre

    train_losses.append(train_mae)
    val_errors.append(val_mae)
    test_errors.append(test_mae)
    
    print(
        '\nEpoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, Validation norm MAE {:.7f}, Val MRE {:.7f}, Test MAE: {:.7f}, Test norm MAE {:.7f}, Test MRE {:.7f}'
        .format(epoch, lr, loss, val_mae, val_norm_mae, val_mre, test_mae, test_norm_mae, test_mre))


    # early_stopping needs the validation loss to check if it has decresed, and if it has, a checkpoint of the current model will be created
    early_stopping(val_norm_mae, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

print('\n-- Training ended --\n')

# Write final model predictions for train, val, test set to csv file 
model.load_state_dict(torch.load(save_path + '/base_model.pt'))
best_epoch_train_mae, best_epoch_train_mre = write_predictions(train_loader, save_path, 'train')
write_predictions(val_loader, save_path, 'val')
write_predictions(test_loader, save_path, 'test')

# Print best model performance
print(
    'Best model with respect to validation error in epoch {:03d} with \nTrain MAE {:.7f}, Train MRE {:.7f}; \nVal MAE {:.7f}, MRE {:.7f}; \nTest MAE {:.7f}, MRE {:.7f} \n'
    .format(best_epoch, best_epoch_train_mae, best_epoch_train_mre, best_val_mae, best_epoch_val_mre, best_epoch_test_mae, best_epoch_test_mre))


# Write model performance to general csv file
with open(save_path + '/..' + '/model_results_' + str(gnn) + '_layers_' + str(num_layers) + '_depth_' + str(num_nodes) + '.csv','a+', newline='') as result_file:
     wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
     wr.writerow([save_path[:], gnn, num_layers, num_nodes, best_epoch_train_mae.item(), best_epoch_train_mre.item(), best_epoch_test_mae.item(), best_epoch_test_mre, best_val_mae.item(), best_epoch_val_mre])


# # Write model performance to general csv file
# with open(save_path + '/..' + '/model_results_' + target_name + '_12-GRU12_' + str(conv_type) + 'Conv1_' + str(conv_type2) + 'Conv2_' +  str(dim) + 'dim_' +str(lrate) + 'lr_' + str(lrfactor) + 'lrf_' + str(lrpatience) + 'lrp_' + pool_type + 'pool.csv','a+', newline='') as result_file:
#      wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
#      wr.writerow(['Train error (MAE, MRE)', '', best_epoch_train_mae.item(), best_epoch_train_mre.item()])
#      wr.writerow(['Test error (MAE, MRE)', save_path[:], best_epoch_test_mae.item(), best_epoch_test_mre])
#      wr.writerow(['Val error (MAE, MRE)', 'Epoch ' + str(best_epoch), best_val_mae.item(), best_epoch_val_mre])

# Plot train, val, test details and save plot
plt.plot(range(1,len(train_losses)+1), train_losses, label='Train')
plt.plot(range(1,len(val_errors)+1), val_errors, label = 'Validation')
plt.plot(range(1,len(test_errors)+1), test_errors, label='Test')

es_epoch = val_errors.index(min(val_errors))+1 
plt.axvline(es_epoch, linestyle='--', color='r',label='Early Stopping Epoch')

plt.xlabel('Epochs')
plt.ylabel('Mean absolute error')
plt.grid(True)
plt.legend(frameon=False)
plt.tight_layout()
save_plot = osp.join(save_path, 'loss')
plt.savefig(save_plot, bbox_inches='tight')
plt.close()

print('Model saved in ' + save_path +'\n')

for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")

    
state_dict = model.state_dict()
# print(state_dict)
# print(state_dict.size())

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(state_dict, save_path + "/" + str(gnn) + "_" + str(num_nodes) + "_" + str(num_layers) + '.tar')

