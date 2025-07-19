import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from utils.GNN_data import *
from utils.GNN_model import GIN
from utils.train_model import train_model
from utils.evaluate_model import *
# --------------------------------------
# Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('--Concrete', type=str, default='B1_B2_FB1_FB2', help='Concrete image in dataset')
parser.add_argument('--Unseen_Concrete', type=str, default='CG', help='unseen concrete image')
parser.add_argument('--batch_num', type=int, default=64, help='batch size')
parser.add_argument('--epoch_num', type=int, default=200, help='Epoch number')
parser.add_argument('--avg_scheme', choices=['Voigt', 'Hill'], default='Voigt')
parser.add_argument('--cover_interval', type=int, default=3, help='Cover interval')
parser.add_argument('--overlap', type=float, default=0.7, help='cover overlap')
parser.add_argument('--excluded_subcube', type=str, default='15', help='excluded subcube')
parser.add_argument('--unseen_subcube_size', type=str, default='10_20_25', help='excluded subcube size')
parser.add_argument('--save_model_dir', type=str, default='./examples/saved_GNN_model', help='saved GNN model directory')
args = parser.parse_args()

batch_size = args.batch_num
epoch_num = args.epoch_num
overlap=args.overlap
learning_rate = 0.005

#
data_dir = f'./examples/graph_data/{args.Concrete}_DFS_CoverInterval_{args.cover_interval}_Overlap_{overlap}_except_subcube{args.excluded_subcube}.pkl'

# -------------------------------------------
# Load data and perform data pre-processing
# Load data
data_list = load_data(data_dir)

# Apply the selected material averaging scheme to the dataset
data_list = Elastic_property_avg_scheme(data_list, args.avg_scheme)

# Check if the graph data is directed or undirected. For this study, undirected edges are used.
all_undirected_before = all(is_undirected(data) for data in data_list)
print(f'Graphs are undirected (before conversion): {all_undirected_before}')  # True if all are undirected, False otherwise

# Convert all graphs in the dataset to undirected
data_list = [make_undirected(data) for data in data_list]

# Re-check if all graphs are now undirected
all_undirected_after = all(is_undirected(data) for data in data_list)
print(f'Graphs are undirected (after conversion): {all_undirected_after}')  # True if all are undirected, False otherwise

# Normalize the spatial features in nodes
data_list = normalize_spatial_info(data_list)

# Augment data
data_list = augment_dataset(data_list)

# Split dataset for train/validation/test
train_loader, test_loader, valid_loader = prepare_train_dataset(data_list, batch_size, k_folds=15, random_seed=128)

# ----------------------------------------------
# Define GNN architecture
# Define the model, optimizer, and loss function
no_node_feature = data_list[0].x.shape[1]
model = GIN(dim_h=16, node_feature=no_node_feature)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.MSELoss()
print(model)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)

# ------------------------------------------------
# train model
train_losses, valid_losses, test_losses, R2_trainings, R2_valids, R2_tests, train_acc_total, valid_acc_total, test_acc_total, best_state_dict = train_model(
    model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler, device=device, num_epochs=epoch_num)

# 1) Epoch vs R^2
plt.figure(dpi=300, figsize=(10,8))

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

plt.plot(np.array(train_losses), label='Train_loss')
plt.plot(np.array(test_losses), label='Test_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel(r'MSE Loss ($\mathcal{L})$')
plt.yscale("log")
plt.xlim(-5,args.epoch_num)
plt.tight_layout()
plt.savefig(f'{args.save_model_dir}/epoch_{args.epoch_num}.png')
#-------------------------------------------------
import pandas as pd
import os

# # Create the save directory (if it doesn't exist)
os.makedirs(args.save_model_dir, exist_ok=True)

# Save training loss and test loss to CSV
loss_df = pd.DataFrame({
    'Epoch': np.arange(1, len(train_losses) + 1),
    'Train_Loss': train_losses,
    'Test_Loss': test_losses
})
loss_csv_path = f"{args.save_model_dir}/loss_history.csv"
loss_df.to_csv(loss_csv_path, index=False)

print(f"Loss history saved to {loss_csv_path}")

# ------------------------------------------------
# save model
best_model_path = f'{args.save_model_dir}/epoch_{args.epoch_num}.pt'
torch.save(best_state_dict, best_model_path)

# ------------------------------------------------
# Evaluate model for the test dataset (observe train dataset performance)
R2_K, R2_mu = evaluate_model(model, test_loader, device, args.Concrete, args.cover_interval, overlap, args.excluded_subcube, args.save_model_dir)
R2_K, R2_mu = evaluate_train_model(model, train_loader, device, args.Concrete, args.cover_interval, overlap, args.excluded_subcube, args.save_model_dir)
# ------------------------------------------------


print('training_prediction_completed!')

# ---------------------------------------------
# Save training results to a CSV file
def save_results_to_csv(model, loader, device, save_path):
    model.eval()
    K_true_list, K_pred_list = [], []
    mu_true_list, mu_pred_list = [], []

    with torch.no_grad():
        for data in loader:
            K_pred, mu_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            K_true_list.extend(data.K.cpu().numpy())
            mu_true_list.extend(data.mu.cpu().numpy())
            K_pred_list.extend(K_pred.cpu().numpy())
            mu_pred_list.extend(mu_pred.cpu().numpy())

    # Create a DataFrame for bulk moduli
    df_bulk = pd.DataFrame({
        'True_Bulk_Moduli': K_true_list,
        'Predicted_Bulk_Moduli': K_pred_list
    })

    # Create a DataFrame for shear moduli
    df_shear = pd.DataFrame({
        'True_Shear_Moduli': mu_true_list,
        'Predicted_Shear_Moduli': mu_pred_list
    })

    # Save to CSV with multiple sheets
    with pd.ExcelWriter(save_path) as writer:
        df_bulk.to_excel(writer, sheet_name='Bulk_Moduli', index=False)
        df_shear.to_excel(writer, sheet_name='Shear_Moduli', index=False)

# Save training results
save_results_to_csv(model, train_loader, device, f'{args.save_model_dir}/training_results.xlsx')


# ---------------------------------------------
# Save test results to a CSV file
def save_test_results_to_csv(model, loader, device, save_path):
    model.eval()
    K_true_list, K_pred_list = [], []
    mu_true_list, mu_pred_list = [], []

    with torch.no_grad():
        for data in loader:
            K_pred, mu_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            K_true_list.extend(data.K.cpu().numpy())
            mu_true_list.extend(data.mu.cpu().numpy())
            K_pred_list.extend(K_pred.cpu().numpy())
            mu_pred_list.extend(mu_pred.cpu().numpy())

    # Create a DataFrame for bulk moduli
    df_bulk = pd.DataFrame({
        'True_Bulk_Moduli': K_true_list,
        'Predicted_Bulk_Moduli': K_pred_list
    })

    # Create a DataFrame for shear moduli
    df_shear = pd.DataFrame({
        'True_Shear_Moduli': mu_true_list,
        'Predicted_Shear_Moduli': mu_pred_list
    })

    # Save to CSV with multiple sheets
    with pd.ExcelWriter(save_path) as writer:
        df_bulk.to_excel(writer, sheet_name='Bulk_Moduli', index=False)
        df_shear.to_excel(writer, sheet_name='Shear_Moduli', index=False)

# Save test results
save_test_results_to_csv(model, test_loader, device, f'{args.save_model_dir}/test_results.xlsx')
