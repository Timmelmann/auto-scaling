import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import List
import gc
import argparse
import json
import os
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_df_improved(df: pd.DataFrame, services: List[str] = None, time_col='timestamp',
                           value_col='value', reporter_col='destination', interval='1min',
                           use_log_transform=True, remove_outliers=True) -> pd.DataFrame:
    df = df[[time_col, reporter_col, value_col]].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit='s')
    else:
        df[time_col] = pd.to_datetime(df[time_col])

    if services is None:
        services = sorted(df[reporter_col].unique())

    df.set_index(time_col, inplace=True)
    df_pivot = df.groupby([pd.Grouper(freq=interval), reporter_col])[value_col].sum().unstack(fill_value=0)
    df_pivot = df_pivot.reindex(columns=services, fill_value=0)
    df_pivot.fillna(0, inplace=True)

    print(f"Original data shape: {df_pivot.shape}")
    print(f"Original data range: {df_pivot.min().min():.2f} to {df_pivot.max().max():.2f}")

    if remove_outliers:
        print("Removing extreme outliers...")
        upper_limits = df_pivot.quantile(0.95)
        df_pivot = df_pivot.clip(upper=upper_limits, axis=1)
        print(f"After outlier removal: {df_pivot.min().min():.2f} to {df_pivot.max().max():.2f}")

    if use_log_transform:
        print("Applying log transformation...")
        df_pivot = np.log1p(df_pivot)
        print(f"After log transform: {df_pivot.min().min():.2f} to {df_pivot.max().max():.2f}")

    return df_pivot


def train_val_test_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    return train_df, val_df, test_df


def normalize_dataframe_improved(df: pd.DataFrame, method='robust', mean_series=None,
                                 std_series=None, scaler=None) -> tuple:
    if method == 'minmax':
        if scaler is None:
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df),
                index=df.index,
                columns=df.columns
            )
            return df_scaled, None, None, scaler
        else:
            df_scaled = pd.DataFrame(
                scaler.transform(df),
                index=df.index,
                columns=df.columns
            )
            return df_scaled, None, None, scaler

    elif method == 'robust':
        if scaler is None:
            scaler = RobustScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df),
                index=df.index,
                columns=df.columns
            )
            return df_scaled, None, None, scaler
        else:
            df_scaled = pd.DataFrame(
                scaler.transform(df),
                index=df.index,
                columns=df.columns
            )
            return df_scaled, None, None, scaler

    elif method == 'zscore':
        if mean_series is None:
            mean_series = df.mean()
        if std_series is None:
            std_series = df.std() + 1e-8

        df_scaled = (df - mean_series) / std_series
        return df_scaled, mean_series, std_series, None


def create_sequences_memory_efficient_with_augmentation(df_scaled: pd.DataFrame, window: int, horizon: int,
                                                        augment_training=True) -> tuple[
    List[np.ndarray], List[np.ndarray]]:
    total_window = window + horizon
    if len(df_scaled) <= total_window:
        raise ValueError(f"Not enough data. Available: {len(df_scaled)}, needed: {total_window}")

    X, Y = [], []
    timestamps = df_scaled.index

    for i in range(len(df_scaled) - total_window):
        x_window_data = df_scaled.iloc[i:i + window].values

        if augment_training:
            noise_scale = 0.01 * np.std(x_window_data)
            x_window_data = x_window_data + np.random.normal(0, noise_scale, x_window_data.shape)

        current_timestamps = timestamps[i:i + window]
        hours = current_timestamps.hour
        days_of_week = current_timestamps.dayofweek

        hour_sin = np.sin(2 * np.pi * hours / 24)
        dow_sin = np.sin(2 * np.pi * days_of_week / 7)

        temporal_features = np.stack([hour_sin, dow_sin], axis=1)

        x_window_data_reshaped = x_window_data[:, :, np.newaxis]
        temporal_expanded = np.expand_dims(temporal_features, axis=1)
        temporal_expanded = np.repeat(temporal_expanded, df_scaled.shape[1], axis=1)

        combined_x_window = np.concatenate([x_window_data_reshaped, temporal_expanded], axis=-1)
        X.append(combined_x_window.transpose(1, 0, 2))  # (N, T, 3)

        Y_segment = df_scaled.iloc[i + window:i + window + horizon].values.T
        Y.append(Y_segment)

    return X, Y


class GCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, X, A):
        if A.size(0) != X.size(0):
            A = A.expand(X.size(0), -1, -1)
        AX = torch.bmm(A, X)
        out = torch.matmul(AX, self.weight) + self.bias
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class GraphBasedModell(nn.Module):
    def __init__(self, num_services, input_dim=3, embedding_dim=32,
                 gcn_hidden_dim=128, lstm_hidden_dim=256, horizon=1,
                 dropout_rate=0.2, dynamic_adj_threshold=0.3):
        super().__init__()

        self.horizon = horizon
        self.num_services = num_services
        self.dynamic_adj_threshold = dynamic_adj_threshold
        self.input_dim = input_dim

        print(f"BALANCED: GCN dim {gcn_hidden_dim}, LSTM dim {lstm_hidden_dim}")

        self.static_adj = nn.Parameter(torch.eye(num_services) * 0.1)
        self.adj_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.gcn1 = GCNLayer(input_dim, gcn_hidden_dim, dropout_rate=dropout_rate)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_hidden_dim, dropout_rate=dropout_rate)

        self.lstm = nn.LSTM(
            gcn_hidden_dim, lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if lstm_hidden_dim > 1 else 0
        )

        self.lstm_dropout = nn.Dropout(dropout_rate)
        self.lstm_layer_norm = nn.LayerNorm(lstm_hidden_dim)

        head_hidden = max(64, lstm_hidden_dim // 2)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(head_hidden // 2, horizon)
            ) for _ in range(num_services)
        ])

        self.service_scales = nn.Parameter(torch.ones(num_services))
        self.service_biases = nn.Parameter(torch.zeros(num_services))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _gcn_forward(self, X, A):
        gcn_out1 = F.relu(self.gcn1(X, A))
        gcn_out2 = F.relu(self.gcn2(gcn_out1, A))
        return gcn_out2

    def compute_causal_dynamic_adj(self, X, current_timestep, threshold=0.3):
        B, N, T = X.size()

        if current_timestep <= 0:
            return self.static_adj.unsqueeze(0).expand(B, -1, -1)

        past_X = X[:, :, :current_timestep]
        past_X = past_X - past_X.mean(dim=2, keepdim=True)
        std = past_X.std(dim=2, keepdim=True) + 1e-8
        norm_past_X = past_X / std

        adj = torch.matmul(norm_past_X, norm_past_X.transpose(1, 2)) / current_timestep

        threshold = threshold * 1.5
        adj = torch.where(adj >= threshold, adj, torch.zeros_like(adj))

        adj = F.normalize(adj, p=2, dim=-1)

        return adj

    def forward(self, X):
        B, N, T, D = X.size()


        X = F.dropout(X, p=0.1, training=self.training)

        gcn_outputs = []
        for t in range(T):
            A_dynamic = self.compute_causal_dynamic_adj(
                X[:, :, :, 0].contiguous(), t, self.dynamic_adj_threshold
            )


            weights = F.softmax(self.adj_weights * 0.5, dim=0)
            A = weights[0] * A_dynamic + weights[1] * self.static_adj.unsqueeze(0).expand(B, -1, -1)

            Xt = X[:, :, t, :]
            h = self._gcn_forward(Xt, A)
            gcn_outputs.append(h)


        gcn_seq = torch.stack(gcn_outputs, dim=2)

        lstm_input = gcn_seq.reshape(B * N, T, -1)

        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        final_hidden = h_n[-1]
        final_hidden = self.lstm_layer_norm(final_hidden)
        final_hidden = self.lstm_dropout(final_hidden)

        service_features = final_hidden.view(B, N, -1)

        predictions = []
        for i in range(N):
            service_feature = service_features[:, i, :]
            service_pred = self.prediction_heads[i](service_feature)

            service_pred = service_pred * torch.tanh(self.service_scales[i]) + self.service_biases[i] * 0.1
            predictions.append(service_pred)

        predictions = torch.stack(predictions, dim=1)
        return predictions


def train_with_anti_overfitting(model, X_train, Y_train, X_val, Y_val,
                                criterion, optimizer, device, services,
                                batch_size=64, num_epochs=150, patience=8):
    service_weights = []
    for i in range(len(services)):
        service_var = Y_train[:, i, :].var().item()
        weight = 1.0 / (service_var + 1e-8)
        service_weights.append(weight)

    service_weights = torch.tensor(service_weights, device=device)
    service_weights = service_weights / service_weights.sum() * len(services)

    print("Service-specific loss weights:")
    for i, (service, weight) in enumerate(zip(services, service_weights)):
        print(f"  {service}: {weight:.3f}")

    best_val_loss = float('inf')
    wait = 0
    train_losses = []
    val_losses = []

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    print(f"ANTI-OVERFITTING SETTINGS:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Patience: {patience}")
    print(f"  - Learning rate scheduler: ReduceLROnPlateau")

    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()

            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)

            predictions = model(batch_X)

            batch_losses = criterion(predictions, batch_Y)
            weighted_losses = batch_losses.mean(dim=(0, 2)) * service_weights
            loss = weighted_losses.mean()

            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += 5e-5 * l2_reg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_Y = batch_Y.to(device, non_blocking=True)

                predictions = model(batch_X)
                batch_losses = criterion(predictions, batch_Y)
                weighted_losses = batch_losses.mean(dim=(0, 2)) * service_weights
                val_loss = weighted_losses.mean()

                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"EARLY STOPPING at epoch {epoch + 1} (patience={patience})")
                break

        if epoch % 10 == 0 or epoch < 10:
            val_train_ratio = avg_val_loss / avg_train_loss
            print(
                f"Epoch {epoch + 1:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, Ratio={val_train_ratio:.3f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining completed:")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Final ratio: {val_losses[-1] / train_losses[-1]:.3f}")

    return best_val_loss, best_model_state


def denormalize_predictions(predictions, targets, normalization_method, mean_series=None,
                            std_series=None, scaler=None):
    if normalization_method in ['minmax', 'robust']:
        B, N, H = predictions.shape
        pred_reshaped = predictions.reshape(-1, N)
        target_reshaped = targets.reshape(-1, N)

        pred_orig = scaler.inverse_transform(pred_reshaped).reshape(B, N, H)
        target_orig = scaler.inverse_transform(target_reshaped).reshape(B, N, H)

        return pred_orig, target_orig

    elif normalization_method == 'zscore':
        mean_np = mean_series.values.reshape(1, -1, 1)
        std_np = std_series.values.reshape(1, -1, 1)

        pred_orig = predictions * std_np + mean_np
        target_orig = targets * std_np + mean_np

        return pred_orig, target_orig


def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    non_zero_idx = np.abs(y_true) > epsilon

    if np.sum(non_zero_idx) == 0:
        return 0.0

    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / np.abs(y_true[non_zero_idx]))) * 100
    return mape


def evaluate_model_anti_overfit(model, X_data, Y_data, services, device, batch_size=64,
                                normalization_method='robust', mean_series=None,
                                std_series=None, scaler=None, use_log_transform=True):
    model.eval()

    eval_dataset = TensorDataset(X_data, Y_data)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_Y in eval_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)

            predictions = model(batch_X)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_Y.cpu())

    y_pred = torch.cat(all_predictions, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    y_pred_denorm, y_true_denorm = denormalize_predictions(
        y_pred, y_true, normalization_method, mean_series, std_series, scaler
    )

    if use_log_transform:
        y_pred_orig = np.expm1(y_pred_denorm)
        y_true_orig = np.expm1(y_true_denorm)
        y_pred_orig = np.maximum(y_pred_orig, 0)
        y_true_orig = np.maximum(y_true_orig, 0)
    else:
        y_pred_orig = y_pred_denorm
        y_true_orig = y_true_denorm

    service_results = {}
    total_directional_acc = []
    total_mae = []
    total_rmse = []
    total_smape = []
    total_mape = []

    for i, service in enumerate(services):
        y_s = y_true_orig[:, i, :].flatten()
        y_p = y_pred_orig[:, i, :].flatten()

        mae = mean_absolute_error(y_s, y_p)
        rmse = np.sqrt(mean_squared_error(y_s, y_p))
        smape = 100 * np.mean(2 * np.abs(y_s - y_p) / (np.abs(y_s) + np.abs(y_p) + 1e-8))
        mape = calculate_mape(y_s, y_p)

        if len(y_s) > 1:
            true_diff = np.diff(y_s)
            pred_diff = np.diff(y_p)
            directional_acc = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
            total_directional_acc.append(directional_acc)
        else:
            directional_acc = 0

        print(
            f"{service}: MAE={mae:.2f}, RMSE={rmse:.2f}, SMAPE={smape:.2f}%, MAPE={mape:.2f}%, DA={directional_acc:.1f}%")

        service_results[service] = {
            'mae': mae, 'rmse': rmse, 'smape': smape, 'mape': mape,
            'directional_accuracy': directional_acc
        }

        total_mae.append(mae)
        total_rmse.append(rmse)
        total_smape.append(smape)
        total_mape.append(mape)

    avg_directional_acc = np.mean(total_directional_acc)
    avg_mae = np.mean(total_mae)
    avg_rmse = np.mean(total_rmse)
    avg_smape = np.mean(total_smape)
    avg_mape = np.mean(total_mape)

    print(f"\nOVERALL METRICS:")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average SMAPE: {avg_smape:.2f}%")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average Directional Accuracy: {avg_directional_acc:.1f}%")

    return {
        'services': service_results,
        'average': {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'smape': avg_smape,
            'mape': avg_mape,
            'directional_accuracy': avg_directional_acc
        },
        'overfitting_detected': avg_directional_acc > 75
    }


def optimize_memory_settings():
    import os
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run graph-based LSTM model with Option A hyperparameters')

    parser.add_argument('--data_path', type=str, default="../../resource/transformed_http_1m_7d.csv", required=False,
                        help='Path to the CSV data file')
    parser.add_argument('--window_minutes', type=int, default=12,  # INCREASED FROM 12
                        help='Window size in minutes')
    parser.add_argument('--prediction_minutes', type=int, default=1,
                        help='Prediction horizon in minutes')
    parser.add_argument('--hidden_dim', type=int, default=64,  # INCREASED FROM 64
                        help='LSTM hidden dimension')
    parser.add_argument('--batch_size', type=int, default=64,  # INCREASED FROM 16
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,  # INCREASED FROM 200
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,  # INCREASED FROM 0.001
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=12,  # REDUCED FROM 15
                        help='Patience for early stopping')
    parser.add_argument('--model_path', type=str, default="model_test", required=False,
                        help='Path to save the trained model')
    parser.add_argument('--results_file', type=str, default="result_test", required=False,
                        help='Path to save the results JSON file')
    parser.add_argument('--output_dir', type=str, default="output_test", required=False,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")


    os.makedirs(args.output_dir, exist_ok=True)

    optimize_memory_settings()

    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    window_size = args.window_minutes
    horizon = args.prediction_minutes

    print(f"Using window size: {window_size} timesteps ({args.window_minutes} minutes)")
    print(f"Using prediction horizon: {horizon} timesteps ({args.prediction_minutes} minutes)")

    batch_size = args.batch_size
    use_log_transform = True
    remove_outliers = True
    normalization_method = 'robust'

    services = sorted(df['destination'].unique())
    print(f"Number of services: {len(services)}")

    processed_df = preprocess_df_improved(
        df, services=services, interval='1min',
        use_log_transform=use_log_transform,
        remove_outliers=remove_outliers
    )

    train_df, val_df, test_df = train_val_test_split(processed_df, train_ratio=0.7, val_ratio=0.15)

    if normalization_method in ['minmax', 'robust']:
        train_df_scaled, _, _, scaler = normalize_dataframe_improved(train_df, method=normalization_method)
        val_df_scaled, _, _, _ = normalize_dataframe_improved(val_df, method=normalization_method, scaler=scaler)
        test_df_scaled, _, _, _ = normalize_dataframe_improved(test_df, method=normalization_method, scaler=scaler)
        mean_series, std_series = None, None
    else:
        train_df_scaled, mean_series, std_series, _ = normalize_dataframe_improved(train_df, method='zscore')
        val_df_scaled, _, _, _ = normalize_dataframe_improved(val_df, method='zscore',
                                                              mean_series=mean_series, std_series=std_series)
        test_df_scaled, _, _, _ = normalize_dataframe_improved(test_df, method='zscore',
                                                               mean_series=mean_series, std_series=std_series)
        scaler = None

    print("Creating sequences with data augmentation...")
    X_train_seq, Y_train_seq = create_sequences_memory_efficient_with_augmentation(
        train_df_scaled, window_size, horizon, augment_training=True)
    X_val_seq, Y_val_seq = create_sequences_memory_efficient_with_augmentation(
        val_df_scaled, window_size, horizon, augment_training=False)

    X_train = torch.tensor(np.stack(X_train_seq), dtype=torch.float32)
    Y_train = torch.tensor(np.stack(Y_train_seq), dtype=torch.float32)
    X_val = torch.tensor(np.stack(X_val_seq), dtype=torch.float32)
    Y_val = torch.tensor(np.stack(Y_val_seq), dtype=torch.float32)

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraphBasedModell(
        num_services=len(services),
        input_dim=X_train.shape[-1],
        embedding_dim=16,
        gcn_hidden_dim=32,
        lstm_hidden_dim=args.hidden_dim,
        horizon=horizon,
        dropout_rate=0.3,
        dynamic_adj_threshold=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none')

    print(f"\nStarting training with seed {args.seed}...")
    best_val_loss, best_model_state = train_with_anti_overfitting(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        services=services,
        batch_size=batch_size,
        num_epochs=args.epochs,
        patience=args.patience
    )

    print("\nLoading best model for evaluation...")
    model.load_state_dict(best_model_state)

    torch.save(best_model_state, args.model_path)
    print(f"Model saved to: {args.model_path}")

    X_test_seq, Y_test_seq = create_sequences_memory_efficient_with_augmentation(
        test_df_scaled, window_size, horizon, augment_training=False)
    X_test = torch.tensor(np.stack(X_test_seq), dtype=torch.float32)
    Y_test = torch.tensor(np.stack(Y_test_seq), dtype=torch.float32)

    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    eval_results = evaluate_model_anti_overfit(
        model=model,
        X_data=X_test,
        Y_data=Y_test,
        services=services,
        device=device,
        batch_size=batch_size,
        normalization_method=normalization_method,
        mean_series=mean_series,
        std_series=std_series,
        scaler=scaler,
        use_log_transform=use_log_transform
    )

    with open(args.results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to: {args.results_file}")

    print(f"\nTraining completed with seed {args.seed}!")
    print(f"Overfitting detected: {eval_results['overfitting_detected']}")
    print(f"Average Directional Accuracy: {eval_results['average']['directional_accuracy']:.1f}%")
    print(f"Average MAPE: {eval_results['average']['mape']:.2f}%")
    print(f"Average MAE: {eval_results['average']['mae']:.2f}")
    print(f"Average RMSE: {eval_results['average']['rmse']:.2f}")
    print(f"Average SMAPE: {eval_results['average']['smape']:.2f}%")

    if not eval_results['overfitting_detected']:
        print("SUCCESS: Overfitting successfully prevented!")
    else:
        print("May need even stronger regularization")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()