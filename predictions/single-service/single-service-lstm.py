import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import random


def analyze_service_characteristics(csv_path, destination):
    """
    Analyze characteristics of a specific service to determine best approach.
    """
    df = pd.read_csv(csv_path)
    dest_data = df[df['destination'] == destination].copy()
    time_series = dest_data.groupby('timestamp')['value'].agg(['mean', 'count', 'min', 'max', 'std']).reset_index()
    time_series = time_series.sort_values('timestamp')

    print(f"\n=== ANALYSIS FOR {destination.upper()} ===")
    print(f"Data points: {len(time_series)}")
    print(f"Value range: {time_series['mean'].min():.2f} to {time_series['mean'].max():.2f}")
    print(f"Standard deviation: {time_series['mean'].std():.2f}")
    print(f"Coefficient of variation: {time_series['mean'].std() / time_series['mean'].mean():.3f}")

    Q1 = time_series['mean'].quantile(0.25)
    Q3 = time_series['mean'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((time_series['mean'] < (Q1 - 1.5 * IQR)) |
                (time_series['mean'] > (Q3 + 1.5 * IQR))).sum()
    print(f"Outliers (IQR method): {outliers} ({outliers / len(time_series) * 100:.1f}%)")

    time_series['trend'] = np.arange(len(time_series))
    correlation = np.corrcoef(time_series['mean'], time_series['trend'])[0, 1]
    print(f"Linear trend correlation: {correlation:.3f}")

    if outliers / len(time_series) > 0.1:  
        print("🔧 RECOMMENDATION: Use RobustScaler + outlier filtering")
        approach = "robust"
    elif time_series['mean'].std() / time_series['mean'].mean() > 1.0: 
        print("🔧 RECOMMENDATION: Use simpler model + data smoothing")
        approach = "simple"
    elif abs(correlation) > 0.5: 
        print("🔧 RECOMMENDATION: Use trend-focused features")
        approach = "trend"
    else:
        print("🔧 RECOMMENDATION: Use standard approach")
        approach = "standard"

    return approach, time_series


class SingleServiceLstm:
    def __init__(self, csv_path, destination, approach="standard", **kwargs):
        """
        Adaptive LSTM model that adjusts based on service characteristics.
        """
        self.csv_path = csv_path
        self.destination = destination
        self.approach = approach
        self.sequence_length = kwargs.get('sequence_length', 24)
        self.train_split = kwargs.get('train_split', 0.8)
        self.validation_split = kwargs.get('validation_split', 0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if approach == "robust":
            self.scaler = RobustScaler()
            self.outlier_filter = True
        elif approach == "simple":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.outlier_filter = False
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.outlier_filter = False

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

        print(f"SingleServiceLstm initialized for {destination} with '{approach}' approach")

    def load_and_preprocess_data(self):
        """
        Load and preprocess data based on the chosen approach.
        """
        df = pd.read_csv(self.csv_path)
        dest_data = df[df['destination'] == self.destination].copy()
        time_series = dest_data.groupby('timestamp')['value'].agg(['mean', 'count', 'min', 'max', 'std']).reset_index()
        time_series = time_series.sort_values('timestamp')
        time_series.set_index('timestamp', inplace=True)

        if self.outlier_filter:
            Q1 = time_series['mean'].quantile(0.25)
            Q3 = time_series['mean'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((time_series['mean'] >= (Q1 - 1.5 * IQR)) &
                            (time_series['mean'] <= (Q3 + 1.5 * IQR)))
            time_series = time_series[outlier_mask]
            print(f"Filtered outliers: {len(outlier_mask) - outlier_mask.sum()} points removed")


        if self.approach == "simple":
            time_series['smoothed_mean'] = time_series['mean'].rolling(window=6, center=True).mean()
            time_series['smoothed_mean'].fillna(time_series['mean'], inplace=True)
            feature_cols = ['smoothed_mean']
        elif self.approach == "trend":
            time_series['trend'] = np.arange(len(time_series))
            time_series['rolling_mean_6'] = time_series['mean'].rolling(window=6, min_periods=1).mean()
            time_series['rolling_trend'] = time_series['rolling_mean_6'].diff().fillna(0)
            feature_cols = ['mean', 'rolling_mean_6', 'trend', 'rolling_trend']
        elif self.approach == "robust":
            time_series['median_12'] = time_series['mean'].rolling(window=12, min_periods=1).median()
            time_series['mad_12'] = time_series['mean'].rolling(window=12, min_periods=1).apply(
                lambda x: np.median(np.abs(x - np.median(x)))
            )
            time_series['lag_1'] = time_series['mean'].shift(1).fillna(time_series['mean'].iloc[0])
            feature_cols = ['mean', 'median_12', 'mad_12', 'lag_1']
        else:
            time_series['rolling_mean_12'] = time_series['mean'].rolling(window=12, min_periods=1).mean()
            time_series['lag_1'] = time_series['mean'].shift(1).fillna(time_series['mean'].iloc[0])
            feature_cols = ['mean', 'rolling_mean_12', 'lag_1']

        self.time_series = time_series[feature_cols].copy()
        print(f"Using features: {feature_cols}")
        return self.time_series

    def prepare_sequences(self):
        """
        Prepare sequences with adaptive parameters.
        """
        data = self.time_series.values

        train_size_index = int(len(data) * self.train_split)
        train_data_raw = data[:train_size_index]
        test_data_raw = data[train_size_index:]

        self.scaler.fit(train_data_raw)
        normalized_train_data = self.scaler.transform(train_data_raw)
        normalized_test_data = self.scaler.transform(test_data_raw)
        normalized_data = np.concatenate((normalized_train_data, normalized_test_data), axis=0)

        X, y = [], []
        for i in range(len(normalized_data) - self.sequence_length):
            X.append(normalized_data[i:i + self.sequence_length])
            y.append(normalized_data[i + self.sequence_length, 0])  # Predict first feature

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

        train_size = int(len(X) * self.train_split)
        val_size = int(train_size * self.validation_split)

        X_train, y_train = X[:train_size - val_size], y[:train_size - val_size]
        X_val, y_val = X[train_size - val_size:train_size], y[train_size - val_size:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)

        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_adaptive_model(self):
        """
        Build model with parameters adapted to the service characteristics.
        """
        input_dim = self.time_series.shape[1]

        if self.approach == "simple":
            hidden_dim, dropout_rate = 16, 0.6 
        elif self.approach == "robust":
            hidden_dim, dropout_rate = 20, 0.5 
        else:
            hidden_dim, dropout_rate = 24, 0.4 

        self.model = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0 if hidden_dim < 20 else 0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)

        if self.approach == "robust":
            self.criterion = nn.HuberLoss(delta=0.5) 
        else:
            self.criterion = nn.MSELoss()

        if self.approach == "simple":
            lr = 0.0005 
        else:
            lr = 0.001

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-3 if self.approach != "simple" else 1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=False
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Adaptive model built: {total_params} parameters, approach: {self.approach}")
        return self.model

    def train_adaptive(self, epochs=150, patience=25):
        """
        Train with adaptive parameters based on service characteristics.
        """
        if self.approach == "simple":
            batch_size = 16 
        else:
            batch_size = 32

        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=batch_size,
            shuffle=False
        )

        best_val_loss = float('inf')
        counter = 0
        best_model_state = None
        val_loss_history = []

        for epoch in range(epochs):
            self.model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                lstm_out, _ = self.model[0](X_batch)
                lstm_last = lstm_out[:, -1, :]

                output = lstm_last
                for layer in self.model[1:]:
                    output = layer(output)

                loss = self.criterion(output, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            self.history['train_loss'].append(train_loss)

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    lstm_out, _ = self.model[0](X_batch)
                    lstm_last = lstm_out[:, -1, :]

                    output = lstm_last
                    for layer in self.model[1:]:
                        output = layer(output)

                    val_loss = self.criterion(output, y_batch)
                    val_losses.append(val_loss.item())

            val_loss = np.mean(val_losses)
            val_loss_history.append(val_loss)
            self.history['val_loss'].append(val_loss)

            window_size = min(5, len(val_loss_history))
            smoothed_val_loss = np.mean(val_loss_history[-window_size:])

            self.scheduler.step(smoothed_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}, Smoothed: {smoothed_val_loss:.6f}")

            if smoothed_val_loss < best_val_loss - 1e-6:
                best_val_loss = smoothed_val_loss
                counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def evaluate_adaptive(self):
        """
        Evaluate with proper handling of different scalers.
        """
        self.model.eval()
        with torch.no_grad():
            lstm_out, _ = self.model[0](self.X_test)
            lstm_last = lstm_out[:, -1, :]

            predictions = lstm_last
            for layer in self.model[1:]:
                predictions = layer(predictions)

        predicted = predictions.cpu().numpy()
        actual = self.y_test.cpu().numpy()

        if isinstance(self.scaler, RobustScaler):
            predicted_denorm = np.zeros((len(predicted), self.time_series.shape[1]))
            actual_denorm = np.zeros((len(actual), self.time_series.shape[1]))

            predicted_denorm[:, 0] = predicted.flatten()
            actual_denorm[:, 0] = actual.flatten()

            predicted_values = self.scaler.inverse_transform(predicted_denorm)[:, 0]
            actual_values = self.scaler.inverse_transform(actual_denorm)[:, 0]
        else:
            predicted_denorm = np.zeros((len(predicted), self.time_series.shape[1]))
            actual_denorm = np.zeros((len(actual), self.time_series.shape[1]))

            predicted_denorm[:, 0] = predicted.flatten()
            actual_denorm[:, 0] = actual.flatten()

            predicted_values = self.scaler.inverse_transform(predicted_denorm)[:, 0]
            actual_values = self.scaler.inverse_transform(actual_denorm)[:, 0]

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)

        ss_total = np.sum((actual_values - np.mean(actual_values)) ** 2)
        ss_residual = np.sum((actual_values - predicted_values) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        epsilon = 1e-10
        mape = np.mean(np.abs((actual_values - predicted_values) / (actual_values + epsilon))) * 100

        print(f"Adaptive evaluation for {self.destination}:")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.4f}, MAPE: {mape:.2f}%")
        return rmse, mae, r_squared, mape


def train_problematic_services(csv_path, problem_services=None):
    """
    Train adaptive models for problematic services.
    """
    if problem_services is None:
        problem_services = ['currencyservice', 'paymentservice', 'shippingservice', 'checkoutservice']

    print("=== TRAINING ADAPTIVE MODELS FOR PROBLEMATIC SERVICES ===")

    models = {}
    results = {}

    for service in problem_services:
        print(f"\n{'=' * 50}")
        print(f"ADAPTIVE TRAINING FOR {service.upper()}")
        print(f"{'=' * 50}")

        try:
            approach, _ = analyze_service_characteristics(csv_path, service)

            model = SingleServiceLstm(
                csv_path=csv_path,
                destination=service,
                approach=approach
            )

            # Train
            model.load_and_preprocess_data()
            model.prepare_sequences()
            model.build_adaptive_model()
            model.train_adaptive(epochs=200, patience=30)

            rmse, mae, r_squared, mape = model.evaluate_adaptive()

            models[service] = model
            results[service] = {
                'rmse': rmse, 'mae': mae, 'r_squared': r_squared, 'mape': mape,
                'approach': approach
            }

        except Exception as e:
            print(f"Error training {service}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("ADAPTIVE TRAINING RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Service':<20} {'Approach':<10} {'R-squared':<12} {'MAPE':<8} {'RMSE':<12}")
    print("-" * 62)

    for service, result in results.items():
        print(
            f"{service:<20} {result['approach']:<10} {result['r_squared']:<12.4f} {result['mape']:<8.2f} {result['rmse']:<12.2f}")

    return models, results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train adaptive LSTM models for HTTP traffic data')
    parser.add_argument('--csv_path', type=str, default="../../resource/transformed_http_1m_7d.csv",
                        help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default="lstm_results_adaptive",
                        help='Base directory to save results')
    parser.add_argument('--specific_destinations', type=str, nargs='+',
                        help='List of specific destinations to model (default: all)')
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Sequence length for LSTM')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--future_steps', type=int, default=48,
                        help='Number of future steps to predict')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--adaptive_only', action='store_true',
                        help='Only train problematic services with adaptive approach')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    seed_output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    print(f"=== ADAPTIVE LSTM TRAINING WITH SEED {args.seed} ===")
    print(f"Output directory: {seed_output_dir}")

    if args.adaptive_only:
        if args.specific_destinations:
            services_to_train = args.specific_destinations
        else:
            services_to_train = ['currencyservice', 'paymentservice', 'shippingservice', 'checkoutservice']

        models, results = train_problematic_services(args.csv_path, services_to_train)
    else:
        df = pd.read_csv(args.csv_path)
        all_destinations = df['destination'].unique() if not args.specific_destinations else args.specific_destinations

        problematic_services = ['currencyservice', 'paymentservice', 'shippingservice', 'checkoutservice']

        models = {}
        results = {}

        for destination in all_destinations:
            print(f"\n{'=' * 50}")
            print(f"TRAINING {destination.upper()}")
            print(f"{'=' * 50}")

            try:
                if destination in problematic_services:
                    # Use adaptive approach for problematic services
                    approach, _ = analyze_service_characteristics(args.csv_path, destination)

                    model = SingleServiceLstm(
                        csv_path=args.csv_path,
                        destination=destination,
                        approach=approach,
                        sequence_length=args.sequence_length
                    )

                    model.load_and_preprocess_data()
                    model.prepare_sequences()
                    model.build_adaptive_model()
                    model.train_adaptive(epochs=args.epochs, patience=args.patience)
                    rmse, mae, r_squared, mape = model.evaluate_adaptive()

                    results[destination] = {
                        'rmse': rmse, 'mae': mae, 'r_squared': r_squared, 'mape': mape,
                        'approach': approach, 'seed': args.seed
                    }
                else:
                    print(f"Using standard approach for {destination}")
                    approach = "standard"

                    model = SingleServiceLstm(
                        csv_path=args.csv_path,
                        destination=destination,
                        approach=approach,
                        sequence_length=args.sequence_length
                    )

                    model.load_and_preprocess_data()
                    model.prepare_sequences()
                    model.build_adaptive_model()
                    model.train_adaptive(epochs=min(args.epochs, 100), patience=args.patience)
                    rmse, mae, r_squared, mape = model.evaluate_adaptive()

                    results[destination] = {
                        'rmse': rmse, 'mae': mae, 'r_squared': r_squared, 'mape': mape,
                        'approach': approach, 'seed': args.seed
                    }

                models[destination] = model

            except Exception as e:
                print(f"Error training {destination}: {e}")
                import traceback

                traceback.print_exc()
                continue

    if results:
        results_data = []
        for destination, metrics in results.items():
            results_data.append({
                'destination': destination,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r_squared': metrics['r_squared'],
                'mape': metrics['mape'],
                'approach': metrics.get('approach', 'unknown'),
                'seed': args.seed
            })

        results_df = pd.DataFrame(results_data)

        metrics_file = os.path.join(seed_output_dir, f"all_models_metrics_seed{args.seed}.csv")
        results_df.to_csv(metrics_file, index=False)
        print(f"\nMetrics saved to: {metrics_file}")

        summary_file = os.path.join(seed_output_dir, f"summary_seed{args.seed}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"ADAPTIVE LSTM RESULTS SUMMARY - SEED {args.seed}\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Destination':<20} {'Approach':<10} {'R-squared':<12} {'MAPE':<8} {'RMSE':<12}\n")
            f.write("-" * 62 + "\n")

            for _, row in results_df.iterrows():
                f.write(
                    f"{row['destination']:<20} {row['approach']:<10} {row['r_squared']:<12.4f} {row['mape']:<8.2f} {row['rmse']:<12.2f}\n")

        print(f"Summary saved to: {summary_file}")

        print(f"\n{'=' * 60}")
        print(f"ADAPTIVE LSTM RESULTS SUMMARY - SEED {args.seed}")
        print(f"{'=' * 60}")
        print(f"{'Destination':<20} {'Approach':<10} {'R-squared':<12} {'MAPE':<8} {'RMSE':<12}")
        print("-" * 62)

        for _, row in results_df.iterrows():
            print(
                f"{row['destination']:<20} {row['approach']:<10} {row['r_squared']:<12.4f} {row['mape']:<8.2f} {row['rmse']:<12.2f}")

    print(f"\nTraining completed for seed {args.seed}")
    print(f"Results saved in: {seed_output_dir}")