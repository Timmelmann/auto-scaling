import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
import warnings
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pickle
import random
from scipy.stats import zscore
import json

UNKNOWN_TOKEN_ID = 0
BIDIRECTIONAL_LSTM = False

class EdgeBasedModel(nn.Module):
    def __init__(self, source_vocab_size, destination_vocab_size, sequence_length,
                 embedding_dim=16, hidden_size=64, num_layers=2, dropout_rate=0.3,
                destination_names=None):

        super(EdgeBasedModel, self).__init__()

        self.sequence_length = sequence_length
        self.destination_vocab_size = destination_vocab_size
        self.destination_names = destination_names or []
        self.hidden_size = hidden_size

        self.source_embedding = nn.Embedding(source_vocab_size, embedding_dim)
        self.destination_embedding = nn.Embedding(destination_vocab_size, embedding_dim)

        nn.init.xavier_uniform_(self.source_embedding.weight)
        nn.init.xavier_uniform_(self.destination_embedding.weight)

        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        categorical_features = embedding_dim * 2
        combined_features = hidden_size + categorical_features

        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )

        self.prediction_heads = nn.ModuleDict()
        for i in range(destination_vocab_size):
            self.prediction_heads[f'head_{i}'] = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.2),
                nn.Linear(16, 1)
            )

        self.residual_projection = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.residual_projection.weight)

        self._initialize_linear_layers()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Created EdgeBasedModel:")
        print(f"   - Prediction heads: {len(self.prediction_heads)}")
        print(f"   - Total parameters: {total_params:,}")

    def _initialize_linear_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, source, destination, sequence):
        batch_size = source.size(0)
        device = source.device

        lstm_out, _ = self.lstm(sequence)
        temporal_features = lstm_out[:, -1, :]

        source_emb = self.embedding_dropout(self.source_embedding(source).squeeze(1))
        dest_emb = self.embedding_dropout(self.destination_embedding(destination).squeeze(1))

        combined = torch.cat([temporal_features, source_emb, dest_emb], dim=1)
        fused_features = self.feature_fusion(combined)
        shared_features = self.shared_layer(fused_features)

        outputs = torch.zeros(batch_size, 1, dtype=shared_features.dtype, device=device)

        for i in range(batch_size):
            dest_id = destination[i].item()

            head_name = f'head_{dest_id}'

            if head_name in self.prediction_heads:
                prediction = self.prediction_heads[head_name](shared_features[i:i + 1])

                sequence_mean = torch.mean(sequence[i])
                residual = self.residual_projection(sequence_mean.unsqueeze(0).unsqueeze(0))
                outputs[i] = prediction + 0.05 * residual
            else:
                fallback_head = list(self.prediction_heads.keys())[0]
                outputs[i] = self.prediction_heads[fallback_head](shared_features[i:i + 1])

        return outputs


class DataPreprocessor:
    def __init__(self, sequence_length=48, use_log_scaling=True, log_offset=1.0,
                 outlier_removal=True, outlier_threshold=3.0, min_samples_per_service=100):

        self.sequence_length = sequence_length
        self.use_log_scaling = use_log_scaling
        self.log_offset = log_offset
        self.outlier_removal = outlier_removal
        self.outlier_threshold = outlier_threshold
        self.min_samples_per_service = min_samples_per_service

        self.source_encoder = LabelEncoder()
        self.destination_encoder = LabelEncoder()
        self.value_scaler = RobustScaler()

        self.source_vocab_size = None
        self.destination_vocab_size = None
        self.destination_names = []
        self.destination_id_to_name = {}
        self.destination_name_to_id = {}

        self.pair_statistics = None
        self.stats_scaler = StandardScaler()

        warnings.filterwarnings('ignore')

    def _remove_outliers_isolation_forest(self, df):
        print("Removing outliers using Isolation Forest...")
        initial_count = len(df)

        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )

        features = df[['value']].copy()
        if len(df) > 10000:
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
            features = df[['value', 'hour', 'day_of_week']].copy()

        outlier_mask = isolation_forest.fit_predict(features) == 1
        df_clean = df[outlier_mask].copy()

        removed_count = initial_count - len(df_clean)
        print(f"   Removed {removed_count:,} outliers ({removed_count / initial_count:.1%})")

        return df_clean

    def _remove_outliers_statistical(self, df):
        print("Removing statistical outliers...")
        initial_count = len(df)

        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR

        df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)].copy()

        removed_count = initial_count - len(df_clean)
        print(f"   Removed {removed_count:,} outliers ({removed_count / initial_count:.1%})")

        return df_clean

    def _filter_low_sample_services(self, df):
        print(f"Filtering services with < {self.min_samples_per_service} samples...")

        service_counts = df['destination'].value_counts()
        valid_services = service_counts[service_counts >= self.min_samples_per_service].index

        df_filtered = df[df['destination'].isin(valid_services)].copy()

        removed_services = len(df['destination'].unique()) - len(valid_services)
        print(f"   Removed {removed_services} services, keeping {len(valid_services)} services")

        return df_filtered

    def fit(self, df_train):
        print("Fitting preprocessors...")
        df_work = df_train.copy()

        if self.outlier_removal:
            df_work = self._remove_outliers_statistical(df_work)

        df_work = self._filter_low_sample_services(df_work)

        unique_sources = list(df_work['source'].unique()) + ['UNKNOWN_SOURCE']
        unique_destinations = list(df_work['destination'].unique()) + ['UNKNOWN_DESTINATION']

        self.source_encoder.fit(unique_sources)
        self.destination_encoder.fit(unique_destinations)

        self.source_vocab_size = len(self.source_encoder.classes_)
        self.destination_vocab_size = len(self.destination_encoder.classes_)
        self.destination_names = unique_destinations

        self.destination_id_to_name = {
            self.destination_encoder.transform([name])[0]: name
            for name in unique_destinations
        }
        self.destination_name_to_id = {
            name: self.destination_encoder.transform([name])[0]
            for name in unique_destinations
        }

        self.unknown_source_id = self.source_encoder.transform(['UNKNOWN_SOURCE'])[0]
        self.unknown_dest_id = self.destination_encoder.transform(['UNKNOWN_DESTINATION'])[0]

        if self.use_log_scaling:
            df_work['value_log'] = np.log(df_work['value'] + self.log_offset)
            self.value_scaler.fit(df_work[['value_log']])
        else:
            self.value_scaler.fit(df_work[['value']])

        self._compute_pair_statistics(df_work)

        print(f"Preprocessor fitted successfully:")
        print(f"   - Services: {len(unique_destinations)}")
        print(f"   - Samples after cleaning: {len(df_work):,}")

        return df_work

    def _compute_pair_statistics(self, df_train):
        """Compute statistical features for service pairs"""
        pair_stats = df_train.groupby(['source', 'destination'])['value'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).reset_index()

        pair_stats['std'] = pair_stats['std'].fillna(0)
        pair_stats['range'] = pair_stats['max'] - pair_stats['min']

        total_requests = len(df_train)
        pair_stats['frequency'] = pair_stats['count'] / total_requests

        self.pair_statistics = pair_stats

        stats_cols = ['count', 'mean', 'std', 'frequency', 'range']
        self.stats_scaler.fit(pair_stats[stats_cols])

    def _safe_encode(self, values, encoder, unknown_id):
        """Safely encode values with unknown handling"""
        known_classes = set(encoder.classes_)
        return [encoder.transform([x])[0] if x in known_classes else unknown_id
                for x in values]

    def _get_pair_features(self, df):
        """Get enhanced statistical features for pairs"""
        if self.pair_statistics is None:
            return np.zeros((len(df), 5))

        df_with_stats = df.merge(
            self.pair_statistics,
            on=['source', 'destination'],
            how='left'
        )

        stats_cols = ['count', 'mean', 'std', 'frequency', 'range']
        global_stats = self.pair_statistics[stats_cols].mean()

        for col in stats_cols:
            df_with_stats[col] = df_with_stats[col].fillna(global_stats[col])

        return self.stats_scaler.transform(df_with_stats[stats_cols])

    def transform_and_prepare(self, df_input):
        """Enhanced data transformation"""
        df = df_input.copy()

        print(f"Transforming data (shape: {df.shape})...")

        df['source_encoded'] = self._safe_encode(df['source'], self.source_encoder, self.unknown_source_id)
        df['destination_encoded'] = self._safe_encode(df['destination'], self.destination_encoder, self.unknown_dest_id)

        if self.use_log_scaling:
            df['value_log'] = np.log(df['value'] + self.log_offset)
            df['value_scaled'] = self.value_scaler.transform(df[['value_log']])
        else:
            df['value_scaled'] = self.value_scaler.transform(df[['value']])

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values(['source_encoded', 'destination_encoded', 'datetime'])

        print(f"Data transformed successfully, shape: {df.shape}")
        return df

    def create_sequences_with_stride(self, df_processed):
        """Create sequences with adaptive stride for better data utilization"""
        print(f"Creating sequences (length: {self.sequence_length})...")

        if df_processed.empty:
            return [np.array([]) for _ in range(4)]

        unique_pairs = df_processed[
            ['source_encoded', 'destination_encoded', 'source', 'destination']
        ].drop_duplicates()

        X_sources, X_destinations, X_sequences, y_values = [], [], [], []

        for _, row in unique_pairs.iterrows():
            source_id, dest_id = row['source_encoded'], row['destination_encoded']
            source_name, dest_name = row['source'], row['destination']

            pair_data = df_processed[
                (df_processed['source_encoded'] == source_id) &
                (df_processed['destination_encoded'] == dest_id)
                ].copy()

            if len(pair_data) <= self.sequence_length + 5:
                continue

            n_available = len(pair_data) - self.sequence_length
            target_sequences = min(n_available, self.sequence_length * 2)
            stride = max(1, n_available // target_sequences)

            for i in range(0, len(pair_data) - self.sequence_length, stride):
                seq_data = pair_data.iloc[i:i + self.sequence_length]
                target = pair_data.iloc[i + self.sequence_length]

                X_sources.append(source_id)
                X_destinations.append(dest_id)
                X_sequences.append(seq_data[['value_scaled']].values)
                y_values.append(target['value_scaled'])

        if not X_sequences:
            print("No sequences created")
            return [np.array([]) for _ in range(4)]

        print(f"Created {len(X_sequences):,} sequences from {len(unique_pairs)} service pairs")

        return (
            np.array(X_sources),
            np.array(X_destinations),
            np.array(X_sequences),
            np.array(y_values)
        )


class DataAugmentator:
    def __init__(self, noise_std=0.05, trend_factor=0.1):
        self.noise_std = noise_std
        self.trend_factor = trend_factor

    def augment_batch(self, sequences, targets, augment_prob=0.3):
        """Apply data augmentation to training batch"""
        if np.random.random() > augment_prob:
            return sequences, targets

        augmented_sequences = sequences.clone()

        if np.random.random() < 0.5:
            noise = torch.normal(0, self.noise_std, size=sequences.shape)
            augmented_sequences += noise

        if np.random.random() < 0.3:
            batch_size, seq_len, features = sequences.shape
            for i in range(batch_size):
                if np.random.random() < 0.5:
                    trend = torch.linspace(0, self.trend_factor, seq_len).unsqueeze(1)
                    augmented_sequences[i] += trend * torch.randn(1) * 0.1

        return augmented_sequences, targets


class EdgeBasedTrainer:
    def __init__(self, model, early_stopping_patience=15, min_delta=1e-4):
        self.model = model
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.history = {
            'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [],
            'learning_rates': [], 'generalization_gap': []
        }

        self.augmentator = DataAugmentator()

    def _calculate_r2(self, y_true, y_pred):
        """Calculate R² score - returns Python float"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return float(r2.item())

    def _calculate_service_weights(self, destinations):
        """Calculate weights to balance service representation"""
        dest_counts = Counter(destinations.numpy())
        total = sum(dest_counts.values())
        weights = {}

        for dest_id, count in dest_counts.items():
            weights[dest_id] = total / (count * len(dest_counts))

        return weights

    def train(self, train_data, val_data, epochs=100, batch_size=64,
              learning_rate=0.0005, weight_decay=1e-4, use_scheduler=True):
        """Enhanced training with all improvements"""

        print("Training edge-based model...")

        X_sources_train, X_destinations_train, X_sequences_train, y_values_train = train_data
        X_sources_val, X_destinations_val, X_sequences_val, y_values_val = val_data

        train_tensors = [
            torch.tensor(X_sources_train, dtype=torch.long),
            torch.tensor(X_destinations_train, dtype=torch.long),
            torch.tensor(X_sequences_train, dtype=torch.float32),
            torch.tensor(y_values_train, dtype=torch.float32).reshape(-1, 1)
        ]

        val_tensors = [
            torch.tensor(X_sources_val, dtype=torch.long),
            torch.tensor(X_destinations_val, dtype=torch.long),
            torch.tensor(X_sequences_val, dtype=torch.float32),
            torch.tensor(y_values_val, dtype=torch.float32).reshape(-1, 1)
        ]

        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)

        service_weights = self._calculate_service_weights(train_tensors[1])
        sample_weights = [service_weights[dest.item()] for dest in train_tensors[1]]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training: {len(train_dataset):,} samples, Validation: {len(val_dataset):,} samples")

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_scheduler:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_preds, train_targets = 0.0, [], []

            for sources, destinations, sequences, targets in train_loader:
                sequences, targets = self.augmentator.augment_batch(sequences, targets)

                optimizer.zero_grad()
                outputs = self.model(sources, destinations, sequences)

                loss = criterion(outputs, targets)

                l1_reg = sum(p.abs().sum() for name, p in self.model.named_parameters()
                             if 'embedding' in name)
                loss += 1e-6 * l1_reg

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * sources.size(0)
                train_preds.append(outputs.detach())
                train_targets.append(targets)

            self.model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []

            with torch.no_grad():
                for sources, destinations, sequences, targets in val_loader:
                    outputs = self.model(sources, destinations, sequences)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * sources.size(0)
                    val_preds.append(outputs)
                    val_targets.append(targets)

            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)

            if train_preds and val_preds:
                train_preds_tensor = torch.cat(train_preds)
                train_targets_tensor = torch.cat(train_targets)
                val_preds_tensor = torch.cat(val_preds)
                val_targets_tensor = torch.cat(val_targets)

                train_r2 = self._calculate_r2(train_targets_tensor, train_preds_tensor)
                val_r2 = self._calculate_r2(val_targets_tensor, val_preds_tensor)
            else:
                train_r2 = val_r2 = 0.0

            generalization_gap = val_loss - train_loss
            current_lr = optimizer.param_groups[0]['lr']

            if use_scheduler:
                scheduler.step()
            else:
                scheduler.step(val_loss)

            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['train_r2'].append(float(train_r2))
            self.history['val_r2'].append(float(val_r2))
            self.history['generalization_gap'].append(float(generalization_gap))
            self.history['learning_rates'].append(float(current_lr))

            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
                  f"Gap: {generalization_gap:.4f} | LR: {current_lr:.6f}")

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': float(val_loss),
                    'val_r2': float(val_r2),
                    'history': self.history
                }, 'best_edge_model_checkpoint.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

            if generalization_gap > 0.5:
                print(f"Severe overfitting detected (gap: {generalization_gap:.4f})")
                break

            if current_lr < 1e-7:
                print(f"Learning rate too small ({current_lr}), stopping")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with val_loss: {self.best_val_loss:.4f}")

        self._plot_training_curves()
        return self.history

    def _plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_r2'], label='Train R²', alpha=0.8)
        axes[0, 1].plot(self.history['val_r2'], label='Validation R²', alpha=0.8)
        axes[0, 1].set_title('R² Score Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(self.history['learning_rates'], alpha=0.8)
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['generalization_gap'], alpha=0.8, color='red')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Warning level')
        axes[1, 0].set_title('Generalization Gap (Overfitting Monitor)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].fill_between(range(len(self.history['train_loss'])),
                                self.history['train_loss'],
                                self.history['val_loss'],
                                alpha=0.3, color='blue')
        axes[1, 1].plot(self.history['train_loss'], label='Train', alpha=0.8)
        axes[1, 1].plot(self.history['val_loss'], label='Validation', alpha=0.8)
        axes[1, 1].set_title('Loss Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        if len(self.history['val_r2']) > 1:
            r2_improvement = np.diff(self.history['val_r2'])
            axes[1, 2].plot(r2_improvement, alpha=0.8, color='green')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].set_title('Validation R² Improvement per Epoch')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('edge_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class EdgeBasedEvaluator:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        epsilon = 1e-10

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean(np.square(y_pred - y_true))))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r_squared = float(1 - (ss_residual / (ss_total + epsilon)))

        median_ae = float(np.median(np.abs(y_pred - y_true)))
        max_error = float(np.max(np.abs(y_pred - y_true)))

        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = float(np.mean(true_direction == pred_direction) * 100)
        else:
            directional_accuracy = 0.0

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r_squared': r_squared,
            'median_ae': median_ae,
            'max_error': max_error,
            'directional_accuracy': directional_accuracy,
            'count': int(len(y_true))
        }

    def evaluate_comprehensive(self, test_data):
        print("\n=== EDGE-BASED MODEL EVALUATION ===")

        X_sources_test, X_destinations_test, X_sequences_test, y_values_test = test_data

        if X_sequences_test.size == 0:
            print("No test data available")
            return None, None

        test_tensors = [
            torch.tensor(X_sources_test, dtype=torch.long),
            torch.tensor(X_destinations_test, dtype=torch.long),
            torch.tensor(X_sequences_test, dtype=torch.float32),
            torch.tensor(y_values_test, dtype=torch.float32).reshape(-1, 1)
        ]

        test_dataset = TensorDataset(*test_tensors)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        self.model.eval()
        all_predictions, all_targets, all_destinations = [], [], []

        with torch.no_grad():
            for sources, destinations, sequences, targets in test_loader:
                outputs = self.model(sources, destinations, sequences)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_destinations.append(destinations.cpu().numpy())

        y_pred_scaled = np.vstack(all_predictions).flatten()
        y_true_scaled = np.vstack(all_targets).flatten()
        destinations_array = np.hstack(all_destinations)

        y_pred_original = self.preprocessor.value_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true_original = self.preprocessor.value_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        if self.preprocessor.use_log_scaling:
            y_pred_original = np.exp(y_pred_original) - self.preprocessor.log_offset
            y_true_original = np.exp(y_true_original) - self.preprocessor.log_offset

        overall_metrics = self.calculate_comprehensive_metrics(y_true_original, y_pred_original)

        print(f"Overall Test Results:")
        print(f"   MAE: {overall_metrics['mae']:.0f}")
        print(f"   RMSE: {overall_metrics['rmse']:.0f}")
        print(f"   MAPE: {overall_metrics['mape']:.2f}%")
        print(f"   R²: {overall_metrics['r_squared']:.4f}")
        print(f"   Median AE: {overall_metrics['median_ae']:.0f}")
        print(f"   Directional Accuracy: {overall_metrics['directional_accuracy']:.1f}%")
        print(f"   Samples: {overall_metrics['count']:,}")

        print(f"\nPer-Service Detailed Metrics:")
        print("=" * 84)

        service_metrics = []
        unique_dest_ids = np.unique(destinations_array)

        for dest_id in unique_dest_ids:
            service_name = self.preprocessor.destination_id_to_name.get(dest_id, f"Unknown_{dest_id}")

            service_mask = destinations_array == dest_id
            service_y_true = y_true_original[service_mask]
            service_y_pred = y_pred_original[service_mask]

            if len(service_y_true) < 5:
                continue

            service_metric = self.calculate_comprehensive_metrics(service_y_true, service_y_pred)
            service_metric['service'] = service_name
            service_metric['dest_id'] = int(dest_id)

            service_metrics.append(service_metric)

            print(f"{service_name}")
            print(f"   MAE:  {service_metric['mae']:>10.0f}")
            print(f"   RMSE: {service_metric['rmse']:>10.0f}")
            print(f"   MAPE: {service_metric['mape']:>6.2f}%")
            print(f"   R²:   {service_metric['r_squared']:>7.4f}")
            print(f"   Count: {service_metric['count']:>6,}")
            print("-" * 40)

        service_metrics_df = pd.DataFrame(service_metrics)
        if not service_metrics_df.empty:
            service_metrics_df = service_metrics_df.sort_values('rmse')

            print(f"\nService Performance Summary (sorted by RMSE):")
            print("=" * 84)

            for _, row in service_metrics_df.iterrows():
                print(f"{row['service']:<18} | "
                      f"MAE: {row['mae']:>8.0f} | "
                      f"RMSE: {row['rmse']:>8.0f} | "
                      f"MAPE: {row['mape']:>6.2f}% | "
                      f"R²: {row['r_squared']:>7.4f} | "
                      f"N: {row['count']:>6,}")

            print(f"\nSummary Statistics Across Services:")
            print(
                f"   Best RMSE:     {service_metrics_df['rmse'].min():.0f} ({service_metrics_df.loc[service_metrics_df['rmse'].idxmin(), 'service']})")
            print(
                f"   Worst RMSE:    {service_metrics_df['rmse'].max():.0f} ({service_metrics_df.loc[service_metrics_df['rmse'].idxmax(), 'service']})")
            print(f"   Mean RMSE:     {service_metrics_df['rmse'].mean():.0f}")
            print(f"   RMSE Std:      {service_metrics_df['rmse'].std():.0f}")
            print(f"   Services > 0.5 R²: {(service_metrics_df['r_squared'] > 0.5).sum()}/{len(service_metrics_df)}")
            print(f"   Services > 0.0 R²: {(service_metrics_df['r_squared'] > 0.0).sum()}/{len(service_metrics_df)}")

            service_metrics_df.to_csv('edge_per_service_metrics.csv', index=False)
            print(f"\nDetailed metrics saved to: edge_per_service_metrics.csv")

        return overall_metrics, service_metrics_df


def temporal_split(df, train_ratio=0.7, val_ratio=0.15, min_service_samples=10):
    """Enhanced temporal split with service balance checking"""
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    n_total = len(df_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    df_train = df_sorted.iloc[:n_train].copy()
    df_val = df_sorted.iloc[n_train:n_train + n_val].copy()
    df_test = df_sorted.iloc[n_train + n_val:].copy()

    print(f"Temporal split - Train: {len(df_train):,} ({len(df_train) / n_total:.1%}), "
          f"Val: {len(df_val):,} ({len(df_val) / n_total:.1%}), "
          f"Test: {len(df_test):,} ({len(df_test) / n_total:.1%})")

    train_end = df_train['timestamp'].max()
    val_start = df_val['timestamp'].min()
    val_end = df_val['timestamp'].max()
    test_start = df_test['timestamp'].min()

    assert train_end <= val_start, "Temporal overlap detected!"
    assert val_end <= test_start, "Temporal overlap detected!"

    train_services = set(df_train['destination'].unique())
    val_services = set(df_val['destination'].unique())
    test_services = set(df_test['destination'].unique())

    common_services = train_services & val_services & test_services
    print(f"Services in all splits: {len(common_services)}/{len(train_services)} total")

    if len(common_services) < len(train_services) * 0.8:
        print("Warning: Some services may have insufficient representation in validation/test sets")

    print("Temporal split verified - no data leakage")
    return df_train, df_val, df_test


def train_edge_based_model(df, args):

    df_train, df_val, df_test = temporal_split(df, train_ratio=0.7, val_ratio=0.15)

    preprocessor = DataPreprocessor(
        sequence_length=48,
        use_log_scaling=args.log_scaling,
        outlier_removal=True,
        min_samples_per_service=100,
    )

    df_train_clean = preprocessor.fit(df_train)

    df_train_processed = preprocessor.transform_and_prepare(df_train_clean)
    df_val_processed = preprocessor.transform_and_prepare(df_val)
    df_test_processed = preprocessor.transform_and_prepare(df_test)

    train_data = preprocessor.create_sequences_with_stride(df_train_processed)
    val_data = preprocessor.create_sequences_with_stride(df_val_processed)
    test_data = preprocessor.create_sequences_with_stride(df_test_processed)

    if train_data[0].size == 0:
        print("No training sequences created")
        return

    print(f"\nFinal sequence counts:")
    print(f"   Training: {len(train_data[0]):,} sequences")
    print(f"   Validation: {len(val_data[0]):,} sequences")
    print(f"   Test: {len(test_data[0]):,} sequences")

    model = EdgeBasedModel(
        source_vocab_size=preprocessor.source_vocab_size,
        destination_vocab_size=preprocessor.destination_vocab_size,
        sequence_length=48,
        embedding_dim=16,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0.3,
        destination_names=preprocessor.destination_names
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Heads: {len(model.prediction_heads)}")

    trainer = EdgeBasedTrainer(model, early_stopping_patience=15)
    history = trainer.train(
        train_data, val_data,
        epochs=150,
        batch_size=64,
        learning_rate=0.0003,
        weight_decay=1e-4,
        use_scheduler=True
    )

    overall_metrics = None
    if test_data[0].size > 0:
        evaluator = EdgeBasedEvaluator(model, preprocessor)
        overall_metrics, service_metrics_df = evaluator.evaluate_comprehensive(test_data)

        if service_metrics_df is not None and not service_metrics_df.empty:
            negative_r2_count = (service_metrics_df['r_squared'] < 0).sum()
            good_r2_count = (service_metrics_df['r_squared'] > 0.3).sum()

            print(f"\nModel Performance Analysis:")
            print(f"   Services with negative R²: {negative_r2_count}/{len(service_metrics_df)}")
            print(f"   Services with R² > 0.3: {good_r2_count}/{len(service_metrics_df)}")
            print(
                f"   Overall improvement: {'Significant' if negative_r2_count < len(service_metrics_df) * 0.2 else 'Needs work'}")

    model_path = os.path.join(args.model_dir, f"{args.model_name}_edge.pt")
    os.makedirs(args.model_dir, exist_ok=True)

    model_config = {
        'source_vocab_size': int(preprocessor.source_vocab_size),
        'destination_vocab_size': int(preprocessor.destination_vocab_size),
        'sequence_length': 48,
        'embedding_dim': 16,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_rate': 0.3,
        'destination_names': preprocessor.destination_names
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_history': history,
        'overall_metrics': overall_metrics
    }, model_path)

    with open(f"{model_path}_preprocessor.pkl", 'wb') as f:
        pickle.dump({
            'preprocessor': preprocessor,
            'args': vars(args),
            'improvements_applied': [
                'outlier_removal', 'batch_normalization',
                'residual_connections', 'data_augmentation', 'weighted_sampling',
                'improved_optimization', 'enhanced_early_stopping'
            ]
        }, f)

    print(f"\nEdge-based model saved to: {model_path}")
    print("Edge-based model training completed!")

    return model, preprocessor, history


def parse_arguments():
    parser = argparse.ArgumentParser(description='Edge-Based HTTP Request Prediction Model')
    parser.add_argument('--data_file', type=str, default='../../datasets/transformed_http_1m_7d.csv')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--model_name', type=str, default='http_lstm_edgebased')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train')
    parser.add_argument('--log_scaling', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()
    set_seeds(args.seed)

    print("Loading data...")
    df = pd.read_csv(args.data_file)

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Unique services: {df['destination'].nunique()}")
    print(f"Services: {df['destination'].unique()}")

    if args.mode == 'train':
        train_edge_based_model(df, args)
    else:
        print(f"Mode '{args.mode}' not implemented yet for edge-based version")


if __name__ == "__main__":
    main()