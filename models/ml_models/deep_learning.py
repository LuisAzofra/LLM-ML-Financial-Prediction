"""
Modelos de Deep Learning para predicción financiera
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, Input,
    Attention, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """Modelo LSTM para predicción de series temporales financieras"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 10):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler_params = {}
    
    def build_model(self, units: List[int] = [128, 64, 32], 
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001) -> Model:
        """
        Construye el modelo LSTM
        """
        model = Sequential()
        
        # Primera capa LSTM
        model.add(LSTM(units[0], return_sequences=len(units) > 1,
                      input_shape=(self.seq_length, self.n_features)))
        model.add(Dropout(dropout_rate))
        
        # Capas LSTM adicionales
        for i, unit in enumerate(units[1:], 1):
            return_seq = i < len(units) - 1
            model.add(LSTM(unit, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Capas densas
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        # Compilar
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info(f"Modelo LSTM construido: {units}")
        return model
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str],
                    target_col: str, train_ratio: float = 0.8) -> Tuple:
        """
        Prepara datos para entrenamiento LSTM
        """
        # Normalizar
        data = df[feature_cols + [target_col]].values
        
        # Guardar parámetros de normalización
        for i, col in enumerate(feature_cols + [target_col]):
            self.scaler_params[col] = {
                'min': data[:, i].min(),
                'max': data[:, i].max()
            }
            data[:, i] = (data[:, i] - self.scaler_params[col]['min']) / \
                        (self.scaler_params[col]['max'] - self.scaler_params[col]['min'])
        
        # Crear secuencias
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length, :-1])
            y.append(data[i+self.seq_length, -1])
        
        X, y = np.array(X), np.array(y)
        
        # Dividir
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        Entrena el modelo LSTM
        """
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        """
        return self.model.predict(X, verbose=0)
    
    def inverse_transform_prediction(self, pred: np.ndarray, target_col: str) -> np.ndarray:
        """
        Desnormaliza las predicciones
        """
        params = self.scaler_params[target_col]
        return pred * (params['max'] - params['min']) + params['min']
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo
        """
        self.model.save(filepath)
        logger.info(f"Modelo guardado: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carga el modelo
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Modelo cargado: {filepath}")


class GRUModel:
    """Modelo GRU para predicción de series temporales"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 10):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def build_model(self, units: List[int] = [128, 64],
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001) -> Model:
        """
        Construye el modelo GRU
        """
        model = Sequential([
            GRU(units[0], return_sequences=len(units) > 1,
               input_shape=(self.seq_length, self.n_features)),
            Dropout(dropout_rate),
            GRU(units[1]) if len(units) > 1 else GRU(units[0]),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        Entrena el modelo GRU
        """
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        """
        return self.model.predict(X, verbose=0)


class CNNLSTMModel:
    """Modelo híbrido CNN-LSTM"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 10):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
    
    def build_model(self, cnn_filters: List[int] = [64, 32],
                   lstm_units: List[int] = [64, 32],
                   kernel_size: int = 3,
                   dropout_rate: float = 0.2) -> Model:
        """
        Construye modelo CNN-LSTM
        """
        model = Sequential()
        
        # Capas CNN
        for i, filters in enumerate(cnn_filters):
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                               activation='relu',
                               input_shape=(self.seq_length, self.n_features),
                               padding='same'))
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                               activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
        
        # Capas LSTM
        for i, units in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Capas densas
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info("Modelo CNN-LSTM construido")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        Entrena el modelo
        """
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        """
        return self.model.predict(X, verbose=0)


class AttentionLSTMModel:
    """Modelo LSTM con mecanismo de atención"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 10):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
    
    def build_model(self, lstm_units: int = 64,
                   attention_heads: int = 4,
                   dropout_rate: float = 0.2) -> Model:
        """
        Construye modelo LSTM con atención
        """
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        # LSTM bidireccional
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
        x = Dropout(dropout_rate)(x)
        
        # Atención multi-cabeza
        attention_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units
        )(x, x)
        
        x = LayerNormalization()(attention_output + x)
        
        # LSTM adicional
        x = LSTM(lstm_units // 2)(x)
        x = Dropout(dropout_rate)(x)
        
        # Capas densas
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info("Modelo Attention-LSTM construido")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        Entrena el modelo
        """
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        """
        return self.model.predict(X, verbose=0)


class DeepLearningEnsemble:
    """Ensemble de modelos de deep learning"""
    
    def __init__(self, seq_length: int = 60, n_features: int = 10):
        self.seq_length = seq_length
        self.n_features = n_features
        self.models = {}
        self.weights = {}
    
    def build_all_models(self):
        """
        Construye todos los modelos del ensemble
        """
        # LSTM
        lstm = LSTMModel(self.seq_length, self.n_features)
        lstm.build_model()
        self.models['lstm'] = lstm
        
        # GRU
        gru = GRUModel(self.seq_length, self.n_features)
        gru.build_model()
        self.models['gru'] = gru
        
        # CNN-LSTM
        cnn_lstm = CNNLSTMModel(self.seq_length, self.n_features)
        cnn_lstm.build_model()
        self.models['cnn_lstm'] = cnn_lstm
        
        # Attention-LSTM
        attention = AttentionLSTMModel(self.seq_length, self.n_features)
        attention.build_model()
        self.models['attention'] = attention
        
        logger.info("Todos los modelos construidos")
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 50):
        """
        Entrena todos los modelos
        """
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\nEntrenando {name}...")
            
            if hasattr(model, 'train'):
                history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)
                
                # Calcular error de validación
                val_loss = min(history['val_loss'])
                results[name] = val_loss
                logger.info(f"  Mejor val_loss: {val_loss:.4f}")
        
        # Calcular pesos basados en rendimiento (inverso del error)
        total_inv_error = sum(1/v for v in results.values())
        self.weights = {name: (1/error)/total_inv_error 
                       for name, error in results.items()}
        
        logger.info(f"\nPesos del ensemble: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción ponderada del ensemble
        """
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred.flatten())
            weights.append(self.weights.get(name, 1.0))
        
        # Promedio ponderado
        predictions = np.array(predictions)
        weights = np.array(weights) / sum(weights)
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa todos los modelos y el ensemble
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        results = {}
        
        # Evaluar modelos individuales
        for name, model in self.models.items():
            pred = model.predict(X_test).flatten()
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            results[name] = {'RMSE': rmse, 'MAE': mae}
        
        # Evaluar ensemble
        ensemble_pred = self.predict(X_test)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        results['ensemble'] = {'RMSE': ensemble_rmse, 'MAE': ensemble_mae}
        
        return results
