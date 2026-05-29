"""
Sistema Híbrido ML + LLM
Integra predicciones cuantitativas con análisis cualitativo
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ml_models.traditional_ml import TraditionalMLModels, FeatureEngineer, VolatilityPredictor
from models.ml_models.deep_learning import LSTMModel, DeepLearningEnsemble
from agents.base_agent import AnalysisResult


class HybridPredictor:
    """
    Sistema híbrido que combina modelos ML tradicionales, deep learning y LLMs
    para predicción financiera y detección de patrones
    """
    
    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble
        
        # Modelos ML tradicionales
        self.ml_models = TraditionalMLModels()
        self.feature_engineer = FeatureEngineer()
        self.volatility_predictor = VolatilityPredictor()
        
        # Modelos Deep Learning
        self.lstm_model = None
        self.dl_ensemble = None
        
        # Resultados
        self.predictions = {}
        self.hybrid_scores = {}
        
        logger.info("Sistema Híbrido ML+LLM inicializado")
    
    def train_ml_models(self, df: pd.DataFrame, target_col: str = 'Close',
                       prediction_horizon: int = 5) -> Dict:
        """
        Entrena modelos ML tradicionales
        """
        logger.info("=" * 60)
        logger.info("ENTRENANDO MODELOS ML TRADICIONALES")
        logger.info("=" * 60)
        
        # Crear características
        df_features = self.feature_engineer.create_features(
            df, target_col, prediction_horizon
        )
        
        feature_cols = self.feature_engineer.feature_names
        
        # Preparar datos
        X, y = self.feature_engineer.prepare_ml_data(
            df_features, feature_cols, 'target_return', seq_length=10
        )
        
        # Dividir temporalmente
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Datos: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Características: {len(feature_cols)}")
        
        # Entrenar modelos
        results = {}
        
        # Random Forest
        logger.info("\n🌲 Entrenando Random Forest...")
        rf_result = self.ml_models.train_random_forest(X_train, y_train)
        results['random_forest'] = rf_result
        
        # XGBoost
        logger.info("\n🚀 Entrenando XGBoost...")
        xgb_result = self.ml_models.train_xgboost(X_train, y_train)
        results['xgboost'] = xgb_result
        
        # LightGBM
        logger.info("\n💡 Entrenando LightGBM...")
        lgb_result = self.ml_models.train_lightgbm(X_train, y_train)
        results['lightgbm'] = lgb_result
        
        # Evaluar
        logger.info("\n📊 Evaluando modelos...")
        eval_results = self.ml_models.evaluate_models(X_test, y_test)
        logger.info(f"\n{eval_results}")
        
        return results
    
    def train_deep_learning(self, df: pd.DataFrame, feature_cols: List[str],
                           target_col: str = 'Close', seq_length: int = 60) -> Dict:
        """
        Entrena modelos de Deep Learning
        """
        logger.info("=" * 60)
        logger.info("ENTRENANDO MODELOS DEEP LEARNING")
        logger.info("=" * 60)
        
        if self.use_ensemble:
            # Usar ensemble de modelos DL
            self.dl_ensemble = DeepLearningEnsemble(
                seq_length=seq_length,
                n_features=len(feature_cols)
            )
            
            self.dl_ensemble.build_all_models()
            
            # Preparar datos
            data = df[feature_cols + [target_col]].values
            
            # Normalizar
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Crear secuencias
            X, y = [], []
            for i in range(len(data_scaled) - seq_length):
                X.append(data_scaled[i:i+seq_length, :-1])
                y.append(data_scaled[i+seq_length, -1])
            
            X, y = np.array(X), np.array(y)
            
            # Dividir
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            val_split = int(len(X_train) * 0.9)
            X_train, X_val = X_train[:val_split], X_train[val_split:]
            y_train, y_val = y_train[:val_split], y_train[val_split:]
            
            logger.info(f"Datos: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Entrenar ensemble
            logger.info("\n🧠 Entrenando ensemble de modelos DL...")
            self.dl_ensemble.train_all(X_train, y_train, X_val, y_val, epochs=30)
            
            # Evaluar
            results = self.dl_ensemble.evaluate(X_test, y_test)
            
            logger.info("\n📊 Resultados del ensemble:")
            for model_name, metrics in results.items():
                logger.info(f"  {model_name}: {metrics}")
            
            return results
        
        else:
            # Usar solo LSTM
            self.lstm_model = LSTMModel(seq_length=seq_length, n_features=len(feature_cols))
            self.lstm_model.build_model(units=[128, 64, 32])
            
            X_train, X_test, y_train, y_test = self.lstm_model.prepare_data(
                df, feature_cols, target_col
            )
            
            history = self.lstm_model.train(X_train, y_train, X_test, y_test, epochs=50)
            
            return {'lstm': history}
    
    def train_volatility_model(self, df: pd.DataFrame) -> Dict:
        """
        Entrena modelo especializado en predicción de volatilidad
        """
        logger.info("=" * 60)
        logger.info("ENTRENANDO MODELO DE VOLATILIDAD")
        logger.info("=" * 60)
        
        self.volatility_predictor.train(df)
        
        return {'status': 'trained'}
    
    def predict(self, df: pd.DataFrame, feature_cols: List[str],
               include_ml: bool = True, include_dl: bool = True) -> Dict:
        """
        Realiza predicciones con todos los modelos
        """
        predictions = {}
        
        # Predicciones ML tradicionales
        if include_ml and self.ml_models.models:
            logger.info("\n🔮 Prediciendo con modelos ML...")
            
            # Preparar datos
            latest_data = df[feature_cols].iloc[-10:].values.flatten().reshape(1, -1)
            
            for name, model in self.ml_models.models.items():
                pred = model.predict(latest_data)[0]
                predictions[f'ml_{name}'] = {
                    'predicted_return': pred,
                    'confidence': 0.7
                }
        
        # Predicciones Deep Learning
        if include_dl:
            if self.dl_ensemble:
                logger.info("\n🧠 Prediciendo con ensemble DL...")

                # Preparar secuencia: el scaler debe haberse ajustado sobre el histórico
                # de entrenamiento (accesible vía self.dl_ensemble.scaler). Ajustar
                # un nuevo MinMaxScaler sobre el último segmento introducía fuga y
                # degradaba la predicción — ahora reutilizamos el scaler entrenado.
                seq_length = self.dl_ensemble.seq_length
                latest_sequence = df[feature_cols].iloc[-seq_length:].values

                scaler = getattr(self.dl_ensemble, 'scaler', None)
                if scaler is not None:
                    try:
                        latest_scaled = scaler.transform(latest_sequence)
                    except Exception:
                        # Fallback seguro: usar todo el histórico disponible para ajustar
                        from sklearn.preprocessing import MinMaxScaler
                        fallback = MinMaxScaler().fit(df[feature_cols].values)
                        latest_scaled = fallback.transform(latest_sequence)
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    fallback = MinMaxScaler().fit(df[feature_cols].values)
                    latest_scaled = fallback.transform(latest_sequence)

                X_pred = np.array([latest_scaled])

                pred = self.dl_ensemble.predict(X_pred)[0]
                predictions['dl_ensemble'] = {
                    'predicted_price': pred,
                    'confidence': 0.75
                }
        
        # Predicción de volatilidad
        if self.volatility_predictor.models:
            logger.info("\n📈 Prediciendo volatilidad...")
            
            # Preparar características de volatilidad
            df_vol = self.volatility_predictor.create_volatility_features(df)
            feature_cols_vol = [col for col in df_vol.columns 
                              if col.startswith(('hist_vol_', 'abs_return_', 'return_sq_'))]
            
            latest_vol = df_vol[feature_cols_vol].iloc[-1:].values
            vol_preds = self.volatility_predictor.predict(latest_vol)
            
            predictions['volatility'] = {
                model: float(pred[0]) for model, pred in vol_preds.items()
            }
        
        self.predictions = predictions
        return predictions
    
    def combine_with_llm_analysis(self, ml_predictions: Dict, 
                                  llm_analysis: AnalysisResult) -> Dict:
        """
        Combina predicciones ML con análisis LLM
        """
        logger.info("=" * 60)
        logger.info("COMBINANDO ML + LLM")
        logger.info("=" * 60)
        
        # Extraer señal del análisis LLM
        llm_signal = self._llm_recommendation_to_signal(llm_analysis.recommendation)
        llm_confidence = llm_analysis.confidence
        
        # Calcular señal promedio de ML
        ml_signals = []
        for model_name, pred in ml_predictions.items():
            if 'predicted_return' in pred:
                signal = 1 if pred['predicted_return'] > 0 else -1
                ml_signals.append(signal * pred.get('confidence', 0.5))
            elif 'predicted_price' in pred:
                # Comparar con precio actual
                signal = 1  # Simplificado
                ml_signals.append(signal * pred.get('confidence', 0.5))
        
        avg_ml_signal = np.mean(ml_signals) if ml_signals else 0
        
        # Pesos para combinación
        ml_weight = 0.4
        llm_weight = 0.6
        
        # Combinar señales
        hybrid_score = (avg_ml_signal * ml_weight + llm_signal * llm_weight)
        hybrid_confidence = (0.7 * ml_weight + llm_confidence * llm_weight)
        
        # Determinar recomendación final
        if hybrid_score > 0.5:
            final_recommendation = 'COMPRA_FUERTE'
        elif hybrid_score > 0.2:
            final_recommendation = 'COMPRA'
        elif hybrid_score < -0.5:
            final_recommendation = 'VENTA_FUERTE'
        elif hybrid_score < -0.2:
            final_recommendation = 'VENTA'
        else:
            final_recommendation = 'MANTENER'
        
        result = {
            'hybrid_score': hybrid_score,
            'hybrid_confidence': hybrid_confidence,
            'final_recommendation': final_recommendation,
            'ml_signal': avg_ml_signal,
            'llm_signal': llm_signal,
            'ml_predictions': ml_predictions,
            'llm_analysis': {
                'recommendation': llm_analysis.recommendation,
                'confidence': llm_analysis.confidence,
                'reasoning': llm_analysis.reasoning
            }
        }
        
        self.hybrid_scores = result
        
        logger.info(f"\n📊 Resultado Híbrido:")
        logger.info(f"   Score: {hybrid_score:.3f}")
        logger.info(f"   Confianza: {hybrid_confidence:.3f}")
        logger.info(f"   Recomendación: {final_recommendation}")
        
        return result
    
    def _llm_recommendation_to_signal(self, recommendation: str) -> float:
        """Convierte recomendación LLM a señal numérica"""
        mapping = {
            'COMPRA_FUERTE': 1.0,
            'COMPRA': 0.7,
            'APROBAR': 0.5,
            'APROBAR_CON_PRECAUCION': 0.3,
            'MANTENER': 0.0,
            'HOLD': 0.0,
            'NEUTRO': 0.0,
            'RECHAZAR': -0.3,
            'VENTA': -0.7,
            'VENTA_FUERTE': -1.0
        }
        return mapping.get(recommendation, 0.0)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene importancia de características de los modelos"""
        importance_data = []
        
        if 'random_forest' in self.ml_models.models:
            rf_model = self.ml_models.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                for i, imp in enumerate(rf_model.feature_importances_):
                    importance_data.append({
                        'feature': f'feature_{i}',
                        'importance': imp,
                        'model': 'random_forest'
                    })
        
        return pd.DataFrame(importance_data)
    
    def save_models(self, path: str):
        """Guarda todos los modelos entrenados"""
        import joblib
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelos ML
        self.ml_models.save_models(f"{path}/ml_models")
        
        # Guardar modelos DL
        if self.lstm_model:
            self.lstm_model.save_model(f"{path}/lstm_model.h5")
        
        # Guardar configuración
        config = {
            'use_ensemble': self.use_ensemble,
            'feature_names': self.feature_engineer.feature_names
        }
        
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Modelos guardados en {path}")
    
    def load_models(self, path: str):
        """Carga modelos previamente entrenados"""
        import json
        
        # Cargar configuración
        with open(f"{path}/config.json", 'r') as f:
            config = json.load(f)
        
        self.use_ensemble = config['use_ensemble']
        
        # Cargar modelos ML
        self.ml_models.load_models(f"{path}/ml_models")
        
        # Cargar modelos DL
        if os.path.exists(f"{path}/lstm_model.h5"):
            self.lstm_model = LSTMModel()
            self.lstm_model.load_model(f"{path}/lstm_model.h5")
        
        logger.info(f"Modelos cargados desde {path}")


class HybridTradingSystem:
    """
    Sistema de trading completo que integra ML, Deep Learning y LLMs
    """
    
    def __init__(self):
        self.hybrid_predictor = HybridPredictor()
        self.communication_bus = None
        self.portfolio_manager = None
        
    def setup_agents(self, communication_bus, portfolio_manager):
        """Configura los agentes del sistema"""
        self.communication_bus = communication_bus
        self.portfolio_manager = portfolio_manager
    
    def run_analysis(self, symbol: str, df: pd.DataFrame, 
                    news_data: List[Dict] = None) -> Dict:
        """
        Ejecuta análisis completo híbrido
        """
        logger.info("=" * 80)
        logger.info(f"ANÁLISIS HÍBRIDO COMPLETO: {symbol}")
        logger.info("=" * 80)
        
        # 1. Entrenar modelos si es necesario
        if not self.hybrid_predictor.ml_models.models:
            feature_cols = ['Close', 'Volume'] + [col for col in df.columns 
                         if col.startswith(('SMA_', 'RSI', 'MACD', 'BB_'))]
            self.hybrid_predictor.train_ml_models(df)
        
        # 2. Realizar predicciones ML
        feature_cols = [col for col in df.columns if col not in ['target_return', 'target_direction', 'target_price']]
        ml_predictions = self.hybrid_predictor.predict(df, feature_cols)
        
        # 3. Ejecutar análisis de agentes
        if self.portfolio_manager:
            data = {
                'symbol': symbol,
                'price_data': df,
                'news': news_data or [],
                'ml_prediction': ml_predictions.get('ml_xgboost', {}),
                'portfolio': self.portfolio_manager.portfolio
            }
            
            final_analysis = self.portfolio_manager.analyze(data)
            
            # 4. Combinar ML + LLM
            hybrid_result = self.hybrid_predictor.combine_with_llm_analysis(
                ml_predictions, final_analysis
            )
            
            return hybrid_result
        
        return ml_predictions
