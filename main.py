"""
================================================================================
SISTEMA HÍBRIDO ML + LLM PARA PREDICCIÓN FINANCIERA
Trabajo Fin de Grado
================================================================================

Este script ejecuta el sistema completo con:
- Datos financieros reales (Yahoo Finance)
- Noticias reales (RSS feeds)
- LLM gratuito (Ollama/HuggingFace)
- Predicción de volatilidad GARCH
- Backtesting económico riguroso
"""

from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos del sistema
from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor
from data.news_fetcher import NewsFetcher
from models.pattern_detection.chart_patterns import PatternDetector
from models.ml_models.traditional_ml import TraditionalMLModels, FeatureEngineer
from models.ml_models.volatility_models import GARCHVolatilityModel, calculate_realized_volatility
from agents.technical_agent.technical_analyst import TechnicalAnalystAgent
from agents.sentiment_agent.sentiment_analyst import SentimentAnalystAgent
from agents.risk_agent.risk_manager import RiskManagerAgent
from agents.manager_agent.portfolio_manager import PortfolioManagerAgent
from agents.base_agent import AgentCommunicationBus
from evaluation.backtest import Backtester, RobustnessTester


def load_and_prepare_data(symbol: str, start_date: str, end_date: str, 
                         asset_type: str = 'stock') -> pd.DataFrame:
    """
    Carga y prepara datos reales para análisis.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"CARGA DE DATOS REALES: {symbol}")
    logger.info(f"{'='*70}")
    
    # Cargar datos
    loader = FinancialDataLoader()
    df = loader.download_stock_data(symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"❌ No se pudieron cargar datos para {symbol}")
        return pd.DataFrame()
    
    # Procesar datos
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    
    logger.info(f"✅ Datos cargados exitosamente:")
    logger.info(f"   Símbolo: {symbol}")
    logger.info(f"   Registros: {len(df)}")
    logger.info(f"   Período: {df.index[0].date()} a {df.index[-1].date()}")
    logger.info(f"   Precio actual: ${df['Close'].iloc[-1]:.2f}")
    logger.info(f"   Volatilidad (anual): {df['Volatility'].iloc[-1]*100:.1f}%")
    
    return df


def train_volatility_model(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Entrena modelo GARCH para predicción de volatilidad.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"PREDICCIÓN DE VOLATILIDAD GARCH: {symbol}")
    logger.info(f"{'='*70}")
    
    try:
        # Calcular retornos
        returns = df['Close'].pct_change().dropna().values
        
        # Crear modelo GARCH
        garch = GARCHVolatilityModel(p=1, q=1, model_type='GARCH', distribution='t')
        
        # Ajustar modelo
        metrics = garch.fit(returns, disp='off')
        
        # Realizar forecast
        forecast = garch.forecast(horizon=5)
        
        logger.info(f"\n📊 Modelo GARCH ajustado:")
        logger.info(f"   AIC: {metrics['aic']:.2f}")
        logger.info(f"   BIC: {metrics['bic']:.2f}")
        logger.info(f"\n🔮 Predicción de volatilidad (5 días):")
        logger.info(f"   Valor esperado: {forecast.forecast_value*100:.2f}%")
        logger.info(f"   Intervalo 95%: [{forecast.confidence_interval[0]*100:.2f}%, {forecast.confidence_interval[1]*100:.2f}%]")
        
        return {
            'model': garch,
            'forecast': forecast,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Error en modelo GARCH: {e}")
        return None


def train_ml_models(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Entrena modelos de Machine Learning.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"ENTRENAMIENTO ML: {symbol}")
    logger.info(f"{'='*70}")
    
    # Preparar características
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df, prediction_horizon=5)
    
    feature_cols = feature_engineer.feature_names
    
    # Preparar datos
    X, y = feature_engineer.prepare_ml_data(df_features, feature_cols, 'target_return')
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"📊 Datos: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"📊 Características: {len(feature_cols)}")
    
    # Entrenar modelos
    ml_models = TraditionalMLModels()
    
    logger.info("\n🌲 Entrenando Random Forest...")
    ml_models.train_random_forest(X_train, y_train)
    
    logger.info("\n🚀 Entrenando XGBoost...")
    ml_models.train_xgboost(X_train, y_train)
    
    logger.info("\n💡 Entrenando LightGBM...")
    ml_models.train_lightgbm(X_train, y_train)
    
    # Evaluar
    logger.info("\n📊 Evaluando modelos...")
    results = ml_models.evaluate_models(X_test, y_test)
    
    print("\n" + results.to_string())
    
    return {
        'models': ml_models,
        'results': results,
        'feature_engineer': feature_engineer
    }


def run_multi_agent_analysis(df: pd.DataFrame, 
                             symbol: str, 
                             ml_prediction: Dict = None,
                             use_llm: bool = True,
                             llm_provider: str = "ollama") -> Dict:
    """
    Ejecuta análisis multi-agente con LLM real.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"ANÁLISIS MULTI-AGENTE: {symbol}")
    logger.info(f"{'='*70}")
    
    # Crear bus de comunicación
    comm_bus = AgentCommunicationBus()
    
    # Crear agentes
    technical_agent = TechnicalAnalystAgent()
    sentiment_agent = SentimentAnalystAgent(use_llm=use_llm, llm_provider=llm_provider)
    risk_agent = RiskManagerAgent(use_garch=True)
    portfolio_manager = PortfolioManagerAgent(comm_bus, use_llm=use_llm, llm_provider=llm_provider)
    
    # Registrar agentes
    comm_bus.register_agent(technical_agent)
    comm_bus.register_agent(sentiment_agent)
    comm_bus.register_agent(risk_agent)
    comm_bus.register_agent(portfolio_manager)
    
    # Preparar datos
    data = {
        'symbol': symbol,
        'price_data': df,
        'ml_prediction': ml_prediction,
        'portfolio': portfolio_manager.portfolio
    }
    
    # Ejecutar análisis
    result = portfolio_manager.analyze(data)
    
    logger.info(f"\n✅ Análisis completado")
    logger.info(f"   Decisión: {result.recommendation}")
    logger.info(f"   Confianza: {result.confidence:.2f}")
    
    return {
        'result': result,
        'portfolio_manager': portfolio_manager
    }


def run_backtest_rigorous(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Ejecuta backtesting económico riguroso.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTESTING ECONÓMICO RIGUROSO: {symbol}")
    logger.info(f"{'='*70}")
    
    backtester = Backtester(initial_capital=100000)
    
    # Crear generador de señales basado en cruce de medias móviles
    def signal_generator(data):
        if len(data) < 50:
            return 0
        
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1]
        current = data['Close'].iloc[-1]
        
        # Señal de cruce
        if current > sma_20 > sma_50:
            return 1
        elif current < sma_20 < sma_50:
            return -1
        return 0
    
    # Ejecutar backtest
    bt_result = backtester.run_backtest(df, signal_generator, symbol)
    
    # Imprimir reporte completo
    backtester.print_report(bt_result)
    
    # Test de robustez por regímenes
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST DE ROBUSTEZ POR REGÍMENES")
    logger.info(f"{'='*70}")
    
    robustness_tester = RobustnessTester(backtester)
    regime_results = robustness_tester.test_regime_robustness(df, signal_generator, symbol)
    
    # Calcular score de robustez
    robustness_score = robustness_tester.calculate_robustness_score(regime_results)
    
    logger.info(f"\n📊 Score de Robustez: {robustness_score['overall']:.2f}")
    
    return {
        'backtest_result': bt_result,
        'regime_results': regime_results,
        'robustness_score': robustness_score
    }


def generate_final_report(symbol: str, results: Dict):
    """
    Genera reporte final del análisis.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"REPORTE FINAL: {symbol}")
    logger.info(f"{'='*70}")
    
    report = []
    report.append("=" * 80)
    report.append("ESTUDIO PROSPECTIVO DE LLMs Y ML EN PREDICCIÓN FINANCIERA")
    report.append("Sistema Híbrido con Datos Reales y LLM Gratuito")
    report.append("=" * 80)
    report.append(f"\nSímbolo analizado: {symbol}")
    report.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Resultados ML
    if 'ml' in results:
        report.append("\n" + "-" * 80)
        report.append("1. MODELOS DE MACHINE LEARNING")
        report.append("-" * 80)
        ml_results = results['ml']['results']
        for _, row in ml_results.iterrows():
            report.append(f"\n{row['Modelo']}:")
            report.append(f"  - RMSE: {row['RMSE']:.4f}")
            report.append(f"  - R²: {row['R²']:.4f}")
    
    # Resultados GARCH
    if 'garch' in results and results['garch']:
        report.append("\n" + "-" * 80)
        report.append("2. PREDICCIÓN DE VOLATILIDAD GARCH")
        report.append("-" * 80)
        forecast = results['garch']['forecast']
        report.append(f"\nVolatilidad predicha (5 días): {forecast.forecast_value*100:.2f}%")
        report.append(f"Intervalo 95%: [{forecast.confidence_interval[0]*100:.2f}%, {forecast.confidence_interval[1]*100:.2f}%]")
    
    # Resultados Multi-Agente
    if 'multi_agent' in results:
        report.append("\n" + "-" * 80)
        report.append("3. SISTEMA MULTI-AGENTE")
        report.append("-" * 80)
        ma_result = results['multi_agent']['result']
        report.append(f"\nDecisión final: {ma_result.recommendation}")
        report.append(f"Confianza: {ma_result.confidence:.2%}")
        
        # Análisis individuales
        if 'individual_analyses' in ma_result.data:
            report.append(f"\nAnálisis individuales:")
            for agent_name, analysis in ma_result.data['individual_analyses'].items():
                report.append(f"  - {agent_name}: {analysis['recommendation']} (conf: {analysis['confidence']:.2f})")
    
    # Resultados Backtest
    if 'backtest' in results:
        report.append("\n" + "-" * 80)
        report.append("4. BACKTEST ECONÓMICO RIGUROSO")
        report.append("-" * 80)
        bt = results['backtest']['backtest_result']
        report.append(f"\nRetorno Total: {bt.total_return*100:.2f}%")
        report.append(f"Retorno Anualizado: {bt.annualized_return*100:.2f}%")
        report.append(f"Sharpe Ratio: {bt.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {bt.sortino_ratio:.2f}")
        report.append(f"Calmar Ratio: {bt.calmar_ratio:.2f}")
        report.append(f"Max Drawdown: {bt.max_drawdown*100:.2f}%")
        report.append(f"Ulcer Index: {bt.ulcer_index:.4f}")
        report.append(f"Win Rate: {bt.win_rate*100:.1f}%")
        report.append(f"Profit Factor: {bt.profit_factor:.2f}")
        
        # Robustez
        if 'robustness_score' in results['backtest']:
            rs = results['backtest']['robustness_score']
            report.append(f"\nScore de Robustez: {rs['overall']:.2f}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Guardar reporte
    report_path = os.path.join(os.path.dirname(__file__), 'results', f'REPORTE_FINAL_{symbol}.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"\n✅ Reporte guardado: {report_path}")


def main():
    """
    Función principal del sistema.
    """
    logger.info("=" * 80)
    logger.info("SISTEMA HÍBRIDO ML + LLM PARA PREDICCIÓN FINANCIERA")
    logger.info("Versión 2.0 - Datos Reales, LLM Gratuito, GARCH")
    logger.info("=" * 80)
    
    # Configuración
    symbols = ['AAPL', 'MSFT']  # Acciones
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Verificar si Ollama está disponible
    use_llm = True
    llm_provider = "ollama"
    
    logger.info(f"\nConfiguración:")
    logger.info(f"  Período: {start_date} a {end_date}")
    logger.info(f"  Símbolos: {symbols}")
    logger.info(f"  LLM: {llm_provider if use_llm else 'No (fallback a reglas)'}")
    
    all_results = {}
    
    for symbol in symbols:
        logger.info(f"\n{'#'*80}")
        logger.info(f"PROCESANDO: {symbol}")
        logger.info(f"{'#'*80}")
        
        results = {}
        
        # 1. Cargar y preparar datos
        df = load_and_prepare_data(symbol, start_date, end_date)
        
        if df.empty:
            logger.warning(f"Saltando {symbol} por falta de datos")
            continue
        
        # 2. Entrenar modelo GARCH para volatilidad
        garch_result = train_volatility_model(df, symbol)
        results['garch'] = garch_result
        
        # 3. Entrenar modelos ML
        ml_results = train_ml_models(df, symbol)
        results['ml'] = ml_results
        
        # 4. Ejecutar análisis multi-agente
        ml_pred = {'predicted_return': 0.01, 'confidence': 0.6}  # Placeholder
        multi_agent_results = run_multi_agent_analysis(
            df, symbol, ml_pred, 
            use_llm=use_llm, 
            llm_provider=llm_provider
        )
        results['multi_agent'] = multi_agent_results
        
        # 5. Backtesting económico riguroso
        backtest_results = run_backtest_rigorous(df, symbol)
        results['backtest'] = backtest_results
        
        # 6. Generar reporte
        generate_final_report(symbol, results)
        
        all_results[symbol] = results
    
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"\n✅ Resultados guardados en: {os.path.join(os.path.dirname(__file__), 'results')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
