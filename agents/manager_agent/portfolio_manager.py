"""
Agente Gestor/Coordinador (Portfolio Manager) con LLM Intérprete REAL
Orquesta todos los agentes y toma decisiones finales usando LLM gratuito
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent, AnalysisResult, AgentCommunicationBus
from utils.llm_client import LLMClient, SimpleLLMClient


class PortfolioManagerAgent(BaseAgent):
    """
    Agente Gestor que coordina todos los agentes especializados.
    Usa un LLM gratuito como "intérprete" para tomar decisiones finales.
    """
    
    def __init__(self, communication_bus: AgentCommunicationBus = None, 
                 use_llm: bool = True, llm_provider: str = "ollama"):
        super().__init__(
            name="PortfolioManager",
            role="Gestor de Portfolio",
            description="Gestor senior de inversiones que coordina un equipo de analistas. "
                       "Toma decisiones finales basadas en análisis técnico, fundamental, "
                       "de sentimiento y de riesgo. Utiliza un LLM como intérprete para "
                       "sintetizar toda la información y tomar decisiones informadas."
        )
        
        self.communication_bus = communication_bus or AgentCommunicationBus()
        self.agent_analyses = {}
        self.decision_history = []
        self.portfolio = {
            'cash': 100000,  # Capital inicial
            'positions': {},
            'total_value': 100000
        }
        
        # Pesos base para cada tipo de análisis (pueden ser ajustados por el LLM)
        self.analysis_weights = {
            'technical': 0.30,
            'sentiment': 0.25,
            'risk': 0.30,
            'ml_prediction': 0.15
        }
        
        # Inicializar cliente LLM
        if use_llm:
            try:
                self.llm_client = LLMClient(provider=llm_provider)
                logger.info(f"✅ PortfolioManager usando LLM: {llm_provider}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo inicializar LLM: {e}")
                self.llm_client = SimpleLLMClient()
        else:
            self.llm_client = SimpleLLMClient()
    
    def register_agents(self, agents: List[BaseAgent]):
        """Registra agentes especializados"""
        for agent in agents:
            self.communication_bus.register_agent(agent)
            self.log(f"Agente registrado: {agent.name}")
    
    def process_message(self, message):
        """Procesa mensajes recibidos"""
        if message.message_type == 'ANALYSIS_COMPLETE':
            agent_name = message.sender
            self.agent_analyses[agent_name] = message.content
            self.log(f"Análisis recibido de {agent_name}")
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Coordina el análisis de todos los agentes y toma decisión final
        usando el LLM como intérprete.
        
        Args:
            data: Diccionario con todos los datos necesarios
        """
        symbol = data.get('symbol', 'Unknown')
        self.log(f"=" * 70)
        self.log(f"INICIANDO ANÁLISIS INTEGRADO PARA: {symbol}")
        self.log(f"=" * 70)
        
        # Recopilar análisis de todos los agentes
        analyses = {}
        
        # 1. Análisis Técnico
        technical_agent = self.communication_bus.get_agent('TechnicalAnalyst')
        if technical_agent:
            self.log("\n📊 Solicitando análisis técnico...")
            analyses['technical'] = technical_agent.analyze(data)
            self.log(f"   Recomendación: {analyses['technical'].recommendation} "
                    f"(confianza: {analyses['technical'].confidence:.2f})")
        
        # 2. Análisis de Sentimiento (con noticias reales)
        sentiment_agent = self.communication_bus.get_agent('SentimentAnalyst')
        if sentiment_agent:
            self.log("\n📰 Solicitando análisis de sentimiento...")
            analyses['sentiment'] = sentiment_agent.analyze(data)
            self.log(f"   Recomendación: {analyses['sentiment'].recommendation} "
                    f"(confianza: {analyses['sentiment'].confidence:.2f})")
        
        # 3. Análisis de Riesgo
        risk_agent = self.communication_bus.get_agent('RiskManager')
        if risk_agent:
            self.log("\n⚠️  Solicitando análisis de riesgo...")
            analyses['risk'] = risk_agent.analyze(data)
            self.log(f"   Recomendación: {analyses['risk'].recommendation} "
                    f"(confianza: {analyses['risk'].confidence:.2f})")
        
        # 4. Predicción ML (si está disponible)
        ml_prediction = data.get('ml_prediction')
        if ml_prediction:
            self.log("\n🤖 Incorporando predicción ML...")
            analyses['ml_prediction'] = self._create_ml_analysis(ml_prediction)
        
        # 5. USAR LLM COMO INTÉRPRETE
        self.log("\n🧠 Consultando al LLM como intérprete...")
        final_decision = self._llm_interpret_and_decide(symbol, analyses, data)
        
        # Actualizar portfolio si es necesario
        if final_decision['action'] in ['COMPRAR', 'VENDER']:
            self._execute_decision(symbol, final_decision)
        
        # Guardar en historial
        self.decision_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'decision': final_decision,
            'analyses': {k: v.recommendation for k, v in analyses.items()}
        })
        
        self.log(f"\n" + "=" * 70)
        self.log(f"DECISIÓN FINAL: {final_decision['action']}")
        self.log(f"Confianza: {final_decision['confidence']:.2f}")
        self.log(f"=" * 70)
        
        return AnalysisResult(
            agent_name=self.name,
            analysis_type='portfolio_management',
            confidence=final_decision['confidence'],
            recommendation=final_decision['action'],
            reasoning=final_decision['reasoning'],
            data={
                'individual_analyses': {k: {
                    'recommendation': v.recommendation,
                    'confidence': v.confidence,
                    'reasoning': v.reasoning
                } for k, v in analyses.items()},
                'portfolio': self.portfolio,
                'execution_details': final_decision.get('execution', {}),
                'llm_interpretation': final_decision.get('llm_interpretation', {})
            }
        )
    
    def _create_ml_analysis(self, ml_prediction: Dict) -> AnalysisResult:
        """Crea un objeto AnalysisResult desde predicción ML"""
        predicted_return = ml_prediction.get('predicted_return', 0)
        confidence = ml_prediction.get('confidence', 0.5)
        
        if predicted_return > 0.02:
            recommendation = 'COMPRA_FUERTE'
        elif predicted_return > 0.01:
            recommendation = 'COMPRA'
        elif predicted_return < -0.02:
            recommendation = 'VENTA_FUERTE'
        elif predicted_return < -0.01:
            recommendation = 'VENTA'
        else:
            recommendation = 'MANTENER'
        
        return AnalysisResult(
            agent_name='ML_Predictor',
            analysis_type='ml_prediction',
            confidence=confidence,
            recommendation=recommendation,
            reasoning=f"Predicción ML: retorno esperado {predicted_return*100:.2f}%",
            data=ml_prediction
        )
    
    def _llm_interpret_and_decide(self, symbol: str, 
                                   analyses: Dict[str, AnalysisResult],
                                   data: Dict) -> Dict:
        """
        Usa el LLM para interpretar todos los análisis y tomar una decisión.
        """
        # Preparar resúmenes para el LLM
        technical_summary = self._prepare_technical_summary(analyses.get('technical'))
        sentiment_summary = self._prepare_sentiment_summary(analyses.get('sentiment'))
        risk_summary = self._prepare_risk_summary(analyses.get('risk'))
        ml_summary = self._prepare_ml_summary(analyses.get('ml_prediction'), data)

        # Consultar al LLM
        try:
            llm_result = self.llm_client.interpret_market_data(
                technical_summary=technical_summary,
                sentiment_summary=sentiment_summary,
                risk_summary=risk_summary,
                ml_summary=ml_summary
            )
            
            parsed = llm_result.get('parsed', {})
            
            # Convertir recomendación del LLM al formato del sistema
            llm_rec = parsed.get('recommendation', 'HOLD').upper()
            action_map = {
                'BUY': 'COMPRAR',
                'SELL': 'VENDER',
                'HOLD': 'MANTENER',
                'WAIT': 'ESPERAR'
            }
            
            return {
                'action': action_map.get(llm_rec, 'MANTENER'),
                'confidence': parsed.get('confidence', 0.5),
                'position_size': parsed.get('position_size', 'none'),
                'reasoning': parsed.get('reasoning', 'Decisión basada en análisis LLM'),
                'risk_level': parsed.get('risk_level', 'medium'),
                'time_horizon': parsed.get('time_horizon', 'medium'),
                'llm_interpretation': parsed,
                'llm_raw_response': llm_result.get('text', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Error en interpretación LLM: {e}")
            # Fallback a decisión ponderada
            return self._fallback_decision(analyses)
    
    def _prepare_technical_summary(self, technical: Optional[AnalysisResult]) -> str:
        """Prepara resumen técnico para el LLM"""
        if not technical:
            return "No hay análisis técnico disponible."
        
        data = technical.data
        trend = data.get('trend', {})
        indicators = data.get('indicators', {})
        patterns = data.get('patterns', [])
        
        summary = f"""
Recomendación Técnica: {technical.recommendation} (confianza: {technical.confidence:.2f})
Tendencia: {trend.get('direction', 'N/A')}
Cambio 20d: {trend.get('price_change_20d', 0):.2f}%
RSI: {indicators.get('RSI', {}).get('value', 'N/A')}
MACD: {indicators.get('MACD', {}).get('signal', 'N/A')}
Patrones detectados: {', '.join([p['type'] for p in patterns[:3]]) if patterns else 'Ninguno'}
"""
        return summary.strip()
    
    def _prepare_sentiment_summary(self, sentiment: Optional[AnalysisResult]) -> str:
        """Prepara resumen de sentimiento para el LLM"""
        if not sentiment:
            return "No hay análisis de sentimiento disponible."
        
        data = sentiment.data
        aggregated = data.get('aggregated', {})
        news_sentiment = data.get('news_sentiment', {})
        
        summary = f"""
Recomendación Sentimiento: {sentiment.recommendation} (confianza: {sentiment.confidence:.2f})
Score de sentimiento: {aggregated.get('score', 0):.2f}
Interpretación: {aggregated.get('interpretation', 'N/A')}
Noticias analizadas: {data.get('news_count', 0)}
Sentimiento noticias: {news_sentiment.get('sentiment', 'N/A')}
"""
        return summary.strip()
    
    def _prepare_ml_summary(self, ml_analysis: Optional[AnalysisResult], data: Dict) -> str:
        """Prepara resumen cuantitativo de ML para el LLM"""
        # Try to get the full ml_result from the data dict (set by api.py)
        ml_full = data.get('ml_result', {})
        if not ml_full and ml_analysis:
            ml_full = ml_analysis.data

        if not ml_full:
            return ""

        pred_pct = round(ml_full.get('ensemble_prediction', 0) * 100, 3)
        direction = ml_full.get('ensemble_direction', 'N/A')
        confidence = round(ml_full.get('ensemble_confidence', 0) * 100, 1)
        horizon = ml_full.get('prediction_horizon_days', 5)
        stacking = "stacking meta-learner" if ml_full.get('stacking_used') else "weighted average"
        lstm_used = ml_full.get('lstm_available', False)

        models = ml_full.get('models', [])
        model_lines = []
        for m in models:
            model_lines.append(
                f"  - {m['name']}: RMSE={m['rmse']:.4f}, R²={m['r2']:.3f}"
            )

        top_feats = ml_full.get('top_features', [])
        feat_str = ", ".join([f['name'] for f in top_feats[:3]]) if top_feats else "N/A"

        # Individual model direction agreement (critical for confidence calibration)
        predictions = ml_full.get('predictions', {})
        if predictions:
            pred_vals = list(predictions.values())
            n_positive = sum(1 for p in pred_vals if p > 0.001)
            n_negative = sum(1 for p in pred_vals if p < -0.001)
            n_neutral  = len(pred_vals) - n_positive - n_negative
            agreement_str = f"Bullish:{n_positive} | Bearish:{n_negative} | Neutral:{n_neutral}"
        else:
            agreement_str = "N/A"

        # Regime features from top features (if present)
        regime_feats = [f['name'] for f in top_feats if any(
            kw in f['name'] for kw in ['vol_regime', 'vol_pct', 'bull_market', 'zscore', 'mom_']
        )]
        regime_str = ", ".join(regime_feats[:3]) if regime_feats else "None detected"

        lines = [
            f"Ensemble method: {stacking} {'+ LSTM' if lstm_used else ''}",
            f"Predicted {horizon}-day return: {pred_pct:+.3f}% → {direction}",
            f"Ensemble confidence: {confidence}% (calibrated — lower is more honest)",
            f"Model direction agreement: {agreement_str}",
            f"Top signal features: {feat_str}",
            f"Active regime features: {regime_str}",
            "Model cross-validation results (walk-forward):",
        ] + model_lines

        return "\n".join(lines)

    def _prepare_risk_summary(self, risk: Optional[AnalysisResult]) -> str:
        """Prepara resumen de riesgo para el LLM"""
        if not risk:
            return "No hay análisis de riesgo disponible."
        
        data = risk.data
        risk_metrics = data.get('risk_metrics', {})
        
        summary = f"""
Recomendación Riesgo: {risk.recommendation} (confianza: {risk.confidence:.2f})
VaR 95%: {risk_metrics.get('var_95', 0)*100:.2f}%
Volatilidad: {risk_metrics.get('volatility', 0)*100:.1f}%
Max Drawdown: {risk_metrics.get('max_drawdown', 0)*100:.1f}%
Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}
"""
        return summary.strip()
    
    def _fallback_decision(self, analyses: Dict[str, AnalysisResult]) -> Dict:
        """Decisión fallback cuando el LLM no está disponible"""
        # Mapear recomendaciones a scores
        recommendation_scores = {
            'COMPRA_FUERTE': 2, 'COMPRA': 1, 'APROBAR': 1,
            'APROBAR_CON_PRECAUCION': 0.5, 'MANTENER': 0, 'HOLD': 0,
            'NEUTRO': 0, 'VENTA': -1, 'RECHAZAR': -1, 'VENTA_FUERTE': -2
        }
        
        weighted_score = 0
        total_confidence = 0
        
        for analysis_type, analysis in analyses.items():
            weight = self.analysis_weights.get(analysis_type, 0.2)
            score = recommendation_scores.get(analysis.recommendation, 0)
            confidence = analysis.confidence
            
            weighted_score += score * weight * confidence
            total_confidence += weight * confidence
        
        # Normalizar score
        if total_confidence > 0:
            normalized_score = weighted_score / total_confidence
        else:
            normalized_score = 0
        
        # Determinar acción
        if normalized_score >= 1.5:
            action = 'COMPRAR'
            confidence = min(abs(normalized_score) / 2, 1.0)
        elif normalized_score >= 0.5:
            action = 'COMPRAR'
            confidence = min(abs(normalized_score), 1.0)
        elif normalized_score <= -1.5:
            action = 'VENDER'
            confidence = min(abs(normalized_score) / 2, 1.0)
        elif normalized_score <= -0.5:
            action = 'VENDER'
            confidence = min(abs(normalized_score), 1.0)
        else:
            action = 'MANTENER'
            confidence = 0.5
        
        # Verificar aprobación de riesgo
        if 'risk' in analyses:
            risk_rec = analyses['risk'].recommendation
            if risk_rec == 'RECHAZAR':
                action = 'MANTENER'
                confidence *= 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'score': normalized_score,
            'reasoning': 'Decisión ponderada (fallback sin LLM)',
            'risk_level': 'medium',
            'time_horizon': 'medium',
            'llm_interpretation': {}
        }
    
    def _execute_decision(self, symbol: str, decision: Dict):
        """Ejecuta la decisión en el portfolio"""
        action = decision['action']
        position_size = decision.get('position_size', 'none')
        
        # Calcular tamaño de posición basado en recomendación
        size_multipliers = {
            'small': 0.05,    # 5% del capital
            'medium': 0.10,   # 10% del capital
            'large': 0.20,    # 20% del capital
            'none': 0
        }
        
        if action == 'COMPRAR':
            multiplier = size_multipliers.get(position_size, 0.05)
            position_value = self.portfolio['cash'] * multiplier
            
            if position_value > 0:
                if symbol not in self.portfolio['positions']:
                    self.portfolio['positions'][symbol] = {
                        'shares': 0,
                        'avg_price': 0,
                        'value': 0
                    }
                
                self.portfolio['cash'] -= position_value
                self.log(f"💰 COMPRA ejecutada: ${position_value:,.2f} en {symbol} (tamaño: {position_size})")
        
        elif action == 'VENDER':
            if symbol in self.portfolio['positions']:
                position_value = self.portfolio['positions'][symbol]['value']
                self.portfolio['cash'] += position_value
                del self.portfolio['positions'][symbol]
                self.log(f"💰 VENTA ejecutada: ${position_value:,.2f} de {symbol}")
    
    def get_portfolio_summary(self) -> Dict:
        """Obtiene resumen del portfolio"""
        total_positions_value = sum(p['value'] for p in self.portfolio['positions'].values())
        self.portfolio['total_value'] = self.portfolio['cash'] + total_positions_value
        
        return {
            'cash': self.portfolio['cash'],
            'positions_value': total_positions_value,
            'total_value': self.portfolio['total_value'],
            'n_positions': len(self.portfolio['positions']),
            'positions': self.portfolio['positions']
        }
    
    def get_decision_history(self) -> pd.DataFrame:
        """Obtiene historial de decisiones"""
        if not self.decision_history:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'timestamp': d['timestamp'],
                'symbol': d['symbol'],
                'decision': d['decision']['action'],
                'confidence': d['decision']['confidence'],
                'score': d['decision'].get('score', 0)
            }
            for d in self.decision_history
        ])
