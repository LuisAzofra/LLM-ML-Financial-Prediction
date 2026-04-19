"""
Agente de Análisis Técnico
Analiza indicadores técnicos y patrones chartistas
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent, AnalysisResult
from models.pattern_detection.chart_patterns import PatternDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalAnalystAgent(BaseAgent):
    """
    Agente especializado en análisis técnico
    Analiza indicadores técnicos, patrones chartistas y tendencias de precios
    """
    
    def __init__(self):
        super().__init__(
            name="TechnicalAnalyst",
            role="Análisis Técnico",
            description="Experto en análisis técnico de mercados financieros. "
                       "Especializado en indicadores técnicos, patrones chartistas "
                       "y análisis de tendencias de precios."
        )
        self.pattern_detector = PatternDetector()
    
    def process_message(self, message):
        """Procesa mensajes recibidos"""
        if message.message_type == 'PRICE_DATA':
            self.log("Recibidos datos de precios para análisis")
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Realiza análisis técnico completo
        
        Args:
            data: Diccionario con datos de precios y otros indicadores
        """
        df = data.get('price_data')
        symbol = data.get('symbol', 'Unknown')
        
        if df is None or df.empty:
            return AnalysisResult(
                agent_name=self.name,
                analysis_type='technical',
                confidence=0.0,
                recommendation='HOLD',
                reasoning='No hay datos disponibles para análisis',
                data={}
            )
        
        self.log(f"Analizando {symbol}...")
        
        # Análisis de tendencia
        trend_analysis = self._analyze_trend(df)
        
        # Análisis de indicadores
        indicator_analysis = self._analyze_indicators(df)
        
        # Detección de patrones
        patterns = self.pattern_detector.detect_all_patterns(df)
        
        # Análisis de soporte y resistencia
        support_resistance = self._find_support_resistance(df)
        
        # Análisis de volumen
        volume_analysis = self._analyze_volume(df)
        
        # Consolidar análisis
        recommendation = self._generate_recommendation(
            trend_analysis, indicator_analysis, patterns, volume_analysis
        )
        
        confidence = self._calculate_confidence(
            trend_analysis, indicator_analysis, patterns
        )
        
        reasoning = self._generate_reasoning(
            trend_analysis, indicator_analysis, patterns, support_resistance
        )
        
        result = AnalysisResult(
            agent_name=self.name,
            analysis_type='technical',
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            data={
                'trend': trend_analysis,
                'indicators': indicator_analysis,
                'patterns': [{'type': p.pattern_type.value, 'confidence': p.confidence} for p in patterns[:3]],
                'support_resistance': support_resistance,
                'volume': volume_analysis
            }
        )
        
        self.add_to_memory({
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': confidence
        })
        
        return result
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analiza la tendencia actual"""
        close = df['Close']
        
        # Medias móviles
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Tendencia a corto plazo
        if current_price > sma_20 > sma_50:
            trend = 'ALCISTA_FUERTE'
            strength = 1.0
        elif current_price > sma_20:
            trend = 'ALCISTA'
            strength = 0.7
        elif current_price < sma_20 < sma_50:
            trend = 'BAJISTA_FUERTE'
            strength = 1.0
        elif current_price < sma_20:
            trend = 'BAJISTA'
            strength = 0.7
        else:
            trend = 'LATERAL'
            strength = 0.5
        
        # Pendiente de la tendencia
        price_change_20d = (current_price - close.iloc[-20]) / close.iloc[-20] * 100
        
        return {
            'direction': trend,
            'strength': strength,
            'price_change_20d': price_change_20d,
            'above_sma20': current_price > sma_20,
            'above_sma50': current_price > sma_50
        }
    
    def _analyze_indicators(self, df: pd.DataFrame) -> Dict:
        """Analiza indicadores técnicos"""
        indicators = {}
        
        # RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            indicators['RSI'] = {
                'value': rsi,
                'signal': 'SOBRECOMPRA' if rsi > 70 else 'SOBREVENTA' if rsi < 30 else 'NEUTRO',
                'interpretation': self._interpret_rsi(rsi)
            }
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            macd_hist = df['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in df.columns else macd - macd_signal
            
            indicators['MACD'] = {
                'value': macd,
                'signal_value': macd_signal,
                'histogram': macd_hist,
                'signal': 'COMPRA' if macd > macd_signal and macd_hist > 0 else 
                         'VENTA' if macd < macd_signal and macd_hist < 0 else 'NEUTRO'
            }
        
        # Bandas de Bollinger
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_position = df['BB_Percent'].iloc[-1] if 'BB_Percent' in df.columns else None
            indicators['Bollinger'] = {
                'position': bb_position,
                'signal': 'SOBRECOMPRA' if bb_position and bb_position > 0.8 else
                         'SOBREVENTA' if bb_position and bb_position < 0.2 else 'NEUTRO'
            }
        
        # ATR (volatilidad)
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            indicators['ATR'] = {
                'value': atr,
                'percent_of_price': atr / current_price * 100
            }
        
        return indicators
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpreta el valor del RSI"""
        if rsi > 70:
            return f"RSI en {rsi:.1f}: Condición de sobrecompra. Posible corrección a la baja."
        elif rsi > 50:
            return f"RSI en {rsi:.1f}: Momentum alcista moderado."
        elif rsi > 30:
            return f"RSI en {rsi:.1f}: Momentum bajista moderado."
        else:
            return f"RSI en {rsi:.1f}: Condición de sobreventa. Posible rebote al alza."
    
    def _find_support_resistance(self, df: pd.DataFrame, n_levels: int = 3) -> Dict:
        """Encuentra niveles de soporte y resistencia"""
        highs = df['High'].values
        lows = df['Low'].values
        
        # Encontrar pivots (simplificado)
        from scipy.signal import find_peaks
        
        # Resistencias (picos)
        resistance_peaks, _ = find_peaks(highs, distance=10, prominence=highs.std()*0.5)
        resistances = sorted(highs[resistance_peaks], reverse=True)[:n_levels]
        
        # Soportes (valles)
        support_peaks, _ = find_peaks(-lows, distance=10, prominence=lows.std()*0.5)
        supports = sorted(lows[support_peaks])[:n_levels]
        
        current_price = df['Close'].iloc[-1]
        
        return {
            'resistances': resistances,
            'supports': supports,
            'nearest_resistance': min([r for r in resistances if r > current_price], default=None),
            'nearest_support': max([s for s in supports if s < current_price], default=None),
            'current_price': current_price
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analiza el volumen de trading"""
        if 'Volume' not in df.columns:
            return {}
        
        volume = df['Volume']
        volume_sma = volume.rolling(20).mean()
        
        current_volume = volume.iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        volume_trend = 'CRECIENTE' if current_volume > avg_volume * 1.2 else \
                      'DECRECIENTE' if current_volume < avg_volume * 0.8 else 'NORMAL'
        
        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume,
            'trend': volume_trend
        }
    
    def _generate_recommendation(self, trend: Dict, indicators: Dict, 
                                patterns: List, volume: Dict) -> str:
        """Genera recomendación basada en el análisis"""
        score = 0
        
        # Puntuación de tendencia
        if trend['direction'] == 'ALCISTA_FUERTE':
            score += 2
        elif trend['direction'] == 'ALCISTA':
            score += 1
        elif trend['direction'] == 'BAJISTA_FUERTE':
            score -= 2
        elif trend['direction'] == 'BAJISTA':
            score -= 1
        
        # Puntuación de indicadores
        if 'RSI' in indicators:
            rsi_signal = indicators['RSI']['signal']
            if rsi_signal == 'SOBREVENTA':
                score += 1
            elif rsi_signal == 'SOBRECOMPRA':
                score -= 1
        
        if 'MACD' in indicators:
            macd_signal = indicators['MACD']['signal']
            if macd_signal == 'COMPRA':
                score += 1
            elif macd_signal == 'VENTA':
                score -= 1
        
        # Puntuación de patrones
        for pattern in patterns[:2]:
            if pattern.pattern_type.value in ['doble_suelo', 'cuña_descendente']:
                score += pattern.confidence
            elif pattern.pattern_type.value in ['doble_techo', 'cuña_ascendente']:
                score -= pattern.confidence
        
        # Generar recomendación
        if score >= 2:
            return 'COMPRA_FUERTE'
        elif score >= 1:
            return 'COMPRA'
        elif score <= -2:
            return 'VENTA_FUERTE'
        elif score <= -1:
            return 'VENTA'
        else:
            return 'MANTENER'
    
    def _calculate_confidence(self, trend: Dict, indicators: Dict, 
                             patterns: List) -> float:
        """Calcula nivel de confianza del análisis"""
        confidence = 0.5
        
        # Confianza basada en tendencia clara
        if trend['direction'] in ['ALCISTA_FUERTE', 'BAJISTA_FUERTE']:
            confidence += 0.2
        
        # Confianza basada en concordancia de indicadores
        indicator_signals = []
        if 'RSI' in indicators:
            indicator_signals.append(indicators['RSI']['signal'])
        if 'MACD' in indicators:
            indicator_signals.append(indicators['MACD']['signal'])
        
        if len(indicator_signals) >= 2:
            if all(s in ['SOBRECOMPRA', 'COMPRA'] for s in indicator_signals) or \
               all(s in ['SOBREVENTA', 'VENTA'] for s in indicator_signals):
                confidence += 0.15
        
        # Confianza basada en patrones detectados
        if patterns:
            avg_pattern_conf = np.mean([p.confidence for p in patterns[:3]])
            confidence += avg_pattern_conf * 0.15
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, trend: Dict, indicators: Dict, 
                           patterns: List, support_resistance: Dict) -> str:
        """Genera explicación del análisis"""
        reasoning_parts = []
        
        # Tendencia
        reasoning_parts.append(f"Tendencia: {trend['direction'].replace('_', ' ')} "
                              f"(cambio 20d: {trend['price_change_20d']:.2f}%)")
        
        # Indicadores
        if 'RSI' in indicators:
            reasoning_parts.append(indicators['RSI']['interpretation'])
        
        if 'MACD' in indicators:
            macd = indicators['MACD']
            reasoning_parts.append(f"MACD: {macd['signal']} (histograma: {macd['histogram']:.4f})")
        
        # Patrones
        if patterns:
            pattern_names = [p.pattern_type.value for p in patterns[:2]]
            reasoning_parts.append(f"Patrones detectados: {', '.join(pattern_names)}")
        
        # Soporte/Resistencia
        if support_resistance.get('nearest_resistance'):
            reasoning_parts.append(f"Resistencia cercana: ${support_resistance['nearest_resistance']:.2f}")
        if support_resistance.get('nearest_support'):
            reasoning_parts.append(f"Soporte cercano: ${support_resistance['nearest_support']:.2f}")
        
        return " | ".join(reasoning_parts)
