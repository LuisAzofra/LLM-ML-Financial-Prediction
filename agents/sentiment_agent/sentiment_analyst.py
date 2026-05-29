"""
Agente de Análisis de Sentimiento REAL
Analiza noticias reales usando LLM gratuito (Ollama/HuggingFace)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent, AnalysisResult
from data.news_fetcher import NewsFetcher, get_news_for_symbol
from utils.llm_client import LLMClient, SimpleLLMClient


class SentimentAnalystAgent(BaseAgent):
    """
    Agente especializado en análisis de sentimiento REAL.
    Obtiene noticias financieras reales y las analiza con LLM gratuito.
    """
    
    def __init__(self, use_llm: bool = True, llm_provider: str = "ollama"):
        super().__init__(
            name="SentimentAnalyst",
            role="Análisis de Sentimiento",
            description="Experto en análisis de sentimiento de mercado. "
                       "Obtiene noticias financieras reales de RSS feeds "
                       "y las analiza usando LLM para evaluar el mood del mercado."
        )
        
        self.use_llm = use_llm
        self.news_fetcher = NewsFetcher()
        
        # Inicializar cliente LLM
        if use_llm:
            try:
                self.llm_client = LLMClient(provider=llm_provider)
                logger.info(f"✅ SentimentAnalyst usando LLM: {llm_provider}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo inicializar LLM: {e}")
                logger.warning("   Usando análisis basado en reglas")
                self.llm_client = SimpleLLMClient()
        else:
            self.llm_client = SimpleLLMClient()
    
    def process_message(self, message):
        """Procesa mensajes recibidos"""
        if message.message_type == 'NEWS_DATA':
            self.log("Recibidos datos de noticias para análisis")
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Realiza análisis de sentimiento REAL.
        
        Args:
            data: Diccionario con symbol, news (opcional), etc.
        """
        symbol = data.get('symbol', 'Unknown')
        is_crypto = data.get('is_crypto', False)
        
        self.log(f"Analizando sentimiento para {symbol}...")
        
        # 1. OBTENER NOTICIAS REALES
        provided_news = data.get('news', [])
        
        if provided_news and len(provided_news) > 0:
            # Usar noticias proporcionadas
            news_data = provided_news
            self.log(f"Usando {len(news_data)} noticias proporcionadas")
        else:
            # Obtener noticias reales
            self.log("📰 Obteniendo noticias reales...")
            news_result = self.news_fetcher.get_cached_news(
                symbol=symbol,
                is_crypto=is_crypto,
                max_items=15
            )
            news_data = news_result.get('news', [])
            self.log(f"✅ {len(news_data)} noticias obtenidas")
        
        if not news_data:
            logger.warning(f"⚠️  No se encontraron noticias para {symbol}")
            return AnalysisResult(
                agent_name=self.name,
                analysis_type='sentiment',
                confidence=0.0,
                recommendation='HOLD',
                reasoning='No hay noticias disponibles para análisis',
                data={'sentiment_score': 0, 'news_count': 0}
            )
        
        # 2. ANÁLISIS DE SENTIMIENTO CON LLM
        news_sentiment = self._analyze_news_with_llm(news_data)
        
        # 3. OBTENER INDICADORES DE SENTIMIENTO DE MERCADO
        market_sentiment = self._get_market_sentiment_indicators(symbol, is_crypto)
        
        # 4. AGREGAR SENTIMIENTO
        aggregated_sentiment = self._aggregate_sentiment(
            news_sentiment, market_sentiment
        )
        
        # 5. GENERAR RECOMENDACIÓN
        recommendation = self._generate_recommendation(aggregated_sentiment)
        
        # 6. GENERAR RAZONAMIENTO DETALLADO
        reasoning = self._generate_reasoning(
            news_data, news_sentiment, market_sentiment, aggregated_sentiment
        )
        
        result = AnalysisResult(
            agent_name=self.name,
            analysis_type='sentiment',
            confidence=aggregated_sentiment['confidence'],
            recommendation=recommendation,
            reasoning=reasoning,
            data={
                'news_count': len(news_data),
                'news_analyzed': [n['title'] for n in news_data[:5]],
                'news_sentiment': news_sentiment,
                'market_sentiment': market_sentiment,
                'aggregated': aggregated_sentiment
            }
        )
        
        self.add_to_memory({
            'symbol': symbol,
            'sentiment': aggregated_sentiment['score'],
            'recommendation': recommendation,
            'news_count': len(news_data)
        })
        
        return result
    
    def _analyze_news_with_llm(self, news_items: List[Dict]) -> Dict:
        """
        Analiza noticias usando el LLM.
        Combina múltiples noticias en un análisis agregado.
        """
        if not news_items:
            return {'score': 0, 'confidence': 0, 'sentiment': 'neutral'}
        
        # Preparar texto combinado de noticias
        combined_text = "\n\n".join([
            f"Noticia {i+1}: {item['title']}\n{item.get('summary', '')[:200]}"
            for i, item in enumerate(news_items[:10])  # Analizar top 10
        ])
        
        # Casting numérico seguro: el LLM puede devolver strings o tipos raros
        def _to_float(x, default):
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        # Analizar con LLM
        try:
            llm_result = self.llm_client.analyze_sentiment(combined_text)
            parsed = llm_result.get('parsed', {})

            # Convertir sentimiento a score numérico
            sentiment_map = {
                'bullish': 1.0,
                'positive': 0.7,
                'neutral': 0.0,
                'negative': -0.7,
                'bearish': -1.0
            }

            sentiment_str = parsed.get('sentiment', 'neutral').lower()
            # No fiarse del formato del LLM: castear y acotar a rangos válidos
            # (un modelo pequeño puede devolver score=5, "0.5" o confidence=1.5)
            score = _to_float(parsed.get('score', sentiment_map.get(sentiment_str, 0)), 0.0)
            score = float(np.clip(score, -1.0, 1.0))
            confidence = _to_float(parsed.get('confidence', 0.5), 0.5)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            return {
                'score': score,
                'confidence': confidence,
                'sentiment': sentiment_str,
                'key_points': parsed.get('key_points', []),
                'reasoning': parsed.get('reasoning', ''),
                'llm_raw_response': llm_result.get('text', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis LLM: {e}")
            # Fallback a análisis simple
            return self._fallback_sentiment_analysis(news_items)
    
    def _fallback_sentiment_analysis(self, news_items: List[Dict]) -> Dict:
        """Análisis de sentimiento fallback usando palabras clave"""
        positive_words = ['bullish', 'growth', 'profit', 'gain', 'surge', 'rally', 
                         'strong', 'positive', 'upbeat', 'optimistic', 'breakthrough',
                         'alcista', 'crecimiento', 'ganancia', 'aumento', 'fuerte']
        
        negative_words = ['bearish', 'loss', 'decline', 'crash', 'drop', 'fall',
                         'weak', 'negative', 'pessimistic', 'concern', 'risk', 'fear',
                         'bajista', 'pérdida', 'caída', 'debilidad', 'recesión']
        
        scores = []
        for item in news_items:
            text = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
            
            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)
            
            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                score = 0
            
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        
        return {
            'score': avg_score,
            'confidence': 0.5,
            'sentiment': 'bullish' if avg_score > 0.2 else 'bearish' if avg_score < -0.2 else 'neutral',
            'positive_count': sum(1 for s in scores if s > 0),
            'negative_count': sum(1 for s in scores if s < 0),
            'key_points': [],
            'reasoning': 'Análisis basado en palabras clave (fallback)'
        }
    
    def _get_market_sentiment_indicators(self, symbol: str, is_crypto: bool) -> Dict:
        """Obtiene indicadores de sentimiento de mercado"""
        indicators = {}
        
        # Fear & Greed Index (para cripto o mercado general)
        try:
            sentiment_data = self.news_fetcher.fetch_market_sentiment_indicators()
            if 'fear_greed_index' in sentiment_data:
                fgi = sentiment_data['fear_greed_index']
                indicators['fear_greed'] = {
                    'value': fgi['value'],
                    'classification': fgi['classification'],
                    'signal': self._interpret_fear_greed(fgi['value'])
                }
        except Exception as e:
            logger.warning(f"⚠️  Error obteniendo Fear & Greed: {e}")
        
        return indicators
    
    def _interpret_fear_greed(self, value: int) -> str:
        """Interpreta el índice Fear & Greed"""
        if value < 25:
            return 'EXTREME_FEAR'
        elif value < 45:
            return 'FEAR'
        elif value < 55:
            return 'NEUTRAL'
        elif value < 75:
            return 'GREED'
        else:
            return 'EXTREME_GREED'
    
    def _aggregate_sentiment(self, news: Dict, market: Dict) -> Dict:
        """Agrega diferentes fuentes de sentimiento"""
        # Pesos para cada fuente
        weights = {
            'news': 0.7,
            'market': 0.3
        }
        
        # Acumulamos suma ponderada y el peso de las fuentes realmente presentes.
        # Normalizar por la suma de pesos usados (no por el total fijo) evita
        # atenuar el resultado cuando falta alguna fuente (p. ej. solo noticias).
        weighted_score_sum = 0.0
        weighted_confidence_sum = 0.0
        weight_used = 0.0
        any_source = False

        # Noticias
        if news.get('confidence', 0) > 0:
            weighted_score_sum += news['score'] * weights['news']
            weighted_confidence_sum += news['confidence'] * weights['news']
            weight_used += weights['news']
            any_source = True

        # Indicadores de mercado
        if 'fear_greed' in market:
            fgi = market['fear_greed']
            # Invertir: miedo extremo = oportunidad de compra (sentimiento positivo)
            fgi_signal = fgi.get('signal', 'NEUTRAL')
            if fgi_signal == 'EXTREME_FEAR':
                market_score = 0.8  # Contrarian
            elif fgi_signal == 'FEAR':
                market_score = 0.3
            elif fgi_signal == 'GREED':
                market_score = -0.3
            elif fgi_signal == 'EXTREME_GREED':
                market_score = -0.8
            else:
                market_score = 0

            weighted_score_sum += market_score * weights['market']
            weighted_confidence_sum += 0.6 * weights['market']
            weight_used += weights['market']
            any_source = True

        if not any_source or weight_used <= 0:
            return {'score': 0, 'confidence': 0, 'interpretation': 'NEUTRAL'}

        aggregated_score = weighted_score_sum / weight_used
        aggregated_confidence = weighted_confidence_sum / weight_used
        
        return {
            'score': aggregated_score,
            'confidence': min(aggregated_confidence, 1.0),
            'interpretation': self._interpret_sentiment_score(aggregated_score)
        }
    
    def _interpret_sentiment_score(self, score: float) -> str:
        """Interpreta el score de sentimiento"""
        if score >= 0.5:
            return 'MUY_POSITIVO'
        elif score >= 0.2:
            return 'POSITIVO'
        elif score > -0.2:
            return 'NEUTRAL'
        elif score > -0.5:
            return 'NEGATIVO'
        else:
            return 'MUY_NEGATIVO'
    
    def _generate_recommendation(self, sentiment: Dict) -> str:
        """Genera recomendación basada en sentimiento"""
        score = sentiment['score']
        
        if score >= 0.5:
            return 'COMPRA_FUERTE'
        elif score >= 0.2:
            return 'COMPRA'
        elif score > -0.2:
            return 'MANTENER'
        elif score > -0.5:
            return 'VENTA'
        else:
            return 'VENTA_FUERTE'
    
    def _generate_reasoning(self, news: List[Dict], news_sentiment: Dict,
                           market_sentiment: Dict, aggregated: Dict) -> str:
        """Genera explicación detallada del análisis"""
        parts = []
        
        # Resumen de noticias
        parts.append(f"Analizadas {len(news)} noticias")
        
        # Sentimiento de noticias
        if news_sentiment.get('sentiment'):
            parts.append(f"Sentimiento: {news_sentiment['sentiment'].upper()}")
            parts.append(f"Score: {news_sentiment['score']:.2f}")
        
        # Indicadores de mercado
        if 'fear_greed' in market_sentiment:
            fgi = market_sentiment['fear_greed']
            parts.append(f"Fear & Greed: {fgi['value']} ({fgi['classification']})")
        
        # Puntos clave
        if news_sentiment.get('key_points'):
            parts.append(f"Puntos clave: {', '.join(news_sentiment['key_points'][:2])}")
        
        # Resultado agregado
        parts.append(f"Score final: {aggregated['score']:.2f} ({aggregated['interpretation']})")
        
        return " | ".join(parts)
