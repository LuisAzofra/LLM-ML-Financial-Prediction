"""
Cliente LLM gratuito para el sistema multi-agente
Soporta Ollama (local) y Hugging Face Inference API (gratuita)
"""
import requests
import json
import logging
from typing import Optional, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Cliente unificado para LLMs gratuitos.
    Prioriza Ollama local, con fallback a Hugging Face.
    """
    
    def __init__(self, provider: str = "ollama", model: Optional[str] = None):
        """
        Inicializa el cliente LLM.
        
        Args:
            provider: 'ollama' o 'huggingface'
            model: Nombre del modelo (None para usar default)
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self.session = requests.Session()
        self.last_response_time = 0
        
        # Configurar según proveedor
        if self.provider == "ollama":
            self.base_url = "http://localhost:11434"
            self._check_ollama_available()
        elif self.provider == "huggingface":
            self.api_token = None  # Se puede usar sin token con rate limits
            self.base_url = "https://api-inference.huggingface.co/models"
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")
    
    def _get_default_model(self) -> str:
        """Retorna el modelo por defecto según el proveedor"""
        defaults = {
            "ollama": "qwen2.5",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
        }
        return defaults.get(self.provider, "qwen2.5")
    
    def _check_ollama_available(self) -> bool:
        """Verifica si Ollama está corriendo y ajusta el modelo al primero disponible."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"✅ Ollama disponible. Modelos: {model_names}")

                # If requested model isn't available, fall back to first available
                preferred = ['qwen2.5', 'qwen2', 'llama3.1', 'llama3.2', 'llama3', 'mistral', 'phi3', 'gemma2']
                if self.model not in model_names and model_names:
                    for pref in preferred:
                        matching = [m for m in model_names if m.startswith(pref)]
                        if matching:
                            self.model = matching[0]
                            logger.info(f"✅ Usando modelo alternativo: {self.model}")
                            break
                    else:
                        self.model = model_names[0]
                        logger.info(f"✅ Usando primer modelo disponible: {self.model}")
                return True
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  Ollama no está corriendo. Inicia con: ollama serve")
            logger.warning("   O usa Hugging Face como fallback.")
        except Exception as e:
            logger.warning(f"⚠️  Error conectando a Ollama: {e}")
        return False

    def _extract_json_with_retry(self, text: str, required_keys: list,
                                  system_prompt: str = '', max_attempts: int = 3) -> Optional[dict]:
        """
        Intenta extraer un JSON válido del texto del LLM con hasta max_attempts intentos.
        En el último intento, re-prompta al modelo con instrucciones estrictas.
        """
        import re

        def _try_parse(raw: str) -> Optional[dict]:
            # Attempt 1: find first {...} block
            m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
            # Attempt 2: strip text before first { and after last }
            start = raw.find('{')
            end   = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except Exception:
                    pass
            return None

        parsed = _try_parse(text)
        if parsed:
            return parsed

        # Final attempt: re-prompt with strict instruction
        if self.provider == 'ollama' and max_attempts > 1:
            strict_prompt = (
                "Your previous response could not be parsed as JSON. "
                "Respond ONLY with a valid JSON object — no explanation, no markdown, "
                "no code blocks. Just the raw JSON object.\n\n"
                f"Required keys: {required_keys}\n\nOriginal response:\n{text}"
            )
            retry_result = self._generate_ollama(strict_prompt, system_prompt,
                                                  temperature=0.1, max_tokens=512)
            parsed = _try_parse(retry_result.get('text', ''))
            if parsed:
                return parsed

        return None
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Genera una respuesta del LLM.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Instrucciones de sistema (opcional)
            temperature: Creatividad (0.0 - 1.0)
            max_tokens: Máximo de tokens a generar
            
        Returns:
            Dict con 'text', 'tokens_used', 'response_time'
        """
        start_time = time.time()
        
        if self.provider == "ollama":
            result = self._generate_ollama(prompt, system_prompt, temperature, max_tokens)
        else:
            result = self._generate_huggingface(prompt, system_prompt, temperature, max_tokens)
        
        result['response_time'] = time.time() - start_time
        self.last_response_time = result['response_time']
        
        return result
    
    def _generate_ollama(self, prompt: str, system_prompt: Optional[str],
                         temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Genera usando Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            return {
                'text': data.get('response', '').strip(),
                'tokens_used': data.get('eval_count', 0),
                'model': self.model,
                'provider': 'ollama'
            }
        except requests.exceptions.ConnectionError:
            logger.error("❌ No se pudo conectar a Ollama. ¿Está corriendo?")
            return {
                'text': "[ERROR: Ollama no disponible]",
                'tokens_used': 0,
                'error': 'connection_failed'
            }
        except Exception as e:
            logger.error(f"❌ Error en Ollama: {e}")
            return {
                'text': f"[ERROR: {str(e)}]",
                'tokens_used': 0,
                'error': str(e)
            }
    
    def _generate_huggingface(self, prompt: str, system_prompt: Optional[str],
                              temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Genera usando Hugging Face Inference API (gratuita)"""
        url = f"{self.base_url}/{self.model}"
        
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        # Construir payload según el modelo
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        try:
            response = self.session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 429:
                logger.warning("⚠️  Rate limit de HuggingFace alcanzado. Esperando...")
                time.sleep(20)
                return self._generate_huggingface(prompt, system_prompt, temperature, max_tokens)
            
            response.raise_for_status()
            data = response.json()
            
            # HuggingFace puede devolver lista o dict
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get('generated_text', '')
            else:
                text = data.get('generated_text', '')
            
            return {
                'text': text.strip(),
                'tokens_used': len(text.split()),  # Estimación
                'model': self.model,
                'provider': 'huggingface'
            }
        except Exception as e:
            logger.error(f"❌ Error en HuggingFace: {e}")
            return {
                'text': f"[ERROR: {str(e)}]",
                'tokens_used': 0,
                'error': str(e)
            }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Análisis de sentimiento especializado para textos financieros.
        Usa un prompt específico para extraer sentimiento cuantificado.
        """
        system_prompt = (
            "You are a senior financial market analyst specializing in sentiment analysis "
            "for investment decisions. You have deep expertise in reading market psychology "
            "from news flow.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY valid JSON — no markdown, no explanation outside the JSON.\n"
            "2. The 'score' MUST be a float between -1.0 (extremely bearish) and 1.0 (extremely bullish).\n"
            "3. Be conservative: only assign extreme scores (|score| > 0.7) for clearly decisive news.\n"
            "4. Distinguish between short-term sentiment (next 1-5 days) and structural factors.\n"
            "5. Base your analysis strictly on the provided text — do not assume facts not stated.\n\n"
            'Required JSON format: {"sentiment":"bullish|bearish|neutral","score":<float>,'
            '"confidence":<float>,"key_points":["..."],"reasoning":"<max 60 words>",'
            '"time_sensitivity":"immediate|short_term|medium_term"}'
        )

        prompt = (
            f"Analyze the following financial text and extract the investment sentiment:\n\n"
            f"TEXT: '{text}'\n\n"
            f"Respond with ONLY the JSON object:"
        )

        required_keys = ['sentiment', 'score', 'confidence']
        result = self.generate(prompt, system_prompt, temperature=0.2, max_tokens=512)

        parsed = self._extract_json_with_retry(
            result.get('text', ''), required_keys, system_prompt
        )
        result['parsed'] = parsed if parsed else self._fallback_sentiment_parsing(result.get('text', ''))
        return result
    
    def _fallback_sentiment_parsing(self, text: str) -> Dict:
        """Parseo fallback cuando el JSON no es válido"""
        text_lower = text.lower()
        
        # Detección simple basada en palabras clave
        bullish_words = ['bullish', 'positive', 'optimistic', 'growth', 'buy', 'alza', 'alcista']
        bearish_words = ['bearish', 'negative', 'pessimistic', 'decline', 'sell', 'baja', 'bajista']
        
        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)
        
        if bullish_count > bearish_count:
            sentiment = "bullish"
            score = min(0.3 + 0.1 * bullish_count, 1.0)
        elif bearish_count > bullish_count:
            sentiment = "bearish"
            score = max(-0.3 - 0.1 * bearish_count, -1.0)
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': 0.5,
            'key_points': [],
            'reasoning': 'Análisis basado en palabras clave (fallback)'
        }
    
    def interpret_market_data(self,
                              technical_summary: str,
                              sentiment_summary: str,
                              risk_summary: str,
                              ml_summary: str = '') -> Dict[str, Any]:
        """
        Interpreta datos de mercado como un analista senior.
        Usado por el PortfolioManagerAgent.
        Ahora acepta ml_summary con métricas cuantitativas del ensemble de ML.
        """
        system_prompt = (
            "You are a Senior Investment Fund Manager with 20 years of experience managing "
            "multi-asset portfolios across equities and crypto.\n\n"
            "Your task is to synthesize technical analysis, sentiment, risk assessment, and "
            "quantitative ML model predictions into a single investment decision.\n\n"
            "CRITICAL DECISION RULES (follow in strict order of priority):\n"
            "1. VOLATILITY OVERRIDE: If vol_regime > 1.5 (high volatility) OR vol_percentile > 0.8, "
            "cap confidence at 0.5 and use small/none position sizes. Markets in extreme vol regimes "
            "invalidate short-term ML signals.\n"
            "2. BEAR MARKET RULE: If bull_market=0 (price below 200-day SMA), do NOT recommend BUY "
            "unless there is very strong positive evidence from ALL other signals.\n"
            "3. MODEL AGREEMENT: If ML models disagree on direction (shown in ML section), "
            "default to HOLD regardless of individual signals.\n"
            "4. RISK VETO: If risk assessment recommends RECHAZAR, override any bullish signal "
            "with HOLD.\n"
            "5. SENTIMENT CONTRARIAN: Extreme fear (score < -0.7) can be a buy signal. "
            "Extreme greed (score > 0.7) can be a sell signal.\n\n"
            "CRITICAL: Output ONLY valid JSON — no markdown, no preamble.\n"
            'Required JSON: {"recommendation":"BUY|SELL|HOLD|WAIT","confidence":<0-1>,'
            '"position_size":"small|medium|large|none","reasoning":"<max 150 words>",'
            '"risk_level":"low|medium|high|extreme","time_horizon":"short|medium|long",'
            '"key_risks":["<risk1>","<risk2>"]}'
        )

        ml_section = f"\n=== ML MODEL PREDICTIONS ===\n{ml_summary}\n" if ml_summary else ""

        prompt = (
            f"As a Senior Fund Manager, synthesize the following analyses and make a REALISTIC, "
            f"risk-aware investment decision. Be conservative — it is better to miss an opportunity "
            f"than to take on unquantified risk.\n\n"
            f"=== TECHNICAL ANALYSIS ===\n{technical_summary}\n\n"
            f"=== SENTIMENT ANALYSIS ===\n{sentiment_summary}\n\n"
            f"=== RISK ASSESSMENT ===\n{risk_summary}"
            f"{ml_section}\n\n"
            f"Apply the decision rules from your system prompt strictly. "
            f"What is your investment recommendation? Respond with ONLY the JSON object:"
        )

        required_keys = ['recommendation', 'confidence']
        result = self.generate(prompt, system_prompt, temperature=0.3, max_tokens=1024)

        parsed = self._extract_json_with_retry(
            result.get('text', ''), required_keys, system_prompt
        )
        result['parsed'] = parsed if parsed else self._fallback_interpretation(result.get('text', ''))
        return result
    
    def _fallback_interpretation(self, text: str) -> Dict:
        """Interpretación fallback"""
        text_lower = text.lower()
        
        if 'buy' in text_lower or 'compra' in text_lower:
            recommendation = "BUY"
        elif 'sell' in text_lower or 'venta' in text_lower:
            recommendation = "SELL"
        elif 'hold' in text_lower or 'mantener' in text_lower:
            recommendation = "HOLD"
        else:
            recommendation = "WAIT"
        
        return {
            'recommendation': recommendation,
            'confidence': 0.5,
            'position_size': 'small',
            'reasoning': 'Interpretación basada en análisis textual (fallback)',
            'risk_level': 'medium',
            'time_horizon': 'medium'
        }


# Cliente de fallback simple cuando no hay LLM disponible
class SimpleLLMClient:
    """Cliente simple que simula respuestas cuando no hay LLM disponible"""
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            'text': '[LLM no disponible - usando análisis basado en reglas]',
            'tokens_used': 0,
            'provider': 'none'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        # Análisis simple basado en palabras clave
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        try:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound > 0.2:
                sentiment = "bullish"
            elif compound < -0.2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                'text': json.dumps({
                    'sentiment': sentiment,
                    'score': compound,
                    'confidence': abs(compound)
                }),
                'parsed': {
                    'sentiment': sentiment,
                    'score': compound,
                    'confidence': abs(compound),
                    'key_points': [],
                    'reasoning': 'Análisis VADER (sin LLM)'
                }
            }
        except:
            return {
                'text': '{"sentiment": "neutral", "score": 0, "confidence": 0}',
                'parsed': {
                    'sentiment': 'neutral',
                    'score': 0,
                    'confidence': 0
                }
            }
    
    def interpret_market_data(self, **kwargs) -> Dict[str, Any]:
        return {
            'text': '{"recommendation": "HOLD", "confidence": 0.5}',
            'parsed': {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'position_size': 'none',
                'reasoning': 'Análisis basado en reglas (sin LLM)',
                'risk_level': 'medium',
                'time_horizon': 'medium'
            }
        }
