"""
Cliente LLM gratuito para el sistema multi-agente
Soporta Ollama (local) y Hugging Face Inference API (gratuita)
"""
import requests
import json
import logging
import os
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
        elif self.provider == "local":
            # Inferencia local con transformers — sin Ollama ni HF remoto.
            # Carga perezosa: sólo importa torch/transformers al primer generate.
            self._local_pipeline = None
            self._local_tokenizer = None
        elif self.provider == "local-gguf":
            # Tier 3.4: inferencia local con llama-cpp-python sobre Qwen2.5-1.5B
            # cuantizado a 4-bit GGUF (~1 GB en disco/RAM). Mejor instruction-following
            # que Qwen2.5-0.5B bfloat16 con footprint similar y sin OOM en mac arm64.
            # Modelo y fichero configurables vía TFG_LOCAL_GGUF_FILE.
            self._gguf_llm = None
            self._gguf_repo = self.model
            self._gguf_filename = os.environ.get(
                'TFG_LOCAL_GGUF_FILE',
                'qwen2.5-1.5b-instruct-q4_k_m.gguf',
            )
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")
    
    def _get_default_model(self) -> str:
        """Retorna el modelo por defecto según el proveedor.

        El modelo local es configurable por env var TFG_LOCAL_LLM. Para
        máquinas con swap muy saturado se recomienda SmolLM2-360M
        (~700MB en disco, ~350MB en bfloat16). Qwen2.5-0.5B (default) requiere
        ~600MB en bfloat16. Cambiar a SmolLM2-135M (~120MB bfloat16) si la RAM
        es aún más ajustada.
        """
        defaults = {
            "ollama": "qwen2.5",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
            "local": os.environ.get('TFG_LOCAL_LLM', "Qwen/Qwen2.5-0.5B-Instruct"),
            "local-gguf": os.environ.get(
                'TFG_LOCAL_GGUF_LLM', "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
            ),
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

        def _try_parse(raw: str) -> Optional[dict]:
            if not raw:
                return None
            # (a) intento directo
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # (b) primer objeto {...} BALANCEADO por conteo de profundidad de llaves.
            # Estrategia copiada de agents/debate.py::_safe_parse_json para no
            # capturar objetos anidados (como hacía el regex \{[^{}]*\}).
            start = raw.find('{')
            while start >= 0:
                depth = 0
                for i in range(start, len(raw)):
                    c = raw[i]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            blob = raw[start:i + 1]
                            try:
                                obj = json.loads(blob)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
                            break
                start = raw.find('{', start + 1)
            return None

        def _is_valid(obj) -> bool:
            # Sólo aceptar dicts que contengan TODAS las required_keys.
            if not isinstance(obj, dict):
                return False
            if required_keys:
                return all(k in obj for k in required_keys)
            return True

        parsed = _try_parse(text)
        if _is_valid(parsed):
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
            if _is_valid(parsed):
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
        elif self.provider == "local":
            result = self._generate_local(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "local-gguf":
            result = self._generate_local_gguf(prompt, system_prompt, temperature, max_tokens)
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
    
    # ── Cache a nivel de módulo ──────────────────────────────────────────
    # Compartido entre TODAS las instancias de LLMClient para que varios
    # agentes (SentimentAnalyst, PortfolioManager) no carguen el mismo
    # modelo varias veces en memoria — crítico en CPU con RAM ajustada.
    _LOCAL_PIPELINE_CACHE: Dict[str, Any] = {}
    _GGUF_PIPELINE_CACHE: Dict[str, Any] = {}

    def _init_local_pipeline(self):
        """Carga perezosa del modelo local (transformers, CPU) con cache global."""
        if self._local_pipeline is not None:
            return
        cached = LLMClient._LOCAL_PIPELINE_CACHE.get(self.model)
        if cached is not None:
            self._local_tokenizer, self._local_pipeline = cached
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"⏳ Cargando modelo local '{self.model}' (primera vez descarga ~1GB)…")
        tok = AutoTokenizer.from_pretrained(self.model)
        # bfloat16 en CPU: ~500MB para Qwen-0.5B (vs ~1GB en fp32) — evita OOM
        # en macOS cuando hay presión de swap. bfloat16 está soportado en torch≥1.12
        # y la pérdida de calidad para JSON structured output es despreciable.
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        self._local_tokenizer = tok
        self._local_pipeline = model
        LLMClient._LOCAL_PIPELINE_CACHE[self.model] = (tok, model)
        logger.info(f"✅ Modelo local cargado: {self.model} (bfloat16, cache global)")

    def _generate_local(self, prompt: str, system_prompt: Optional[str],
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Inferencia local usando transformers (CPU). No requiere Ollama ni API."""
        try:
            self._init_local_pipeline()
            import torch
            tok = self._local_tokenizer
            model = self._local_pipeline

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Qwen y la mayoría de chat models exponen un chat template.
            if hasattr(tok, 'apply_chat_template') and tok.chat_template:
                inputs_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                inputs_text = (
                    (f"[SYSTEM]\n{system_prompt}\n\n" if system_prompt else "")
                    + f"[USER]\n{prompt}\n\n[ASSISTANT]\n"
                )

            inputs = tok(inputs_text, return_tensors='pt', truncation=True, max_length=3072)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 512),
                    do_sample=(temperature > 0.01),
                    temperature=max(temperature, 0.01),
                    top_p=0.9,
                    pad_token_id=tok.eos_token_id,
                    repetition_penalty=1.1,
                )
            gen_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
            text = tok.decode(gen_tokens, skip_special_tokens=True).strip()

            return {
                'text': text,
                'tokens_used': int(gen_tokens.shape[0]),
                'model': self.model,
                'provider': 'local',
            }
        except Exception as e:
            logger.error(f"❌ Error en LLM local: {e}")
            return {
                'text': f"[ERROR: {e}]",
                'tokens_used': 0,
                'error': str(e),
            }

    # ── Tier 3.4: provider local-gguf (Qwen2.5-1.5B 4-bit GGUF) ───────────
    def _init_local_gguf_pipeline(self):
        """Carga perezosa del modelo GGUF (llama.cpp) con cache global."""
        if self._gguf_llm is not None:
            return
        cache_key = f"{self._gguf_repo}:{self._gguf_filename}"
        cached = LLMClient._GGUF_PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            self._gguf_llm = cached
            return
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        logger.info(
            f"⏳ Descargando/cargando GGUF '{self._gguf_repo}/{self._gguf_filename}'…"
        )
        gguf_path = hf_hub_download(
            repo_id=self._gguf_repo, filename=self._gguf_filename
        )
        # n_ctx=4096 cubre prompts del bull/bear/judge (Tier 2.3) sin truncar.
        # n_threads=os.cpu_count() en macOS arm64 ≈ 8-10. n_gpu_layers=-1 usa
        # Metal cuando llama-cpp-python se compiló con CMAKE_ARGS=-DLLAMA_METAL=on;
        # si no, llama.cpp ignora silenciosamente y corre todo en CPU.
        llm = Llama(
            model_path=gguf_path,
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            n_gpu_layers=-1,
            verbose=False,
            seed=42,
        )
        self._gguf_llm = llm
        LLMClient._GGUF_PIPELINE_CACHE[cache_key] = llm
        logger.info(
            f"✅ Modelo GGUF cargado: {self._gguf_repo}/{self._gguf_filename}"
        )

    def _generate_local_gguf(self, prompt: str, system_prompt: Optional[str],
                             temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Inferencia local con llama-cpp-python (GGUF 4-bit). No requiere torch."""
        try:
            self._init_local_gguf_pipeline()
            llm = self._gguf_llm

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # create_chat_completion aplica el chat template del modelo (Qwen
            # ChatML) automáticamente, igual que apply_chat_template de
            # transformers — no hace falta formatear el prompt a mano.
            out = llm.create_chat_completion(
                messages=messages,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                max_tokens=min(max_tokens, 512),
                repeat_penalty=1.1,
            )
            choice = (out.get('choices') or [{}])[0]
            text = (choice.get('message') or {}).get('content', '').strip()
            usage = out.get('usage') or {}

            return {
                'text': text,
                'tokens_used': int(usage.get('completion_tokens', 0)),
                'model': self.model,
                'provider': 'local-gguf',
            }
        except Exception as e:
            logger.error(f"❌ Error en LLM local-gguf: {e}")
            return {
                'text': f"[ERROR: {e}]",
                'tokens_used': 0,
                'error': str(e),
            }

    def _generate_huggingface(self, prompt: str, system_prompt: Optional[str],
                              temperature: float, max_tokens: int,
                              attempt: int = 0) -> Dict[str, Any]:
        """Genera usando Hugging Face Inference API (gratuita)"""
        max_retries = 3
        url = f"{self.base_url}/{self.model}"

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        # Construir payload según el modelo
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
                # Tope de reintentos para evitar recursión infinita si HF mantiene el 429
                if attempt >= max_retries:
                    logger.error("❌ Rate limit de HuggingFace persistente tras "
                                 f"{max_retries} reintentos. Abortando.")
                    return {
                        'text': '[ERROR: HuggingFace rate limit (429) persistente]',
                        'tokens_used': 0,
                        'error': '429 rate limit'
                    }
                logger.warning("⚠️  Rate limit de HuggingFace alcanzado. Esperando...")
                time.sleep(20)
                return self._generate_huggingface(prompt, system_prompt, temperature,
                                                  max_tokens, attempt=attempt + 1)
            
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
            "5. Base your analysis strictly on the provided text — do not assume facts not stated.\n"
            "6. 'confidence' must reflect your actual certainty: if the news is generic, aged, or "
            "irrelevant to the 5-day horizon, confidence MUST be ≤ 0.35. Do NOT inflate confidence "
            "just because you produced a score. A well-calibrated 0.25 is better than a fake 0.70.\n"
            "7. If the news mix is mostly noise (ads, generic market recaps, unrelated filings), "
            "set sentiment='neutral' and score=0.0 with confidence ≤ 0.20.\n\n"
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
            "1. ML IS THE PRIMARY SIGNAL. When the ML ensemble has confidence ≥ 0.60 AND >70% of "
            "individual models agree on direction, FOLLOW the ML recommendation — do not default "
            "to HOLD just because sentiment is mixed. ML has a quantitative edge you do not.\n"
            "2. VOLATILITY OVERRIDE: If volatility is extreme (> 60% annualized) OR vol_percentile > 0.9, "
            "downgrade position_size to 'small' but do NOT veto a high-confidence ML signal.\n"
            "3. BEAR MARKET NUANCE: Below 200-day SMA, a BUY is still valid when ML confidence ≥ 0.65 AND "
            "oversold conditions (RSI < 30 or extreme fear sentiment) align. Otherwise prefer HOLD.\n"
            "4. MODEL DISAGREEMENT: Only default to HOLD when models are truly split (≤55% agreement). "
            "If 70%+ agree and the ML confidence ≥ 0.55, trust the ML vote.\n"
            "5. RISK VETO: If risk assessment says RECHAZAR, you may still allow a SMALL position when "
            "the ML confidence is ≥ 0.70 and direction matches the broader trend (SMA 200).\n"
            "6. SENTIMENT CONTRARIAN: Extreme fear (score < -0.7) is a BUY filter when ML also bullish. "
            "Extreme greed (score > 0.7) is a SELL/trim filter when ML bearish.\n"
            "7. AVOID HOLD BIAS: If you recommend HOLD, provide a SPECIFIC reason — 'uncertainty' alone "
            "is not enough. A well-calibrated manager takes calculated bets when the ML edge is real.\n\n"
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
        except Exception as e:
            logger.error(f"❌ Error en análisis de sentimiento VADER: {e}")
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
