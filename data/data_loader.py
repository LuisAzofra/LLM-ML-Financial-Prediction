"""
Módulo para cargar datos financieros y de criptoactivos
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import logging
import time
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """Cargador de datos financieros tradicionales"""
    
    def __init__(self):
        self.cache = {}
    
    def download_stock_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = '1d') -> pd.DataFrame:
        """
        Descarga datos históricos de acciones
        
        Args:
            symbol: Símbolo de la acción (ej: 'AAPL', 'MSFT')
            start_date: Fecha inicio 'YYYY-MM-DD'
            end_date: Fecha fin 'YYYY-MM-DD'
            interval: Intervalo ('1d', '1h', '1wk', etc.)
        """
        try:
            logger.info(f"Descargando datos de {symbol}...")

            import urllib.request
            import urllib.error
            import json
            import time

            # Map intervals for Yahoo API v8
            interval_map = {
                '1d': '1d', '1wk': '1wk', '1mo': '1mo',
                '1h': '1h', '1m': '1m'
            }
            api_interval = interval_map.get(interval, '1d')

            # Use period1/period2 Unix timestamps — this is the ONLY way to guarantee
            # that Yahoo returns data strictly ending at end_date (no leakage for backtesting)
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            p1 = int(start_dt.timestamp())
            p2 = int(end_dt.timestamp())

            # URL to Yahoo API v8 with explicit date range
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval={api_interval}"

            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'application/json'
            })

            logger.info(f"Usando Yahoo API v8 para {symbol} ({start_date} → {end_date})...")

            # Retry con backoff exponencial para errores 429 (rate limiting)
            data = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=15) as response:
                        if response.status == 200:
                            data = json.loads(response.read().decode('utf-8'))
                            break
                except urllib.error.HTTPError as http_err:
                    if http_err.code == 429:
                        wait_sec = (2 ** attempt) * 2  # 2, 4, 8, 16 s
                        logger.warning(f"Rate limited (429) para {symbol}, esperando {wait_sec}s (intento {attempt+1}/4)")
                        time.sleep(wait_sec)
                    else:
                        raise  # otros errores HTTP no se reintentan
                except Exception:
                    raise

            if data is None:
                logger.warning(f"No se encontraron datos para {symbol} tras reintentos")
                return pd.DataFrame()

            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]

            # Convert arrays to pandas DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quote.get('open', []),
                'High': quote.get('high', []),
                'Low': quote.get('low', []),
                'Close': quote.get('close', []),
                'Volume': quote.get('volume', [])
            })

            # Índice tz-naive para evitar "Invalid comparison between
            # datetime64[s, UTC] and Timestamp" en todo el pipeline
            df.set_index('Date', inplace=True)

            # Add Adj Close (same as Close for basic charts)
            df['Adj_Close'] = df['Close']
            df['Symbol'] = symbol

            # Clean up
            df.dropna(subset=['Close'], inplace=True)
            # Solo ffill: el bfill rellenaría huecos con valores FUTUROS (look-ahead
            # bias) y contaminaría el backtest. Se elimina deliberadamente.
            df.ffill(inplace=True)
            # ffill no cubre NaN al inicio de columnas != Close; se descartan esas
            # filas en vez de rellenarlas con datos del futuro.
            ohlc_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
            df.dropna(subset=ohlc_cols, inplace=True)

            logger.info(f"Descargados {len(df)} registros de {symbol} (Yahoo JSON API)")
            return df

        except Exception as e:
            logger.error(f"Error descargando {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def download_multiple_stocks(self, symbols: List[str], start_date: str, 
                                 end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Descarga datos de múltiples acciones
        """
        data = {}
        for symbol in symbols:
            df = self.download_stock_data(symbol, start_date, end_date, interval)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.5)  # Evitar rate limiting
        return data
    
    def download_data(self, symbol: str, start_date: str, end_date: str,
                      asset_type: str = 'stock', interval: str = '1d') -> pd.DataFrame:
        """
        Unified data download for stocks and crypto via yfinance.
        
        For crypto, use tickers like 'BTC-USD', 'ETH-USD'.
        For stocks, use tickers like 'AAPL', 'MSFT'.
        """
        # Normalize crypto symbols: if user passes 'BTC' or 'bitcoin', convert to 'BTC-USD'
        if asset_type == 'crypto':
            crypto_map = {
                'BTC': 'BTC-USD', 'BITCOIN': 'BTC-USD',
                'ETH': 'ETH-USD', 'ETHEREUM': 'ETH-USD',
                'BNB': 'BNB-USD', 'SOL': 'SOL-USD', 'SOLANA': 'SOL-USD',
                'XRP': 'XRP-USD', 'ADA': 'ADA-USD', 'CARDANO': 'ADA-USD',
                'DOGE': 'DOGE-USD', 'DOGECOIN': 'DOGE-USD',
                'DOT': 'DOT-USD', 'POLKADOT': 'DOT-USD',
                'AVAX': 'AVAX-USD', 'MATIC': 'MATIC-USD', 'POLYGON': 'MATIC-USD',
                'LINK': 'LINK-USD', 'UNI': 'UNI-USD',
                'LTC': 'LTC-USD', 'LITECOIN': 'LTC-USD',
                'SHIB': 'SHIB-USD',
            }
            upper = symbol.upper().replace('/USDT', '').replace('/USD', '')
            symbol = crypto_map.get(upper, symbol if '-' in symbol else f'{upper}-USD')
        
        return self.download_stock_data(symbol, start_date, end_date, interval)
    
    def get_sp500_list(self) -> List[str]:
        """
        Obtiene lista de empresas del S&P 500
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500 = tables[0]
            return sp500['Symbol'].tolist()
        except Exception as e:
            logger.error(f"Error obteniendo lista S&P 500: {str(e)}")
            # Lista de respaldo con principales empresas
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                   'JPM', 'V', 'WMT', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'BAC',
                   'ABBV', 'PFE', 'KO', 'PEP', 'COST', 'TMO', 'AVGO', 'DIS',
                   'CSCO', 'VZ', 'ADBE', 'CRM', 'ACN', 'WFC', 'MRK', 'NKE']
    
    def get_market_indices(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Descarga principales índices de mercado
        """
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        data = {}
        for name, symbol in indices.items():
            df = self.download_stock_data(symbol, start_date, end_date)
            if not df.empty:
                data[name] = df
        return data


class CryptoDataLoader:
    """Cargador de datos de criptoactivos"""
    
    def __init__(self):
        if not CCXT_AVAILABLE:
            logger.warning("⚠️  ccxt no disponible. CryptoDataLoader no funcionará.")
            self.exchange = None
            return
        self.exchange = ccxt.binance({'enableRateLimit': True})
    
    def download_crypto_data(self, symbol: str, start_date: str, 
                            end_date: str, timeframe: str = '1d') -> pd.DataFrame:
        """
        Descarga datos históricos de criptomonedas
        
        Args:
            symbol: Par de trading (ej: 'BTC/USDT', 'ETH/USDT')
            start_date: Fecha inicio 'YYYY-MM-DD'
            end_date: Fecha fin 'YYYY-MM-DD'
            timeframe: Intervalo ('1d', '4h', '1h', etc.)
        """
        try:
            logger.info(f"Descargando datos de {symbol}...")
            
            # Convertir fechas a timestamps
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Descargar datos
            ohlcv = []
            while since < int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000):
                data = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not data:
                    break
                ohlcv.extend(data)
                since = data[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            
            if not ohlcv:
                logger.warning(f"No se encontraron datos para {symbol}")
                return pd.DataFrame()
            
            # Crear DataFrame
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df['Symbol'] = symbol
            
            logger.info(f"Descargados {len(df)} registros de {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def download_multiple_crypto(self, symbols: List[str], start_date: str, 
                                 end_date: str, timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Descarga datos de múltiples criptomonedas
        """
        data = {}
        for symbol in symbols:
            df = self.download_crypto_data(symbol, start_date, end_date, timeframe)
            if not df.empty:
                data[symbol] = df
        return data
    
    def get_top_cryptos(self, limit: int = 20) -> List[str]:
        """
        Obtiene lista de principales criptomonedas por capitalización
        """
        try:
            markets = self.exchange.load_markets()
            # Filtrar pares USDT
            usdt_pairs = [s for s in markets.keys() if s.endswith('/USDT')]
            # Tomar los más comunes
            common_cryptos = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
                            'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'DOT/USDT',
                            'MATIC/USDT', 'SHIB/USDT', 'AVAX/USDT', 'TRX/USDT',
                            'UNI/USDT', 'LINK/USDT', 'ETC/USDT', 'LTC/USDT']
            return common_cryptos[:limit]
        except Exception as e:
            logger.error(f"Error obteniendo lista de criptos: {str(e)}")
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']


class DataManager:
    """Gestor unificado de datos financieros"""
    
    def __init__(self):
        self.stock_loader = FinancialDataLoader()
        self.crypto_loader = CryptoDataLoader()
        self.processor = None  # Se importará después
    
    def load_all_data(self, stock_symbols: List[str], crypto_symbols: List[str],
                     start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Carga datos de acciones y criptomonedas
        """
        logger.info("=" * 50)
        logger.info("CARGA DE DATOS FINANCIEROS")
        logger.info("=" * 50)
        
        # Cargar acciones
        logger.info("\n📈 Cargando datos de acciones...")
        stocks = self.stock_loader.download_multiple_stocks(
            stock_symbols, start_date, end_date
        )
        
        # Cargar criptomonedas
        logger.info("\n₿ Cargando datos de criptomonedas...")
        cryptos = self.crypto_loader.download_multiple_crypto(
            crypto_symbols, start_date, end_date
        )
        
        logger.info(f"\n✅ Datos cargados: {len(stocks)} acciones, {len(cryptos)} criptomonedas")
        
        return {
            'stocks': stocks,
            'cryptos': cryptos
        }


# Funciones de utilidad
def get_common_stocks() -> List[str]:
    """Retorna lista de acciones comunes para análisis"""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'JPM', 'V', 'WMT', 'JNJ', 'UNH', 'HD', 'PG', 'MA']


def get_common_cryptos() -> List[str]:
    """Retorna lista de criptomonedas comunes para análisis"""
    return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT']


def save_data_to_csv(data: Dict[str, pd.DataFrame], output_dir: str):
    """Guarda los datos en archivos CSV"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in data.items():
        safe_name = name.replace('/', '_')
        filepath = os.path.join(output_dir, f"{safe_name}.csv")
        df.to_csv(filepath)
        logger.info(f"Guardado: {filepath}")
