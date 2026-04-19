"""
Módulo para obtener noticias financieras reales
Usa RSS feeds y APIs gratuitas
"""
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Obtiene noticias financieras de fuentes gratuitas.
    Usa RSS feeds y web scraping ético.
    """
    
    # RSS Feeds financieros gratuitos
    RSS_FEEDS = {
        'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
        'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories',
        'seeking_alpha': 'https://seekingalpha.com/feed.xml',
        'investing_com': 'https://www.investing.com/rss/news.rss',
        'financial_times': 'https://www.ft.com/?format=rss',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'reuters': 'https://www.reutersagency.com/feed/?taxonomy=markets&post_type=reuters-best',
    }
    
    # APIs gratuitas con rate limits razonables
    NEWS_APIS = {
        'newsdata': 'https://newsdata.io/api/1/news',  # 200 requests/día gratis
        'gnews': 'https://gnews.io/api/v4/search',      # 100 requests/día gratis
    }
    
    def __init__(self, cache_duration: int = 300):
        """
        Inicializa el fetcher de noticias.
        
        Args:
            cache_duration: Duración del caché en segundos (default 5 min)
        """
        self.cache = {}
        self.cache_duration = cache_duration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_rss_news(self, 
                       sources: Optional[List[str]] = None,
                       symbol: Optional[str] = None,
                       max_items: int = 20) -> List[Dict]:
        """
        Obtiene noticias de feeds RSS.
        
        Args:
            sources: Lista de fuentes RSS (None para usar todas)
            symbol: Símbolo para filtrar (ej: 'AAPL')
            max_items: Máximo de noticias por fuente
            
        Returns:
            Lista de noticias normalizadas
        """
        all_news = []
        sources = sources or list(self.RSS_FEEDS.keys())
        
        for source in sources:
            try:
                feed_url = self.RSS_FEEDS.get(source)
                if not feed_url:
                    continue
                
                logger.info(f"📰 Fetching RSS: {source}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_items]:
                    news_item = self._normalize_rss_entry(entry, source)
                    
                    # Filtrar por símbolo si se especifica
                    if symbol:
                        title_desc = f"{news_item['title']} {news_item.get('summary', '')}"
                        if symbol.upper() not in title_desc.upper():
                            continue
                    
                    all_news.append(news_item)
                
                # Respetar rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️  Error fetching {source}: {e}")
                continue
        
        # Ordenar por fecha
        all_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        logger.info(f"✅ Total noticias RSS: {len(all_news)}")
        return all_news
    
    def _normalize_rss_entry(self, entry, source: str) -> Dict:
        """Normaliza una entrada RSS al formato estándar"""
        # Parsear fecha
        published = entry.get('published', '')
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_dt = datetime(*entry.published_parsed[:6])
                published = published_dt.isoformat()
        except:
            published = datetime.now().isoformat()
        
        return {
            'title': entry.get('title', ''),
            'summary': entry.get('summary', entry.get('description', '')),
            'link': entry.get('link', ''),
            'published': published,
            'source': source,
            'author': entry.get('author', 'Unknown'),
            'type': 'rss'
        }
    
    def fetch_yahoo_finance_news(self, symbol: str, max_items: int = 10) -> List[Dict]:
        """
        Obtiene noticias específicas de Yahoo Finance para un símbolo.
        
        Args:
            symbol: Símbolo de la acción (ej: 'AAPL')
            max_items: Máximo de noticias
            
        Returns:
            Lista de noticias
        """
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            logger.info(f"📰 Fetching Yahoo Finance news for {symbol}")
            
            feed = feedparser.parse(url)
            news = []
            
            for entry in feed.entries[:max_items]:
                news.append(self._normalize_rss_entry(entry, 'yahoo_finance'))
            
            logger.info(f"✅ {len(news)} noticias de Yahoo Finance para {symbol}")
            return news
            
        except Exception as e:
            logger.warning(f"⚠️  Error fetching Yahoo Finance: {e}")
            return []
    
    def fetch_crypto_news(self, 
                          coin: Optional[str] = None,
                          max_items: int = 15) -> List[Dict]:
        """
        Obtiene noticias de criptomonedas.
        
        Args:
            coin: Nombre de la cripto (ej: 'bitcoin', 'ethereum')
            max_items: Máximo de noticias
            
        Returns:
            Lista de noticias
        """
        news = []
        
        # CoinDesk RSS
        try:
            coindesk_feed = 'https://www.coindesk.com/arc/outboundfeeds/rss/'
            feed = feedparser.parse(coindesk_feed)
            
            for entry in feed.entries[:max_items]:
                item = self._normalize_rss_entry(entry, 'coindesk')
                if coin and coin.lower() not in (item['title'] + item.get('summary', '')).lower():
                    continue
                news.append(item)
        except Exception as e:
            logger.warning(f"⚠️  Error fetching CoinDesk: {e}")
        
        # Cointelegraph RSS
        try:
            cointelegraph_feed = 'https://cointelegraph.com/rss'
            feed = feedparser.parse(cointelegraph_feed)
            
            for entry in feed.entries[:max_items]:
                item = self._normalize_rss_entry(entry, 'cointelegraph')
                if coin and coin.lower() not in (item['title'] + item.get('summary', '')).lower():
                    continue
                news.append(item)
        except Exception as e:
            logger.warning(f"⚠️  Error fetching Cointelegraph: {e}")
        
        news.sort(key=lambda x: x.get('published', ''), reverse=True)
        logger.info(f"✅ {len(news)} noticias crypto")
        return news
    
    def fetch_market_sentiment_indicators(self) -> Dict:
        """
        Obtiene indicadores de sentimiento de mercado de fuentes gratuitas.
        
        Returns:
            Dict con indicadores de sentimiento
        """
        indicators = {}
        
        # Fear & Greed Index (CNN) - scraping
        try:
            url = "https://edition.cnn.com/markets/fear-and-greed"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar el valor del índice
            # Nota: CNN puede cambiar su estructura HTML
            # Este es un ejemplo de cómo hacer scraping
            fear_greed_div = soup.find('div', class_='fear-greed-indicator')
            if fear_greed_div:
                value = fear_greed_div.get_text(strip=True)
                indicators['fear_greed_cnn'] = value
        except Exception as e:
            logger.warning(f"⚠️  Error fetching CNN Fear & Greed: {e}")
        
        # Alternative: usar API gratuita de Alternative.me
        try:
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url, timeout=10)
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                indicators['fear_greed_index'] = {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification'],
                    'timestamp': data['data'][0]['timestamp']
                }
        except Exception as e:
            logger.warning(f"⚠️  Error fetching Fear & Greed API: {e}")
        
        return indicators
    
    def fetch_all_news(self, 
                       symbol: Optional[str] = None,
                       is_crypto: bool = False,
                       max_items: int = 30) -> Dict[str, List[Dict]]:
        """
        Obtiene todas las noticias disponibles.
        
        Args:
            symbol: Símbolo de la acción/cripto
            is_crypto: Si es criptomoneda
            max_items: Máximo total de noticias
            
        Returns:
            Dict con noticias organizadas por fuente
        """
        result = {
            'symbol': symbol,
            'fetched_at': datetime.now().isoformat(),
            'news': [],
            'sentiment_indicators': {}
        }
        
        # Noticias específicas del símbolo
        if symbol:
            if is_crypto:
                result['news'] = self.fetch_crypto_news(coin=symbol.lower(), max_items=max_items)
            else:
                result['news'] = self.fetch_yahoo_finance_news(symbol, max_items=max_items)
        
        # Si no hay suficientes noticias específicas, agregar generales
        if len(result['news']) < max_items // 2:
            general_news = self.fetch_rss_news(max_items=max_items // 2)
            result['news'].extend(general_news)
        
        # Indicadores de sentimiento
        result['sentiment_indicators'] = self.fetch_market_sentiment_indicators()
        
        # Ordenar y limitar
        result['news'].sort(key=lambda x: x.get('published', ''), reverse=True)
        result['news'] = result['news'][:max_items]
        result['total_news'] = len(result['news'])
        
        return result
    
    def get_cached_news(self, 
                        symbol: Optional[str] = None,
                        is_crypto: bool = False,
                        max_items: int = 30) -> Dict:
        """
        Obtiene noticias con caché.
        """
        cache_key = f"{symbol}_{is_crypto}_{max_items}"
        
        # Verificar caché
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_duration):
                logger.info(f"📰 Usando noticias en caché para {symbol}")
                return cached_data
        
        # Fetch fresh data
        data = self.fetch_all_news(symbol, is_crypto, max_items)
        self.cache[cache_key] = (datetime.now(), data)
        
        return data


# Función de conveniencia para obtener noticias rápidamente
def get_news_for_symbol(symbol: str, is_crypto: bool = False, max_items: int = 20) -> List[Dict]:
    """
    Función rápida para obtener noticias de un símbolo.
    
    Args:
        symbol: Símbolo de la acción o cripto
        is_crypto: True si es criptomoneda
        max_items: Máximo de noticias
        
    Returns:
        Lista de noticias
    """
    fetcher = NewsFetcher()
    result = fetcher.get_cached_news(symbol, is_crypto, max_items)
    return result.get('news', [])


if __name__ == "__main__":
    # Test del módulo
    fetcher = NewsFetcher()
    
    # Test con AAPL
    print("=" * 60)
    print("Test: Noticias AAPL")
    print("=" * 60)
    news = fetcher.fetch_yahoo_finance_news('AAPL', max_items=5)
    for i, item in enumerate(news[:3], 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Fuente: {item['source']}")
        print(f"   Fecha: {item['published']}")
    
    # Test crypto
    print("\n" + "=" * 60)
    print("Test: Noticias Bitcoin")
    print("=" * 60)
    crypto_news = fetcher.fetch_crypto_news('bitcoin', max_items=5)
    for i, item in enumerate(crypto_news[:3], 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Fuente: {item['source']}")
