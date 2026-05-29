"""
Módulo de detección de patrones chartistas
Detecta patrones clásicos de análisis técnico en series temporales
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    DOUBLE_TOP = "doble_techo"
    DOUBLE_BOTTOM = "doble_suelo"
    HEAD_AND_SHOULDERS = "cabeza_hombros"
    INVERSE_HEAD_AND_SHOULDERS = "cabeza_hombros_invertido"
    ASCENDING_TRIANGLE = "triangulo_ascendente"
    DESCENDING_TRIANGLE = "triangulo_descendente"
    SYMMETRICAL_TRIANGLE = "triangulo_simetrico"
    RISING_WEDGE = "cuña_ascendente"
    FALLING_WEDGE = "cuña_descendente"
    FLAG = "bandera"
    PENNANT = "banderin"


@dataclass
class Pattern:
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""


class PatternDetector:
    """Detector de patrones chartistas"""
    
    def __init__(self, window_size: int = 30, min_touches: int = 2):
        self.window_size = window_size
        self.min_touches = min_touches
        self.patterns_detected = []
    
    def find_local_extrema(self, prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encuentra máximos y mínimos locales
        
        Args:
            prices: Array de precios
            order: Número de puntos a cada lado para considerar un extremo
        
        Returns:
            Tuple de (índices_máximos, índices_mínimos)
        """
        from scipy.signal import argrelextrema
        
        maxima = argrelextrema(prices, np.greater, order=order)[0]
        minima = argrelextrema(prices, np.less, order=order)[0]
        
        return maxima, minima
    
    def detect_double_top(self, df: pd.DataFrame, tolerance: float = 0.03) -> List[Pattern]:
        """
        Detecta patrón de doble techo
        
        Formación: Dos picos similares con un valle en medio
        Señal: Bajista
        """
        patterns = []
        prices = df['High'].values
        maxima, _ = self.find_local_extrema(prices)
        
        if len(maxima) < 2:
            return patterns
        
        for i in range(len(maxima) - 1):
            for j in range(i + 1, len(maxima)):
                peak1_idx = maxima[i]
                peak2_idx = maxima[j]
                
                # Verificar que hay un valle entre los picos
                valley_prices = prices[peak1_idx:peak2_idx]
                if len(valley_prices) == 0:
                    continue
                
                valley_idx = peak1_idx + np.argmin(valley_prices)
                valley_price = prices[valley_idx]
                
                peak1_price = prices[peak1_idx]
                peak2_price = prices[peak2_idx]
                
                # Verificar similitud de picos
                price_diff = abs(peak1_price - peak2_price) / peak1_price
                
                # Verificar que el valle es significativamente menor
                valley_depth = (min(peak1_price, peak2_price) - valley_price) / valley_price
                
                if price_diff <= tolerance and valley_depth > 0.02:
                    confidence = 1 - price_diff
                    
                    # Calcular objetivo de precio
                    pattern_height = min(peak1_price, peak2_price) - valley_price
                    price_target = valley_price - pattern_height
                    
                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        start_idx=peak1_idx,
                        end_idx=peak2_idx,
                        confidence=confidence,
                        price_target=price_target,
                        stop_loss=max(peak1_price, peak2_price) * 1.02,
                        description=f"Doble techo detectado. Pico 1: {peak1_price:.2f}, Pico 2: {peak2_price:.2f}, Valle: {valley_price:.2f}"
                    ))
        
        return patterns
    
    def detect_double_bottom(self, df: pd.DataFrame, tolerance: float = 0.03) -> List[Pattern]:
        """
        Detecta patrón de doble suelo
        
        Formación: Dos valles similares con un pico en medio
        Señal: Alcista
        """
        patterns = []
        prices = df['Low'].values
        _, minima = self.find_local_extrema(prices)
        
        if len(minima) < 2:
            return patterns
        
        for i in range(len(minima) - 1):
            for j in range(i + 1, len(minima)):
                bottom1_idx = minima[i]
                bottom2_idx = minima[j]
                
                # Verificar que hay un pico entre los valles
                peak_prices = prices[bottom1_idx:bottom2_idx]
                if len(peak_prices) == 0:
                    continue
                
                peak_idx = bottom1_idx + np.argmax(peak_prices)
                peak_price = prices[peak_idx]
                
                bottom1_price = prices[bottom1_idx]
                bottom2_price = prices[bottom2_idx]
                
                # Verificar similitud de valles
                price_diff = abs(bottom1_price - bottom2_price) / bottom1_price
                
                # Verificar que el pico es significativamente mayor
                peak_height = (peak_price - max(bottom1_price, bottom2_price)) / peak_price
                
                if price_diff <= tolerance and peak_height > 0.02:
                    confidence = 1 - price_diff
                    
                    # Calcular objetivo de precio
                    pattern_height = peak_price - max(bottom1_price, bottom2_price)
                    price_target = peak_price + pattern_height
                    
                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        start_idx=bottom1_idx,
                        end_idx=bottom2_idx,
                        confidence=confidence,
                        price_target=price_target,
                        stop_loss=min(bottom1_price, bottom2_price) * 0.98,
                        description=f"Doble suelo detectado. Valle 1: {bottom1_price:.2f}, Valle 2: {bottom2_price:.2f}, Pico: {peak_price:.2f}"
                    ))
        
        return patterns
    
    def detect_head_and_shoulders(self, df: pd.DataFrame, tolerance: float = 0.05) -> List[Pattern]:
        """
        Detecta patrón de cabeza y hombros
        
        Formación: Hombro izquierdo, cabeza (pico más alto), hombro derecho
        Señal: Bajista
        """
        patterns = []
        prices = df['High'].values
        maxima, _ = self.find_local_extrema(prices, order=3)
        
        if len(maxima) < 3:
            return patterns
        
        for i in range(len(maxima) - 2):
            left_idx = maxima[i]
            head_idx = maxima[i + 1]
            right_idx = maxima[i + 2]
            
            left_price = prices[left_idx]
            head_price = prices[head_idx]
            right_price = prices[right_idx]
            
            # Verificar que la cabeza es más alta que los hombros
            if head_price > left_price and head_price > right_price:
                # Verificar similitud de hombros
                shoulder_diff = abs(left_price - right_price) / left_price
                
                if shoulder_diff <= tolerance:
                    confidence = 1 - shoulder_diff
                    
                    # Encontrar línea de cuello (neckline)
                    neckline_start = min(left_idx, right_idx)
                    neckline_end = max(left_idx, right_idx)
                    neckline_prices = df['Low'].iloc[neckline_start:neckline_end].values
                    neckline = np.min(neckline_prices)
                    
                    # Calcular objetivo
                    pattern_height = head_price - neckline
                    price_target = neckline - pattern_height
                    
                    patterns.append(Pattern(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        start_idx=left_idx,
                        end_idx=right_idx,
                        confidence=confidence,
                        price_target=price_target,
                        stop_loss=head_price * 1.02,
                        description=f"Cabeza y hombros. Hombro I: {left_price:.2f}, Cabeza: {head_price:.2f}, Hombro D: {right_price:.2f}"
                    ))
        
        return patterns
    
    def detect_triangles(self, df: pd.DataFrame, tolerance: float = 0.03) -> List[Pattern]:
        """
        Detecta patrones de triángulos (ascendente, descendente, simétrico)
        """
        patterns = []
        highs = df['High'].values
        lows = df['Low'].values
        
        # Encontrar líneas de tendencia
        window = min(self.window_size, len(df) // 3)
        
        if window < 10:
            return patterns
        
        # Analizar convergencia de máximos y mínimos
        for i in range(window, len(df) - window):
            segment_highs = highs[i-window:i]
            segment_lows = lows[i-window:i]
            
            # Calcular tendencias
            x = np.arange(len(segment_highs))
            
            # Tendencia de máximos
            high_slope = np.polyfit(x, segment_highs, 1)[0]
            # Tendencia de mínimos
            low_slope = np.polyfit(x, segment_lows, 1)[0]
            
            # Triángulo ascendente: línea de resistencia horizontal, soporte ascendente
            if abs(high_slope) < tolerance and low_slope > tolerance:
                patterns.append(Pattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.7,
                    description="Triángulo ascendente detectado"
                ))
            
            # Triángulo descendente: línea de soporte horizontal, resistencia descendente
            elif abs(low_slope) < tolerance and high_slope < -tolerance:
                patterns.append(Pattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.7,
                    description="Triángulo descendente detectado"
                ))
            
            # Triángulo simétrico: convergencia de líneas
            elif high_slope < -tolerance and low_slope > tolerance:
                patterns.append(Pattern(
                    pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    start_idx=i-window,
                    end_idx=i,
                    confidence=0.6,
                    description="Triángulo simétrico detectado"
                ))
        
        return patterns
    
    def detect_wedges(self, df: pd.DataFrame, tolerance: float = 0.03) -> List[Pattern]:
        """
        Detecta patrones de cuñas (ascendente y descendente)
        """
        patterns = []
        highs = df['High'].values
        lows = df['Low'].values
        
        window = min(self.window_size, len(df) // 3)
        
        if window < 10:
            return patterns
        
        for i in range(window, len(df) - window):
            segment_highs = highs[i-window:i]
            segment_lows = lows[i-window:i]
            
            x = np.arange(len(segment_highs))
            
            high_slope = np.polyfit(x, segment_highs, 1)[0]
            low_slope = np.polyfit(x, segment_lows, 1)[0]
            
            # Cuña ascendente: ambas líneas suben, línea de soporte más empinada
            if high_slope > tolerance and low_slope > tolerance * 1.5:
                if high_slope < low_slope:  # Convergencia hacia arriba
                    patterns.append(Pattern(
                        pattern_type=PatternType.RISING_WEDGE,
                        start_idx=i-window,
                        end_idx=i,
                        confidence=0.65,
                        description="Cuña ascendente detectada (señal bajista)"
                    ))
            
            # Cuña descendente: ambas líneas bajan, línea de resistencia más empinada
            elif high_slope < -tolerance and low_slope < -tolerance * 1.5:
                if high_slope < low_slope:  # Convergencia hacia abajo
                    patterns.append(Pattern(
                        pattern_type=PatternType.FALLING_WEDGE,
                        start_idx=i-window,
                        end_idx=i,
                        confidence=0.65,
                        description="Cuña descendente detectada (señal alcista)"
                    ))
        
        return patterns
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Detecta todos los patrones disponibles en el dataframe
        """
        all_patterns = []
        
        all_patterns.extend(self.detect_double_top(df))
        all_patterns.extend(self.detect_double_bottom(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_triangles(df))
        all_patterns.extend(self.detect_wedges(df))
        
        # Ordenar por confianza
        all_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        self.patterns_detected = all_patterns
        
        logger.info(f"Detectados {len(all_patterns)} patrones")
        for p in all_patterns[:5]:  # Mostrar top 5
            logger.info(f"  - {p.pattern_type.value}: confianza={p.confidence:.2f}")
        
        return all_patterns
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """
        Retorna resumen de patrones detectados
        """
        if not self.patterns_detected:
            return pd.DataFrame()
        
        data = []
        for p in self.patterns_detected:
            data.append({
                'Patrón': p.pattern_type.value,
                'Confianza': p.confidence,
                'Inicio': p.start_idx,
                'Fin': p.end_idx,
                'Objetivo': p.price_target,
                'Stop Loss': p.stop_loss,
                'Descripción': p.description
            })
        
        return pd.DataFrame(data)


def visualize_patterns(df: pd.DataFrame, patterns: List[Pattern], symbol: str = ""):
    """
    Visualiza los patrones detectados en un gráfico
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Gráfico de precios
    ax1.plot(df.index, df['Close'], label='Close', color='black', linewidth=1)
    ax1.plot(df.index, df['High'], alpha=0.3, color='gray', linewidth=0.5)
    ax1.plot(df.index, df['Low'], alpha=0.3, color='gray', linewidth=0.5)
    
    # Colorear patrones
    colors = {
        PatternType.DOUBLE_TOP: 'red',
        PatternType.DOUBLE_BOTTOM: 'green',
        PatternType.HEAD_AND_SHOULDERS: 'orange',
        PatternType.ASCENDING_TRIANGLE: 'blue',
        PatternType.DESCENDING_TRIANGLE: 'purple',
        PatternType.SYMMETRICAL_TRIANGLE: 'cyan',
        PatternType.RISING_WEDGE: 'magenta',
        PatternType.FALLING_WEDGE: 'lime'
    }
    
    for pattern in patterns:
        color = colors.get(pattern.pattern_type, 'gray')
        ax1.axvspan(df.index[pattern.start_idx], df.index[pattern.end_idx], 
                   alpha=0.2, color=color, label=pattern.pattern_type.value)
    
    ax1.set_title(f'Patrones Chartistas Detectados - {symbol}')
    ax1.set_ylabel('Precio')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Volumen
    ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray')
    ax2.set_ylabel('Volumen')
    ax2.set_xlabel('Fecha')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
