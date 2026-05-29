"""
Agente de Control de Riesgo REAL
Evalúa y gestiona riesgos usando predicciones GARCH de volatilidad
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent, AnalysisResult
from models.ml_models.volatility_models import (
    GARCHVolatilityModel, calculate_realized_volatility,
    calculate_parkinson_volatility, calculate_garman_klass_volatility
)


@dataclass
class RiskMetrics:
    """Métricas de riesgo calculadas matemáticamente"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    volatility_forecast: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    ulcer_index: float
    beta: Optional[float] = None


class RiskManagerAgent(BaseAgent):
    """
    Agente especializado en gestión de riesgos.
    Usa modelos GARCH para predicción de volatilidad y
    ajusta el position sizing dinámicamente.
    """
    
    def __init__(self, 
                 max_position_size: float = 0.2,
                 max_portfolio_var: float = 0.02,
                 max_drawdown_limit: float = 0.15,
                 risk_target: float = 0.10,  # Volatilidad objetivo anual
                 use_garch: bool = True):
        super().__init__(
            name="RiskManager",
            role="Gestión de Riesgos",
            description="Experto en gestión de riesgos financieros. "
                       "Calcula VaR, CVaR, y predice volatilidad usando GARCH. "
                       "Ajusta el tamaño de posición basándose en el riesgo esperado."
        )
        
        # Límites de riesgo
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.max_drawdown_limit = max_drawdown_limit
        self.risk_target = risk_target
        self.use_garch = use_garch
        
        # Modelos de volatilidad
        self.garch_model = None
        self.volatility_forecast = None
        
        # Métricas actuales
        self.current_metrics = {}
        self.position_risks = {}
    
    def process_message(self, message):
        """Procesa mensajes recibidos"""
        if message.message_type == 'TRADE_PROPOSAL':
            self.log("Recibida propuesta de operación para evaluación de riesgo")
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Realiza análisis de riesgo completo con predicción GARCH.
        
        Args:
            data: Diccionario con datos de precios, posiciones propuestas, etc.
        """
        symbol = data.get('symbol', 'Unknown')
        price_data = data.get('price_data')
        proposed_position = data.get('proposed_position', {})
        portfolio = data.get('portfolio', {})
        
        self.log(f"Evaluando riesgo para {symbol}...")
        
        # 1. CALCULAR MÉTRICAS DE RIESGO REALES
        risk_metrics = self._calculate_risk_metrics(price_data)
        
        # 2. PREDECIR VOLATILIDAD CON GARCH
        if self.use_garch and price_data is not None:
            self._fit_garch_and_forecast(price_data)
        
        # 3. EVALUAR RIESGO DE LA POSICIÓN PROPUESTA
        position_risk = self._evaluate_position_risk(
            symbol, proposed_position, risk_metrics
        )
        
        # 4. EVALUAR RIESGO DE PORTFOLIO
        portfolio_risk = self._evaluate_portfolio_risk(
            portfolio, proposed_position, risk_metrics
        )
        
        # 5. CALCULAR TAMAÑO DE POSICIÓN ÓPTIMO (Position Sizing)
        optimal_position = self._calculate_optimal_position_size(
            risk_metrics, proposed_position
        )
        
        # 6. CALCULAR NIVELES DE STOP-LOSS Y TAKE-PROFIT
        levels = self._calculate_risk_levels(
            price_data, risk_metrics, proposed_position
        )
        
        # 7. GENERAR RECOMENDACIÓN
        recommendation, confidence = self._generate_risk_assessment(
            position_risk, portfolio_risk, risk_metrics
        )
        
        # 8. GENERAR RAZONAMIENTO DETALLADO
        reasoning = self._generate_reasoning(
            risk_metrics, position_risk, portfolio_risk, 
            optimal_position, levels
        )
        
        result = AnalysisResult(
            agent_name=self.name,
            analysis_type='risk',
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            data={
                'risk_metrics': {
                    'var_95': risk_metrics.var_95,
                    'var_99': risk_metrics.var_99,
                    'cvar_95': risk_metrics.cvar_95,
                    'volatility': risk_metrics.volatility,
                    'volatility_forecast': risk_metrics.volatility_forecast,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'sortino_ratio': risk_metrics.sortino_ratio,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'ulcer_index': risk_metrics.ulcer_index,
                    'calmar_ratio': risk_metrics.calmar_ratio
                },
                'position_risk': position_risk,
                'portfolio_risk': portfolio_risk,
                'optimal_position': optimal_position,
                'recommended_levels': levels,
                'garch_forecast': self.volatility_forecast.metrics if self.volatility_forecast else None
            }
        )
        
        self.add_to_memory({
            'symbol': symbol,
            'var_95': risk_metrics.var_95,
            'volatility_forecast': risk_metrics.volatility_forecast,
            'recommendation': recommendation
        })
        
        return result
    
    def _fit_garch_and_forecast(self, price_data: pd.DataFrame):
        """Ajusta modelo GARCH y realiza forecast de volatilidad"""
        try:
            # Calcular retornos
            returns = price_data['Close'].pct_change().dropna().values
            
            if len(returns) < 100:
                logger.warning("⚠️  Datos insuficientes para GARCH (< 100 observaciones)")
                return
            
            # Crear y ajustar modelo GARCH
            self.garch_model = GARCHVolatilityModel(
                p=1, q=1, 
                model_type='GARCH',
                distribution='t'
            )
            
            self.log("Ajustando modelo GARCH(1,1) para predicción de volatilidad...")
            self.garch_model.fit(returns, disp='off')
            
            # Realizar forecast
            self.volatility_forecast = self.garch_model.forecast(horizon=5)
            
            self.log(f"✅ Volatilidad predicha (5d): {self.volatility_forecast.forecast_value*100:.2f}%")
            
        except Exception as e:
            logger.warning(f"⚠️  Error en GARCH: {e}")
            self.garch_model = None
            self.volatility_forecast = None
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame) -> RiskMetrics:
        """Calcula métricas de riesgo matemáticamente"""
        if price_data is None or price_data.empty or len(price_data) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        returns = price_data['Close'].pct_change().dropna().values
        
        if len(returns) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        q05 = np.percentile(returns, 5)
        q01 = np.percentile(returns, 1)
        var_95 = -q05
        var_99 = -q01

        tail_95 = returns[returns <= q05]
        tail_99 = returns[returns <= q01]
        cvar_95 = -tail_95.mean() if tail_95.size > 0 else var_95
        cvar_99 = -tail_99.mean() if tail_99.size > 0 else var_99
        cvar_95 = max(cvar_95, var_95)
        cvar_99 = max(cvar_99, var_99)
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = -drawdown.min()
        
        # Ulcer Index (promedio cuadrático de drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdown[drawdown < 0] ** 2))
        
        # Volatilidad anualizada
        volatility = std_ret * np.sqrt(252)
        
        # Volatilidad predicha por GARCH
        volatility_forecast = self.volatility_forecast.forecast_value if self.volatility_forecast else volatility
        
        # Sharpe Ratio (asumiendo tasa libre de riesgo del 2%)
        risk_free_rate = 0.02
        excess_returns = np.mean(returns) * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino Ratio (solo downside deviation)
        mar_daily = risk_free_rate / 252
        downside_diff = np.minimum(returns - mar_daily, 0.0)
        downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
        sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio (retorno / max drawdown)
        total_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            volatility_forecast=volatility_forecast,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            ulcer_index=ulcer_index
        )
    
    def _evaluate_position_risk(self, symbol: str, position: Dict, 
                                metrics: RiskMetrics) -> Dict:
        """Evalúa el riesgo de una posición específica"""
        position_size = position.get('size', 0)
        entry_price = position.get('entry_price', 0)
        position_type = position.get('type', 'LONG')
        
        position_value = position_size * entry_price
        
        # Pérdida esperada en escenario adverso (VaR)
        expected_loss_var = position_value * metrics.var_95
        expected_loss_cvar = position_value * metrics.cvar_95
        
        # Riesgo de volatilidad
        daily_vol_risk = position_value * metrics.volatility / np.sqrt(252)
        
        # Riesgo de volatilidad predicha (GARCH)
        forecast_vol_risk = position_value * metrics.volatility_forecast / np.sqrt(252)
        
        return {
            'position_value': position_value,
            'expected_loss_var': expected_loss_var,
            'expected_loss_cvar': expected_loss_cvar,
            'daily_vol_risk': daily_vol_risk,
            'forecast_vol_risk': forecast_vol_risk,
            'risk_level': self._classify_risk_level(metrics)
        }
    
    def _evaluate_portfolio_risk(self, portfolio: Dict, proposed_position: Dict,
                                 metrics: RiskMetrics) -> Dict:
        """Evalúa el riesgo agregado del portfolio"""
        current_positions = portfolio.get('positions', {})
        portfolio_value = portfolio.get('total_value', 100000)
        
        # Calcular concentración
        proposed_value = proposed_position.get('size', 0) * proposed_position.get('entry_price', 0)
        current_exposure = sum(p.get('value', 0) for p in current_positions.values())
        total_exposure = current_exposure + proposed_value
        
        concentration = proposed_value / portfolio_value if portfolio_value > 0 else 0
        
        # VaR del portfolio (usando volatilidad predicha)
        portfolio_var = portfolio_value * metrics.var_95

        # VaR con volatilidad GARCH
        portfolio_var_forecast = portfolio_value * metrics.volatility_forecast / np.sqrt(252) * 1.645
        
        return {
            'portfolio_value': portfolio_value,
            'current_exposure': current_exposure,
            'total_exposure': total_exposure,
            'concentration': concentration,
            'portfolio_var': portfolio_var,
            'portfolio_var_forecast': portfolio_var_forecast,
            'concentration_risk': concentration > self.max_position_size
        }
    
    def _calculate_optimal_position_size(self, metrics: RiskMetrics, 
                                         proposed_position: Dict) -> Dict:
        """
        Calcula el tamaño de posición óptimo basado en el riesgo.
        Usa el modelo de Kelly fractal y volatilidad GARCH.
        """
        portfolio_value = proposed_position.get('portfolio_value', 100000)
        
        # Tamaño basado en VaR
        var_based_size = self.max_portfolio_var / (metrics.var_95 + 1e-10)
        
        # Tamaño basado en volatilidad predicha (GARCH)
        if metrics.volatility_forecast > 0:
            vol_based_size = self.risk_target / metrics.volatility_forecast
        else:
            vol_based_size = var_based_size
        
        # Tamaño basado en Kelly (simplificado)
        # f* = (bp - q) / b, donde b = odds, p = prob ganar, q = prob perder
        # Simplificación: usar Sharpe ratio como proxy
        if metrics.sharpe_ratio > 0:
            kelly_fraction = min(metrics.sharpe_ratio / 2, 0.25)  # Kelly fraccional
        else:
            kelly_fraction = 0
        
        # Tamaño más conservador
        optimal_size_fraction = min(var_based_size, vol_based_size, kelly_fraction, self.max_position_size)
        
        optimal_value = portfolio_value * optimal_size_fraction
        
        return {
            'optimal_fraction': optimal_size_fraction,
            'optimal_value': optimal_value,
            'var_based_fraction': var_based_size,
            'vol_based_fraction': vol_based_size,
            'kelly_fraction': kelly_fraction,
            'max_allowed': self.max_position_size
        }
    
    def _calculate_risk_levels(self, price_data: pd.DataFrame, 
                               metrics: RiskMetrics,
                               position: Dict) -> Dict:
        """Calcula niveles de stop-loss y take-profit basados en volatilidad"""
        if price_data is None or price_data.empty:
            return {}
        
        current_price = price_data['Close'].iloc[-1]
        
        # Usar volatilidad predicha si está disponible
        vol = metrics.volatility_forecast if metrics.volatility_forecast > 0 else metrics.volatility
        daily_vol = vol / np.sqrt(252)
        
        position_type = position.get('type', 'LONG')
        
        # Stop-loss basado en múltiplos de volatilidad
        # 2x volatilidad diaria para stop-loss
        # 4x volatilidad diaria para take-profit (ratio 1:2)
        
        if position_type == 'LONG':
            stop_loss = current_price * (1 - 2 * daily_vol)
            take_profit = current_price * (1 + 4 * daily_vol)
        else:  # SHORT
            stop_loss = current_price * (1 + 2 * daily_vol)
            take_profit = current_price * (1 - 4 * daily_vol)
        
        # Niveles adicionales basados en ATR si está disponible
        if 'ATR' in price_data.columns:
            atr = price_data['ATR'].iloc[-1]
            atr_stop = current_price - 2 * atr if position_type == 'LONG' else current_price + 2 * atr
        else:
            atr_stop = stop_loss
        
        # Soporte y resistencia recientes
        support_level = price_data['Low'].rolling(20).min().iloc[-1]
        resistance_level = price_data['High'].rolling(20).max().iloc[-1]
        
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        
        return {
            'entry_price': current_price,
            'stop_loss_vol': stop_loss,
            'stop_loss_atr': atr_stop,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': reward_amount / risk_amount if risk_amount > 0 else 0,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'trailing_stop': current_price * (1 - 1.5 * daily_vol) if position_type == 'LONG' 
                            else current_price * (1 + 1.5 * daily_vol)
        }
    
    def _classify_risk_level(self, metrics: RiskMetrics) -> str:
        """Clasifica el nivel de riesgo"""
        score = 0
        
        # VaR
        if metrics.var_95 > 0.05:
            score += 2
        elif metrics.var_95 > 0.03:
            score += 1
        
        # Volatilidad
        if metrics.volatility > 0.5:
            score += 2
        elif metrics.volatility > 0.3:
            score += 1
        
        # Drawdown
        if metrics.max_drawdown > 0.2:
            score += 2
        elif metrics.max_drawdown > 0.1:
            score += 1
        
        # Ulcer Index
        if metrics.ulcer_index > 0.1:
            score += 1
        
        if score >= 4:
            return 'ALTO'
        elif score >= 2:
            return 'MEDIO'
        else:
            return 'BAJO'
    
    def _generate_risk_assessment(self, position_risk: Dict, portfolio_risk: Dict,
                                  metrics: RiskMetrics) -> Tuple[str, float]:
        """Genera evaluación de riesgo final"""
        
        violations = []
        
        if portfolio_risk.get('concentration_risk'):
            violations.append('Concentración excesiva')
        
        if metrics.var_95 > self.max_portfolio_var:
            violations.append('VaR excede límite')
        
        if metrics.max_drawdown > self.max_drawdown_limit:
            violations.append('Drawdown histórico elevado')
        
        if metrics.volatility_forecast > 0.6:
            violations.append('Volatilidad predicha extrema')
        
        if metrics.ulcer_index > 0.15:
            violations.append('Ulcer Index elevado')
        
        # Generar recomendación
        if len(violations) >= 2:
            recommendation = 'RECHAZAR'
            confidence = 0.9
        elif len(violations) == 1:
            recommendation = 'RECHAZAR'
            confidence = 0.7
        elif position_risk['risk_level'] == 'ALTO':
            recommendation = 'APROBAR_CON_PRECAUCION'
            confidence = 0.6
        elif position_risk['risk_level'] == 'MEDIO':
            recommendation = 'APROBAR'
            confidence = 0.8
        else:
            recommendation = 'APROBAR'
            confidence = 0.9
        
        return recommendation, confidence
    
    def _generate_reasoning(self, metrics: RiskMetrics, position_risk: Dict,
                           portfolio_risk: Dict, optimal_position: Dict,
                           levels: Dict) -> str:
        """Genera explicación detallada del análisis de riesgo"""
        parts = []
        
        # Métricas principales
        parts.append(f"VaR 95%: {metrics.var_95*100:.2f}%")
        parts.append(f"CVaR 95%: {metrics.cvar_95*100:.2f}%")
        parts.append(f"Volatilidad: {metrics.volatility*100:.1f}%")
        
        if metrics.volatility_forecast != metrics.volatility:
            parts.append(f"Vol. predicha (GARCH): {metrics.volatility_forecast*100:.1f}%")
        
        parts.append(f"Max Drawdown: {metrics.max_drawdown*100:.1f}%")
        parts.append(f"Ulcer Index: {metrics.ulcer_index:.3f}")
        parts.append(f"Sharpe: {metrics.sharpe_ratio:.2f} | Sortino: {metrics.sortino_ratio:.2f}")
        
        # Riesgo de posición
        parts.append(f"Nivel de riesgo: {position_risk['risk_level']}")
        parts.append(f"Pérdida esperada (VaR): ${position_risk['expected_loss_var']:,.2f}")
        
        # Tamaño óptimo
        parts.append(f"Tamaño óptimo: {optimal_position['optimal_fraction']*100:.1f}%")
        
        # Concentración
        if portfolio_risk.get('concentration'):
            parts.append(f"Concentración: {portfolio_risk['concentration']*100:.1f}%")
        
        # Niveles
        if levels:
            parts.append(f"Stop-loss: ${levels['stop_loss_vol']:.2f}")
            parts.append(f"Take-profit: ${levels['take_profit']:.2f}")
            parts.append(f"R/R: 1:{levels['risk_reward_ratio']:.1f}")
        
        return " | ".join(parts)
    
    def approve_trade(self, trade_proposal: Dict, risk_analysis: AnalysisResult) -> bool:
        """Aprueba o rechaza una operación propuesta"""
        recommendation = risk_analysis.recommendation
        
        approved = recommendation in ['APROBAR', 'APROBAR_CON_PRECAUCION']
        
        if approved:
            self.log(f"✅ Operación APROBADA para {trade_proposal.get('symbol')}")
        else:
            self.log(f"❌ Operación RECHAZADA para {trade_proposal.get('symbol')}")
        
        return approved
