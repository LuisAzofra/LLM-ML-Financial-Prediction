"""
Módulo de Backtesting y Evaluación Económica Rigurosa
Implementa métricas de rendimiento profesionales según estándares QuantAgents
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representa una operación de trading"""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    position_type: str  # 'LONG' o 'SHORT'
    size: float
    pnl: Optional[float]
    return_pct: Optional[float]
    exit_reason: Optional[str]


@dataclass
class BacktestResult:
    """Resultados completos de un backtest con métricas económicas rigurosas"""
    # Métricas básicas
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    
    # Métricas de trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    payoff_ratio: float
    
    # Métricas de riesgo-ajustadas
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Métricas de drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_factor: float
    
    # Métricas avanzadas
    ulcer_index: float
    ulcer_performance_index: float
    kelly_criterion: float
    common_sense_ratio: float
    
    # Volatilidad
    volatility: float
    downside_volatility: float
    upside_volatility: float
    
    # Métricas de retornos
    skewness: float
    kurtosis: float
    value_at_risk_95: float
    conditional_var_95: float
    
    # Series temporales
    equity_curve: pd.Series
    returns_series: pd.Series
    drawdown_series: pd.Series
    trades: List[Trade]


class Backtester:
    """
    Motor de backtesting profesional con métricas económicas rigurosas.
    Basado en metodología QuantAgents para evaluación de estrategias.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,      # 0.1%
                 slippage: float = 0.0005,       # 0.05%
                 risk_free_rate: float = 0.02):  # 2% anual
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
    def run_backtest(self, 
                     df: pd.DataFrame, 
                     signal_generator: Callable,
                     symbol: str = 'UNKNOWN',
                     position_sizer: Optional[Callable] = None) -> BacktestResult:
        """
        Ejecuta backtest de una estrategia con métricas económicas completas.
        
        Args:
            df: DataFrame con datos de precios
            signal_generator: Función que genera señales (-1, 0, 1)
            symbol: Símbolo del activo
            position_sizer: Función opcional para ajustar tamaño de posición
            
        Returns:
            BacktestResult con métricas económicas completas
        """
        logger.info(f"Iniciando backtest para {symbol}...")
        
        capital = self.initial_capital
        position = 0  # 0 = sin posición, 1 = largo, -1 = corto
        position_size = 0
        entry_price = 0
        
        trades = []
        equity_curve = [capital]
        returns_list = []
        
        current_trade = None
        
        for i in range(1, len(df) - 1):
            current_price = df['Close'].iloc[i]
            date = df.index[i]
            
            # Generar señal
            historical_data = df.iloc[:i+1]
            signal = signal_generator(historical_data)
            
            # Aplicar position sizing si está disponible
            if position_sizer:
                position_value = position_sizer(capital, historical_data)
            else:
                position_value = capital * 0.95  # Default: 95% del capital
            
            # Ejecutar operaciones
            if signal == 1 and position <= 0:  # Señal de compra
                # Cerrar posición corta si existe
                if position == -1:
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) * position_size - \
                          (entry_price + exit_price) * position_size * self.commission
                    capital += pnl
                    
                    current_trade.exit_date = date
                    current_trade.exit_price = exit_price
                    current_trade.pnl = pnl
                    current_trade.return_pct = (exit_price - entry_price) / entry_price * -100
                    current_trade.exit_reason = 'SIGNAL'
                    trades.append(current_trade)
                
                # Abrir posición larga
                position_size = position_value / (current_price * (1 + self.slippage))
                entry_price = current_price * (1 + self.slippage)
                position = 1
                
                current_trade = Trade(
                    entry_date=date,
                    exit_date=None,
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=None,
                    position_type='LONG',
                    size=position_size,
                    pnl=None,
                    return_pct=None,
                    exit_reason=None
                )
            
            elif signal == -1 and position >= 0:  # Señal de venta
                # Cerrar posición larga si existe
                if position == 1:
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position_size - \
                          (entry_price + exit_price) * position_size * self.commission
                    capital += pnl
                    
                    current_trade.exit_date = date
                    current_trade.exit_price = exit_price
                    current_trade.pnl = pnl
                    current_trade.return_pct = (exit_price - entry_price) / entry_price * 100
                    current_trade.exit_reason = 'SIGNAL'
                    trades.append(current_trade)
                
                # Abrir posición corta
                position_size = position_value / (current_price * (1 - self.slippage))
                entry_price = current_price * (1 - self.slippage)
                position = -1
                
                current_trade = Trade(
                    entry_date=date,
                    exit_date=None,
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=None,
                    position_type='SHORT',
                    size=position_size,
                    pnl=None,
                    return_pct=None,
                    exit_reason=None
                )
            
            # Calcular valor del portfolio
            if position == 0:
                portfolio_value = capital
            elif position == 1:
                position_value = position_size * current_price
                portfolio_value = capital - position_size * entry_price + position_value
            else:  # position == -1
                position_value = position_size * (2 * entry_price - current_price)
                portfolio_value = capital - position_size * entry_price + position_value
            
            equity_curve.append(portfolio_value)
            
            # Calcular retorno diario
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                returns_list.append(daily_return)
        
        # Cerrar posición final
        if position != 0:
            final_price = df['Close'].iloc[-1]
            if position == 1:
                exit_price = final_price * (1 - self.slippage)
                pnl = (exit_price - entry_price) * position_size
            else:
                exit_price = final_price * (1 + self.slippage)
                pnl = (entry_price - exit_price) * position_size
            
            capital += pnl
            current_trade.exit_date = df.index[-1]
            current_trade.exit_price = exit_price
            current_trade.pnl = pnl
            trades.append(current_trade)
            equity_curve[-1] = capital
        
        # Calcular métricas económicas completas
        result = self._calculate_metrics(equity_curve, returns_list, trades, df)
        
        logger.info(f"Backtest completado: {result.total_trades} operaciones")
        logger.info(f"Retorno: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _calculate_metrics(self, 
                          equity_curve: List[float], 
                          returns_list: List[float],
                          trades: List[Trade],
                          df: pd.DataFrame) -> BacktestResult:
        """Calcula métricas económicas completas"""
        
        equity_series = pd.Series(equity_curve)
        returns_series = pd.Series(returns_list)
        
        # === MÉTRICAS BÁSICAS ===
        initial_capital = equity_curve[0]
        final_capital = equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Retorno anualizado
        n_days = len(equity_curve)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        
        # === MÉTRICAS DE TRADING ===
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl and t.pnl <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit Factor
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Payoff Ratio (ratio ganancia/pérdida promedio)
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # === MÉTRICAS DE RIESGO ===
        # Volatilidad anualizada
        volatility = returns_series.std() * np.sqrt(252)
        
        mar_daily = self.risk_free_rate / 252
        downside_diff = np.minimum(returns_series - mar_daily, 0.0)
        downside_volatility = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
        
        # Upside volatility (solo retornos positivos)
        upside_returns = returns_series[returns_series > 0]
        upside_volatility = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
        
        # === MÉTRICAS RIESGO-AJUSTADAS ===
        # Sharpe Ratio
        excess_returns = returns_series.mean() * 252 - self.risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino Ratio (usa downside deviation)
        sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
        
        # === MÉTRICAS DE DRAWDOWN ===
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
        
        # Duración del máximo drawdown
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery Factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calmar Ratio (retorno anualizado / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # === MÉTRICAS AVANZADAS ===
        # Ulcer Index (promedio cuadrático de drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdown[drawdown < 0] ** 2)) if any(drawdown < 0) else 0
        
        # Ulcer Performance Index (retorno / ulcer index)
        ulcer_performance_index = annualized_return / ulcer_index if ulcer_index > 0 else 0
        
        # Kelly Criterion (simplificado)
        win_prob = win_rate
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        kelly_criterion = (win_prob * avg_win_loss_ratio - (1 - win_prob)) / avg_win_loss_ratio if avg_win_loss_ratio != 0 else 0
        kelly_criterion = max(0, min(kelly_criterion, 0.25))  # Limitar a 25%
        
        # Common Sense Ratio (Profit Factor * Payoff Ratio)
        common_sense_ratio = profit_factor * payoff_ratio if payoff_ratio != float('inf') else profit_factor
        
        # Omega Ratio (ratio de ganancias sobre pérdidas)
        threshold = 0  # Umbral de retorno
        gains = returns_series[returns_series > threshold] - threshold
        losses = threshold - returns_series[returns_series <= threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() != 0 else float('inf')
        
        # === MÉTRICAS DE DISTRIBUCIÓN ===
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()
        
        # Value at Risk y Conditional VaR
        var_95 = -np.percentile(returns_series, 5)
        cvar_95 = -returns_series[returns_series <= -var_95].mean() if any(returns_series <= -var_95) else var_95
        
        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            avg_drawdown=avg_drawdown,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            ulcer_performance_index=ulcer_performance_index,
            kelly_criterion=kelly_criterion,
            common_sense_ratio=common_sense_ratio,
            volatility=volatility,
            downside_volatility=downside_volatility,
            upside_volatility=upside_volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            equity_curve=equity_series,
            returns_series=returns_series,
            drawdown_series=drawdown,
            trades=trades
        )
    
    def print_report(self, result: BacktestResult):
        """Imprime reporte detallado del backtest"""
        print("\n" + "=" * 70)
        print("REPORTE DE BACKTEST - MÉTRICAS ECONÓMICAS")
        print("=" * 70)
        
        print(f"\n📊 RENDIMIENTO:")
        print(f"   Capital Inicial: ${result.initial_capital:,.2f}")
        print(f"   Capital Final: ${result.final_capital:,.2f}")
        print(f"   Retorno Total: {result.total_return*100:.2f}%")
        print(f"   Retorno Anualizado: {result.annualized_return*100:.2f}%")
        
        print(f"\n📈 OPERACIONES:")
        print(f"   Total Operaciones: {result.total_trades}")
        print(f"   Ganadoras: {result.winning_trades}")
        print(f"   Perdedoras: {result.losing_trades}")
        print(f"   Win Rate: {result.win_rate*100:.1f}%")
        print(f"   Ganancia Promedio: ${result.avg_win:,.2f}")
        print(f"   Pérdida Promedio: ${result.avg_loss:,.2f}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print(f"   Payoff Ratio: {result.payoff_ratio:.2f}")
        
        print(f"\n⚖️  RIESGO-AJUSTADO:")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Sortino Ratio: {result.sortino_ratio:.3f}")
        print(f"   Calmar Ratio: {result.calmar_ratio:.3f}")
        print(f"   Omega Ratio: {result.omega_ratio:.3f}")
        
        print(f"\n📉 DRAWDOWN:")
        print(f"   Max Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"   Avg Drawdown: {result.avg_drawdown*100:.2f}%")
        print(f"   Max DD Duration: {result.max_drawdown_duration} días")
        print(f"   Recovery Factor: {result.recovery_factor:.2f}")
        
        print(f"\n🔬 AVANZADAS:")
        print(f"   Ulcer Index: {result.ulcer_index:.4f}")
        print(f"   Ulcer Performance: {result.ulcer_performance_index:.2f}")
        print(f"   Kelly Criterion: {result.kelly_criterion*100:.1f}%")
        print(f"   Common Sense Ratio: {result.common_sense_ratio:.2f}")
        
        print(f"\n📊 VOLATILIDAD:")
        print(f"   Volatilidad: {result.volatility*100:.1f}%")
        print(f"   Downside Vol: {result.downside_volatility*100:.1f}%")
        print(f"   Upside Vol: {result.upside_volatility*100:.1f}%")
        
        print(f"\n📈 DISTRIBUCIÓN:")
        print(f"   Skewness: {result.skewness:.3f}")
        print(f"   Kurtosis: {result.kurtosis:.3f}")
        print(f"   VaR 95%: {result.value_at_risk_95*100:.2f}%")
        print(f"   CVaR 95%: {result.conditional_var_95*100:.2f}%")
        
        print("\n" + "=" * 70)


class RobustnessTester:
    """
    Tester de robustez para evaluar estrategias en diferentes regímenes de mercado.
    Basado en metodología QuantAgents para validación en mercados reales.
    """
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
    
    def test_regime_robustness(self, 
                               df: pd.DataFrame,
                               signal_generator: Callable,
                               symbol: str) -> Dict[str, BacktestResult]:
        """
        Testea robustez en diferentes regímenes de mercado.
        
        Args:
            df: DataFrame con datos de precios
            signal_generator: Generador de señales
            symbol: Símbolo
            
        Returns:
            Dict con resultados por régimen
        """
        # Calcular volatilidad rolling para identificar regímenes
        returns = df['Close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Definir regímenes
        vol_median = rolling_vol.median()
        
        regimes = {
            'low_volatility': df[rolling_vol < vol_median * 0.7],
            'normal': df[(rolling_vol >= vol_median * 0.7) & (rolling_vol <= vol_median * 1.3)],
            'high_volatility': df[rolling_vol > vol_median * 1.3]
        }
        
        results = {}
        
        for regime_name, regime_df in regimes.items():
            if len(regime_df) > 50:  # Mínimo de datos
                logger.info(f"\nTesteando régimen: {regime_name} ({len(regime_df)} días)")
                
                try:
                    result = self.backtester.run_backtest(
                        regime_df, signal_generator, symbol
                    )
                    results[regime_name] = result
                    
                    logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}, "
                              f"MaxDD: {result.max_drawdown*100:.1f}%")
                except Exception as e:
                    logger.warning(f"  Error en régimen {regime_name}: {e}")
        
        return results
    
    def test_parameter_sensitivity(self,
                                   df: pd.DataFrame,
                                   base_params: Dict,
                                   param_ranges: Dict[str, List],
                                   signal_generator_factory: Callable,
                                   symbol: str) -> pd.DataFrame:
        """
        Testea sensibilidad a parámetros.
        
        Args:
            df: DataFrame con datos
            base_params: Parámetros base
            param_ranges: Rangos de parámetros a testear
            signal_generator_factory: Factory que crea signal generator con params
            symbol: Símbolo
            
        Returns:
            DataFrame con resultados
        """
        results = []
        
        # Generar combinaciones de parámetros
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            params = base_params.copy()
            for name, value in zip(param_names, combination):
                params[name] = value
            
            try:
                signal_gen = signal_generator_factory(params)
                result = self.backtester.run_backtest(df, signal_gen, symbol)
                
                row = params.copy()
                row.update({
                    'sharpe': result.sharpe_ratio,
                    'sortino': result.sortino_ratio,
                    'max_dd': result.max_drawdown,
                    'total_return': result.total_return,
                    'profit_factor': result.profit_factor
                })
                results.append(row)
                
            except Exception as e:
                logger.warning(f"Error con params {params}: {e}")
        
        return pd.DataFrame(results)
    
    def calculate_robustness_score(self, 
                                   regime_results: Dict[str, BacktestResult]) -> Dict[str, float]:
        """
        Calcula score de robustez basado en resultados por régimen.
        
        Args:
            regime_results: Resultados por régimen
            
        Returns:
            Dict con scores de robustez
        """
        if not regime_results:
            return {'overall': 0}
        
        sharpes = [r.sharpe_ratio for r in regime_results.values()]
        max_dds = [r.max_drawdown for r in regime_results.values()]
        
        # Robustez: Sharpe consistente en todos los regímenes
        sharpe_mean = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        sharpe_robustness = sharpe_mean / (sharpe_std + 0.1)  # Penaliza variabilidad
        
        # Robustez de drawdown
        max_dd_mean = np.mean(max_dds)
        max_dd_robustness = 1 / (abs(max_dd_mean) + 0.01)
        
        # Score overall
        overall = (sharpe_robustness * 0.6 + max_dd_robustness * 0.4)
        
        return {
            'overall': overall,
            'sharpe_robustness': sharpe_robustness,
            'max_dd_robustness': max_dd_robustness,
            'sharpe_by_regime': {k: v.sharpe_ratio for k, v in regime_results.items()},
            'max_dd_by_regime': {k: v.max_drawdown for k, v in regime_results.items()}
        }


# Funciones de utilidad para análisis

def calculate_beta(returns_strategy: np.ndarray, 
                   returns_market: np.ndarray) -> float:
    """Calcula beta de la estrategia respecto al mercado"""
    mask = ~(np.isnan(returns_strategy) | np.isnan(returns_market))
    if np.sum(mask) < 2:
        return 0
    
    cov = np.cov(returns_strategy[mask], returns_market[mask])[0, 1]
    var_market = np.var(returns_market[mask])
    
    return cov / var_market if var_market > 0 else 0


def calculate_alpha(returns_strategy: np.ndarray,
                   returns_market: np.ndarray,
                   risk_free_rate: float = 0.02) -> float:
    """Calcula alpha de Jensen"""
    beta = calculate_beta(returns_strategy, returns_market)
    
    mean_strategy = np.mean(returns_strategy) * 252
    mean_market = np.mean(returns_market) * 252
    
    alpha = mean_strategy - (risk_free_rate + beta * (mean_market - risk_free_rate))
    return alpha


def calculate_information_ratio(returns_strategy: np.ndarray,
                                returns_benchmark: np.ndarray) -> float:
    """Calcula Information Ratio"""
    active_returns = returns_strategy - returns_benchmark
    tracking_error = np.std(active_returns) * np.sqrt(252)
    
    return np.mean(active_returns) * 252 / tracking_error if tracking_error > 0 else 0


def calculate_treynor_ratio(returns_strategy: np.ndarray,
                           returns_market: np.ndarray,
                           risk_free_rate: float = 0.02) -> float:
    """Calcula Treynor Ratio"""
    beta = calculate_beta(returns_strategy, returns_market)
    excess_return = np.mean(returns_strategy) * 252 - risk_free_rate
    
    return excess_return / beta if beta > 0 else 0
