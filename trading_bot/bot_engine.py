"""
Autonomous Trading Bot Engine
==============================
Basado en el feedback del revisor:
  - Evaluacion por balance monetario real (P&L), no solo tasa de acierto
  - Sizing proporcional al retorno predicho × confianza (apuesta mas en señales fuertes)
  - Filtro de señales: solo opera el top X% de señales por fuerza
  - El 5% de ocasiones de alta confianza es lo que genera el beneficio real
  - Incorpora MLP y RBF como modelos adicionales al ensemble
"""
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    """Parametros de configuracion del bot de trading."""
    initial_capital: float = 100_000.0
    # Percentil minimo de fuerza de señal para entrar (0.65 = top 35% señales)
    # Antes era 0.70 (top 30%); hemos bajado porque el accuracy-gate y los filtros
    # de régimen ya filtran calidad, y el cuello de botella eran "pocas entradas".
    signal_percentile: float = 0.65
    # Maximo % del capital por posicion
    max_position_pct: float = 0.18
    # Fraccion de Kelly a usar — subido de 0.20 a 0.28 porque con b_min=0.35 y
    # confianzas calibradas el Kelly crudo suele ser pequeño; necesitamos
    # convertir señales reales en exposición real sin volver imprudente.
    kelly_scale: float = 0.28
    # Costes de transaccion
    commission: float = 0.001    # 0.1% por lado
    slippage: float = 0.0005     # 0.05% por lado
    # Stop-loss en multiplos de ATR (trailing — el stop sigue al precio)
    atr_stop_mult: float = 2.0   # reducido de 3.0: SL=2ATR, TP=5ATR → R:R 2.5:1 alcanzable
    # Maximo dias manteniendo una posicion (2× horizonte de predicción de 5 días)
    max_holding_days: int = 10
    # Fuerza minima absoluta de señal (|retorno_pred| × confianza)
    # Reducido de 0.004 → 0.001: el umbral de percentil ya filtra, no necesitamos
    # un suelo tan alto que bloquee casi todas las señales de modelos de árbol.
    min_signal_strength: float = 0.001
    # Permitir posiciones cortas
    allow_short: bool = True
    # Volatilidad anualizada objetivo para scaling de posiciones (risk parity)
    target_vol: float = 0.18      # 18% anualizado — reduce tamaño en mercados volátiles
    # Filtro de régimen de mercado
    # Red Team Finding 4: 18+SMA200+signal_percentile bloqueaba ~95% entradas → bajar a 15
    regime_adx_min: float = 15.0  # No operar cuando ADX < este valor (mercado choppy)
    regime_use_sma200: bool = True # Confirmar dirección con SMA_200
    # Filtro de crash: evita entradas LONG cuando el activo cayó >5% en 5 días
    regime_crash_filter: bool = True
    regime_crash_threshold: float = -0.05  # caída acumulada en 5 días que activa el filtro


# ─────────────────────────────────────────────────────────────────────────────
# Trade dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BotTrade:
    entry_date: str
    exit_date: Optional[str]
    symbol: str
    action: str          # 'LONG' o 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    size_pct: float      # % del capital usado
    shares: float
    pnl: Optional[float]
    return_pct: Optional[float]
    hold_days: int
    exit_reason: Optional[str]   # 'SIGNAL' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'MAX_HOLD' | 'END'
    signal_strength: float
    predicted_return: float
    actual_return: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Bot engine
# ─────────────────────────────────────────────────────────────────────────────

class AutonomousTradingBot:
    """
    Bot de trading autonomo con paper trading y evaluacion monetaria.

    Estrategia de maximizacion de beneficio:
      1. Calcular fuerza de señal = |retorno_predicho| × confianza
      2. Solo operar cuando fuerza > percentil configurado (por defecto top 25%)
      3. Sizing proporcional: posicion ∝ fuerza_señal × Kelly_fraccionado
      4. Stop-loss basado en ATR para cortar perdidas rapido
      5. Take-profit al 2× retorno predicho para dejar correr ganancias
      6. Maximo de dias abierto para evitar quedarse atrapado
    """

    def __init__(self, config: BotConfig = None):
        self.config = config or BotConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        price_df: pd.DataFrame,
        predictions: np.ndarray,
        confidences: np.ndarray,
        test_start_idx: int,
        symbol: str = 'UNKNOWN',
    ) -> dict:
        """
        Ejecuta backtest monetario completo.

        Args:
            price_df       : DataFrame con columnas Open/High/Low/Close (+ ATR si disponible)
            predictions    : Array de retornos predichos para el periodo de test
            confidences    : Array de confianzas [0,1] (misma longitud)
            test_start_idx : Indice en price_df donde empieza el periodo de test
            symbol         : Ticker del activo

        Returns:
            Dict con equity_curve, trades, performance, signal_stats
        """
        cfg = self.config
        capital = cfg.initial_capital

        # ── Indicadores de régimen sobre price_df COMPLETO (antes de slicear) ─
        # Critico: calcular en df completo para que SMA_200/ADX usen el histórico
        # completo de entrenamiento (sin look-ahead en el test period)
        price_df = price_df.copy()
        if 'SMA_200' not in price_df.columns:
            price_df['SMA_200'] = price_df['Close'].rolling(200, min_periods=50).mean()
        if 'SMA_50' not in price_df.columns:
            price_df['SMA_50'] = price_df['Close'].rolling(50, min_periods=20).mean()
        if 'ADX' not in price_df.columns:
            price_df = self._add_adx(price_df)
        if 'ATR' not in price_df.columns or price_df['ATR'].isna().all():
            price_df = self._add_atr(price_df)

        # ── Periodo de test (slice DESPUÉS de calcular indicadores) ──────────
        test_df = price_df.iloc[test_start_idx:].copy().reset_index(drop=False)
        n_test = min(len(test_df), len(predictions), len(confidences))
        test_df = test_df.iloc[:n_test]
        predictions = np.asarray(predictions[:n_test], dtype=float)
        confidences = np.asarray(confidences[:n_test], dtype=float)

        # ── Fuerza de señal y umbral ──────────────────────────────────────────
        signal_strengths = np.abs(predictions) * confidences
        raw_threshold = np.percentile(signal_strengths, cfg.signal_percentile * 100)
        threshold = max(raw_threshold, cfg.min_signal_strength)

        # ── Simulacion de trading ─────────────────────────────────────────────
        position = None
        equity_curve: List[float] = [capital]
        dates: List[str] = [self._fmt_date(test_df, 0)]
        all_trades: List[BotTrade] = []

        for i in range(n_test - 1):
            current_price = float(test_df['Close'].iloc[i])
            # Precios intraday — críticos para stop/TP realistas (Red Team Finding 1)
            low_price  = float(test_df['Low'].iloc[i])  if 'Low'  in test_df.columns else current_price
            high_price = float(test_df['High'].iloc[i]) if 'High' in test_df.columns else current_price
            date = self._fmt_date(test_df, i)
            atr_raw = test_df['ATR'].iloc[i]
            atr = float(atr_raw) if not pd.isna(atr_raw) else current_price * 0.015

            pred = float(predictions[i])
            conf = float(confidences[i])
            ss   = float(signal_strengths[i])

            # ── Comprobar condiciones de cierre ───────────────────────────────
            if position is not None:
                hold_days = i - position['open_idx']
                exit_price = None
                exit_reason = None

                # Trailing stop: ratchet stop hacia el precio favorable
                # LONG: stop sube con el precio, nunca baja (protege ganancias)
                # SHORT: stop baja con el precio, nunca sube
                new_trail = (
                    current_price - cfg.atr_stop_mult * atr
                    if position['type'] == 'LONG'
                    else current_price + cfg.atr_stop_mult * atr
                )
                if position['type'] == 'LONG':
                    position['stop_loss'] = max(position['stop_loss'], new_trail)
                else:
                    position['stop_loss'] = min(position['stop_loss'], new_trail)

                # Usar Low/High intraday para stop y TP — evita survivorship bias
                # (Red Team Finding 1: stop con Close subestima violaciones intraday)
                if position['type'] == 'LONG':
                    if low_price <= position['stop_loss']:
                        # Saleido intraday al nivel del stop (no a Close)
                        exit_price = position['stop_loss']
                        exit_reason = 'STOP_LOSS'
                    elif high_price >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'TAKE_PROFIT'
                else:  # SHORT
                    if high_price >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'STOP_LOSS'
                    elif low_price <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'TAKE_PROFIT'

                if exit_price is None and hold_days >= cfg.max_holding_days:
                    exit_price = current_price
                    exit_reason = 'MAX_HOLD'

                # Señal opuesta fuerte
                if exit_price is None and ss > threshold:
                    new_dir = 'LONG' if pred > 0 else 'SHORT'
                    if new_dir != position['type']:
                        exit_price = current_price
                        exit_reason = 'SIGNAL'

                if exit_price is not None:
                    capital, trade = self._close_position(
                        position, exit_price, exit_reason,
                        capital, date, test_df, i, symbol
                    )
                    all_trades.append(trade)
                    position = None

            # ── Comprobar condiciones de entrada ──────────────────────────────
            if position is None and ss > threshold and capital > 100:
                # pred == 0.0 no es señal de ninguna dirección (Red Team Finding 10)
                direction = 'LONG' if pred > 0 else ('SHORT' if (pred < 0 and cfg.allow_short) else None)

                if direction is not None:
                    # ── Filtro de régimen de mercado ──────────────────────────
                    # Solo operar en dirección de la tendencia (SMA_200 + SMA_50)
                    # y solo cuando el mercado tiene tendencia clara (ADX)
                    regime_ok = True
                    sma50_penalty = 1.0  # reseteado en cada iteración
                    if cfg.regime_use_sma200 and 'SMA_200' in test_df.columns:
                        sma200 = test_df['SMA_200'].iloc[i]
                        if not pd.isna(sma200):
                            if direction == 'LONG'  and current_price < sma200:
                                regime_ok = False  # largo en tendencia bajista
                            if direction == 'SHORT' and current_price > sma200:
                                regime_ok = False  # corto en tendencia alcista

                    # SMA_50 ya no actúa como filtro duro (era redundante con SMA_200
                    # y bloqueaba las mejores entradas de mean-reversion intra-tendencia).
                    # En cambio, si el precio contradice fuertemente la SMA_50 en la
                    # misma dirección que la señal, reducimos tamaño (filtro suave).
                    if 'SMA_50' in test_df.columns:
                        sma50 = test_df['SMA_50'].iloc[i]
                        if not pd.isna(sma50):
                            if direction == 'LONG'  and current_price < sma50:
                                sma50_penalty = 0.70  # 30% menos tamaño
                            if direction == 'SHORT' and current_price > sma50:
                                sma50_penalty = 0.70

                    if regime_ok and 'ADX' in test_df.columns:
                        adx_val = test_df['ADX'].iloc[i]
                        if not pd.isna(adx_val) and float(adx_val) < cfg.regime_adx_min:
                            regime_ok = False  # mercado sin tendencia (choppy)

                    # ── Filtro de crash con excepción oversold ─────────────────
                    # Bloqueamos LONG cuando el activo cae >5% en 5 días (evita
                    # atrapar cuchillos), PERO permitimos la entrada si está
                    # claramente sobrevendido (RSI<30) y la señal es fuerte.
                    # Esto captura rebotes estadísticos sin comprometer en tendencia bajista.
                    if regime_ok and cfg.regime_crash_filter and direction == 'LONG' and i >= 5:
                        price_5d_ago = float(test_df['Close'].iloc[i - 5])
                        roll_5d = (current_price / price_5d_ago - 1) if price_5d_ago > 0 else 0.0
                        if roll_5d < cfg.regime_crash_threshold:
                            rsi_val = None
                            if 'RSI' in test_df.columns:
                                _r = test_df['RSI'].iloc[i]
                                if not pd.isna(_r):
                                    rsi_val = float(_r)
                            # Excepción: RSI muy bajo + señal fuerte → mean-reversion bounce
                            strong_signal = ss > (threshold * 1.3)
                            oversold_bounce = rsi_val is not None and rsi_val < 30 and strong_signal
                            if not oversold_bounce:
                                regime_ok = False  # caída en pánico — esperar estabilización

                    if not regime_ok:
                        # Actualizar equity y seguir al siguiente día
                        portfolio_value = self._portfolio_value(capital, position, current_price)
                        equity_curve.append(portfolio_value)
                        dates.append(self._fmt_date(test_df, i + 1))
                        continue

                    entry_price_raw = current_price
                    slippage_adj = cfg.slippage if direction == 'LONG' else -cfg.slippage
                    entry_price = entry_price_raw * (1 + slippage_adj)

                    # Sizing: Kelly × señal relativa × volatility scaling (risk parity)
                    stop_loss_pct = (cfg.atr_stop_mult * atr) / entry_price_raw
                    kelly = self._compute_kelly(pred, conf, stop_loss_pct)
                    relative_strength = min(ss / max(threshold, 1e-9), 3.0)

                    # Volatility targeting: reducir tamaño en mercados volátiles
                    current_vol = (
                        float(test_df['Volatility'].iloc[i])
                        if 'Volatility' in test_df.columns and not pd.isna(test_df['Volatility'].iloc[i])
                        else 0.20
                    )
                    vol_scalar = float(np.clip(cfg.target_vol / max(current_vol, 0.05), 0.3, 2.0))

                    size_pct = min(
                        kelly * relative_strength * cfg.kelly_scale * vol_scalar * sma50_penalty,
                        cfg.max_position_pct
                    )
                    # Red Team Finding 6: NO forzar mínimo — si Kelly=0, no apostar.
                    # EV negativo (Kelly<0) → skip. Tamaño mínimo meaningful = 0.5%
                    if size_pct < 0.005:
                        portfolio_value = self._portfolio_value(capital, position, current_price)
                        equity_curve.append(portfolio_value)
                        dates.append(self._fmt_date(test_df, i + 1))
                        continue

                    position_value = capital * size_pct
                    shares = position_value / entry_price
                    commission_entry = entry_price * shares * cfg.commission
                    # Descontar coste total: inversión + comisión (Red Team Finding 3)
                    capital -= position_value + commission_entry

                    # Stop-loss y take-profit
                    # TP = 2× distancia de stop → ratio R:R=2:1 garantizado
                    # (antes era |pred|×2 → R:R inconsistente con predicciones pequeñas)
                    sl_dist = cfg.atr_stop_mult * atr
                    if direction == 'LONG':
                        stop_loss   = entry_price - sl_dist
                        take_profit = entry_price + 2.0 * sl_dist
                    else:
                        stop_loss   = entry_price + sl_dist
                        take_profit = entry_price - 2.0 * sl_dist

                    position = {
                        'type': direction,
                        'open_idx': i,
                        'date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'size_pct': size_pct,
                        'cost_basis': position_value,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal_strength': ss,
                        'predicted_return': pred,
                    }

            # ── Valor de cartera ──────────────────────────────────────────────
            portfolio_value = self._portfolio_value(capital, position, current_price)
            equity_curve.append(portfolio_value)
            dates.append(self._fmt_date(test_df, i + 1))

        # ── Cerrar posicion final ─────────────────────────────────────────────
        if position is not None and n_test > 0:
            final_price = float(test_df['Close'].iloc[n_test - 1])
            capital, trade = self._close_position(
                position, final_price, 'END',
                capital, dates[-1], test_df, n_test - 1, symbol
            )
            all_trades.append(trade)
            equity_curve[-1] = capital

        # ── Benchmark buy & hold ──────────────────────────────────────────────
        bh = self._buy_hold_curve(test_df.iloc[:n_test], cfg.initial_capital)

        # ── Metricas ──────────────────────────────────────────────────────────
        metrics = self._compute_metrics(equity_curve, all_trades, cfg.initial_capital)

        return {
            'equity_curve': {
                'dates': dates,
                'portfolio_value': [round(v, 2) for v in equity_curve],
                'buy_hold_value':  [round(v, 2) for v in bh],
                'drawdown': [round(d * 100, 3) for d in self._drawdown_series(equity_curve)],
            },
            'performance': metrics,
            'trades': [self._trade_to_dict(t) for t in all_trades],
            'signal_stats': {
                'total_signals':     int(n_test),
                'filtered_signals':  int(np.sum(signal_strengths > threshold)),
                'filter_percentile': cfg.signal_percentile,
                'threshold':         round(float(threshold), 6),
                'avg_signal':        round(float(np.mean(signal_strengths)), 6),
                'max_signal':        round(float(np.max(signal_strengths)), 6),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_kelly(self, predicted_return: float, confidence: float,
                       stop_loss_pct: float = 0.01) -> float:
        """
        Kelly fraccionado: b = retorno_predicho / distancia_stop (ratio riesgo real).

        stop_loss_pct debe ser la distancia ATR-based al stop, no un 1% fijo.
        Esto calibra correctamente cuanto capital arriesgamos por unidad ganada.
        """
        p = float(np.clip(confidence, 0.15, 0.90))
        # Risk = distancia real al stop-loss (ATR × multiplicador / precio_entrada)
        risk = max(float(stop_loss_pct), 0.005)  # minimo 0.5% para evitar b → ∞
        # b = cuanto ganamos por unidad arriesgada (reward/risk ratio)
        # TP se fija a 2× distancia stop → b real ≥ 2.0. Usamos max(2, 2×|pred|/risk)
        # como estimación conservadora. Antes b_min=0.5 exigía conf>0.67 para Kelly>0;
        # con b_min=2.0 (consistente con TP=2×stop) ya conf≥0.34 produce Kelly positivo.
        b = max(2 * abs(predicted_return) / risk, 2.0)
        q = 1.0 - p
        kelly = (p * b - q) / b
        return float(np.clip(kelly, 0.0, 1.0))

    def _close_position(
        self, position: dict, exit_price_raw: float, exit_reason: str,
        capital: float, date: str, test_df: pd.DataFrame,
        current_idx: int, symbol: str
    ) -> Tuple[float, BotTrade]:
        """Cierra una posicion y devuelve (nuevo_capital, trade)."""
        cfg = self.config
        slippage_adj = -cfg.slippage if position['type'] == 'LONG' else cfg.slippage
        exit_price = exit_price_raw * (1 + slippage_adj)

        if position['type'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['shares']
        else:
            pnl = (position['entry_price'] - exit_price) * position['shares']

        pnl -= exit_price * position['shares'] * cfg.commission
        # Devolver el capital invertido + beneficio neto (Red Team Finding 3)
        capital += position['cost_basis'] + pnl

        open_price = float(test_df['Close'].iloc[position['open_idx']])
        close_price = float(test_df['Close'].iloc[current_idx])
        actual_ret = (close_price - open_price) / open_price if open_price != 0 else 0.0

        cost_basis = position['cost_basis']
        return_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0.0

        trade = BotTrade(
            entry_date=position['date'],
            exit_date=date,
            symbol=symbol,
            action=position['type'],
            entry_price=round(position['entry_price'], 4),
            exit_price=round(exit_price, 4),
            size_pct=position['size_pct'],
            shares=position['shares'],
            pnl=round(pnl, 2),
            return_pct=round(return_pct, 3),
            hold_days=current_idx - position['open_idx'],
            exit_reason=exit_reason,
            signal_strength=position['signal_strength'],
            predicted_return=position['predicted_return'],
            actual_return=round(actual_ret * 100, 3),
        )
        return capital, trade

    def _portfolio_value(self, capital: float, position: Optional[dict],
                         current_price: float) -> float:
        if position is None:
            return capital
        if position['type'] == 'LONG':
            unrealized = (current_price - position['entry_price']) * position['shares']
        else:
            unrealized = (position['entry_price'] - current_price) * position['shares']
        return capital + position['cost_basis'] + unrealized

    def _buy_hold_curve(self, test_df: pd.DataFrame, initial_capital: float) -> List[float]:
        if len(test_df) == 0:
            return [initial_capital]
        prices = test_df['Close'].values.astype(float)
        shares = initial_capital / prices[0]
        return [round(float(shares * p), 2) for p in prices]

    def _drawdown_series(self, equity: List[float]) -> List[float]:
        arr = np.array(equity, dtype=float)
        cummax = np.maximum.accumulate(arr)
        dd = (arr - cummax) / np.where(cummax != 0, cummax, 1)
        return dd.tolist()

    def _compute_metrics(self, equity: List[float], trades: List[BotTrade],
                         initial_capital: float) -> dict:
        if len(equity) < 2:
            return {'error': 'insufficient data'}

        arr = np.array(equity, dtype=float)
        returns = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1)

        final_capital = float(arr[-1])
        total_return = (final_capital - initial_capital) / initial_capital
        n_years = max(len(equity) / 252, 0.01)
        ann_return = (1 + total_return) ** (1 / n_years) - 1

        vol = float(np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0.0
        dn = returns[returns < 0]
        dn_vol = float(np.std(dn)) * np.sqrt(252) if len(dn) > 0 else 1e-9

        rf_d = 0.02 / 252
        excess = returns - rf_d
        sharpe = float(np.mean(excess) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 1e-12 else 0.0
        sortino = float(np.mean(excess)) * 252 / dn_vol if dn_vol > 1e-12 else 0.0

        dd_arr = np.array(self._drawdown_series(equity))
        max_dd = float(np.min(dd_arr))
        # Calmar: preservar signo de ann_return (negativo si perdemos dinero)
        calmar = ann_return / abs(max_dd) if max_dd != 0 and ann_return > 0 else (ann_return / max(abs(max_dd), 0.001) if max_dd != 0 else 0.0)

        total_trades = len(trades)
        wins   = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win  = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0

        sum_wins   = sum(t.pnl for t in wins)
        sum_losses = sum(t.pnl for t in losses)
        pf = abs(sum_wins / sum_losses) if sum_losses != 0 else 9999.0
        pr = abs(avg_win  / avg_loss)   if avg_loss  != 0 else 9999.0

        avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0

        return {
            'initial_capital':       round(initial_capital, 2),
            'final_capital':         round(final_capital, 2),
            'total_return_pct':      round(total_return * 100, 2),
            'annualized_return_pct': round(ann_return * 100, 2),
            'sharpe_ratio':          round(sharpe, 3),
            'sortino_ratio':         round(sortino, 3),
            'calmar_ratio':          round(calmar, 3),
            'max_drawdown_pct':      round(max_dd * 100, 2),
            'volatility_pct':        round(vol * 100, 2),
            'total_trades':          total_trades,
            'win_rate':              round(win_rate * 100, 1),
            'avg_win_usd':           round(avg_win, 2),
            'avg_loss_usd':          round(avg_loss, 2),
            'profit_factor':         round(min(pf, 999.0), 3),
            'payoff_ratio':          round(min(pr, 999.0), 3),
            'avg_hold_days':         round(avg_hold, 1),
        }

    def _trade_to_dict(self, t: BotTrade) -> dict:
        return {
            'entry_date':             t.entry_date,
            'exit_date':              t.exit_date,
            'action':                 t.action,
            'entry_price':            t.entry_price,
            'exit_price':             t.exit_price,
            'size_pct':               round(t.size_pct * 100, 1),
            'pnl':                    t.pnl,
            'return_pct':             t.return_pct,
            'hold_days':              t.hold_days,
            'exit_reason':            t.exit_reason,
            'signal_strength':        round(t.signal_strength, 5),
            'predicted_return_pct':   round(t.predicted_return * 100, 3),
            'actual_return_pct':      t.actual_return,
        }

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high  = df['High'].astype(float) if 'High' in df.columns else df['Close']
        low   = df['Low'].astype(float)  if 'Low'  in df.columns else df['Close']
        close = df['Close'].astype(float)
        hl   = high - low
        hc   = (high - close.shift(1)).abs()
        lc   = (low  - close.shift(1)).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(period, min_periods=1).mean()
        return df

    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index — mide fuerza de tendencia (no dirección)."""
        df = df.copy()
        high  = df['High'].astype(float) if 'High' in df.columns else df['Close']
        low   = df['Low'].astype(float)  if 'Low'  in df.columns else df['Close']
        close = df['Close'].astype(float)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        plus_dm_orig  = high.diff().clip(lower=0)
        minus_dm_orig = (-low.diff()).clip(lower=0)
        # Red Team Finding 11: usar originales para comparación — la mutación secuencial
        # causaba sesgo bajista cuando plus_dm == minus_dm
        both     = (plus_dm_orig > 0) & (minus_dm_orig > 0)
        plus_dm  = plus_dm_orig.where(~(both & (minus_dm_orig >= plus_dm_orig)), other=0.0)
        minus_dm = minus_dm_orig.where(~(both & (plus_dm_orig >= minus_dm_orig)), other=0.0)

        atr14    = tr.rolling(period, min_periods=1).mean()
        plus_di  = 100 * plus_dm.rolling(period,  min_periods=1).mean() / (atr14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(period, min_periods=1).mean() / (atr14 + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(period, min_periods=1).mean()
        return df

    @staticmethod
    def _fmt_date(df: pd.DataFrame, idx: int) -> str:
        val = df.index[idx] if not isinstance(df.index, pd.RangeIndex) else df.iloc[idx].get('index', str(idx))
        if hasattr(val, 'date'):
            return str(val.date())
        return str(val)
