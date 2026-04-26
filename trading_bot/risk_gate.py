"""
risk_gate.py — Tier 1.3 risk control determinístico.

Inspirado en el finding #1 del paper AI-Trader (HKU, arXiv 2512.10971):
*"general intelligence does not translate to trading capability — risk
control determines cross-market robustness"*. La diferencia entre +22%
(Qwen3 Max) y −60% (GPT-5/Gemini) sobre las mismas señales fue
exclusivamente el control de riesgo, no la accuracy.

Cinco capas determinísticas (NO LLM-driven), aplicadas antes de cada
nueva entrada y al inicio de cada jornada:

1. **Vol-target overlay (cartera)**: si la vol realizada de la equity
   curve a 60d > target → reduce tamaño de nuevas entradas. NO compite
   con el `vol_scaling` por activo — operan en niveles distintos:
     · vol_scaling actual (api.py:2483): target_vol / vol_activo
     · vol-target overlay (este gate): target_vol / vol_portfolio

2. **Daily loss limit**: si pnl_diario < −3% del equity → congelar
   nuevas entradas hasta cierre. Sólo se gestionan salidas.

3. **Kill-switch drawdown**: si DD intra-ventana > 12% peak-to-trough
   → cerrar todo y pausar 5 días.

4. **Cap por símbolo**: posición ≤ 25% del equity (formaliza lo que hoy
   es implícito por Kelly).

5. **Cap exposición total**: Σ |w_i| ≤ 100% (sin shorts ya implícito;
   con shorts es restricción real).

Diseño: un solo objeto `RiskGate` con estado, llamado en 2 puntos del
loop de backtest:
  - `on_day_start(ds, equity)`: actualiza peak, gestiona kill-switch
    y daily-loss; devuelve `force_liquidate` (bool) si toca cerrar todo
  - `gate_new_position(ds, equity, open_positions, sym, candidate_size_pct, equity_vals)`:
    devuelve (allowed, adjusted_size_pct, reason)

Activación: con `use_risk_gate=True` en el body del backtest. Default
False para preservar baseline; tras validación se promueve a True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RiskGateConfig:
    """Configuración del risk gate."""
    target_vol_annual: float = 0.15            # vol objetivo a nivel portfolio
    daily_loss_limit_pct: float = 0.03         # -3% intraday → freeze entries
    kill_switch_dd_pct: float = 0.12           # 12% peak-to-trough → liquidate+pause
    kill_switch_pause_days: int = 5            # días de pausa post-kill
    max_position_pct_per_sym: float = 0.25     # cap por símbolo (25%)
    max_total_exposure_pct: float = 1.0        # cap exposición total
    vol_overlay_min: float = 0.30              # multiplier mínimo (no anular del todo)
    vol_overlay_max: float = 1.50              # multiplier máximo (no apalancar más allá)
    vol_lookback_days: int = 60                # ventana para realizar vol portfolio


@dataclass
class RiskGateState:
    """Estado mutable del risk gate."""
    peak_equity: float = 0.0
    paused_until_ds: Optional[str] = None
    last_pause_reason: str = ''
    daily_loss_locked_until_ds: Optional[str] = None
    last_day_open_equity: Optional[float] = None
    last_day_ds: Optional[str] = None
    triggers: Dict[str, int] = field(default_factory=lambda: {
        'vol_overlay_clip': 0,
        'daily_loss_freeze': 0,
        'kill_switch': 0,
        'cap_per_symbol': 0,
        'cap_total_exposure': 0,
    })


class RiskGate:
    """Wrapper determinístico de control de riesgo. Sin LLM, sin ML."""

    def __init__(self, cfg: Optional[RiskGateConfig] = None):
        self.cfg = cfg or RiskGateConfig()
        self.state = RiskGateState()

    # ── Vol-target overlay ────────────────────────────────────────────────
    def _portfolio_realized_vol(self, equity_vals: List[float]) -> float:
        """Vol anualizada de la equity curve sobre los últimos N días."""
        n = self.cfg.vol_lookback_days
        if len(equity_vals) < 10:
            return self.cfg.target_vol_annual  # neutral hasta tener datos
        arr = np.asarray(equity_vals[-(n + 1):], dtype=float)
        if len(arr) < 5:
            return self.cfg.target_vol_annual
        rets = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1.0)
        # filter NaN/inf
        rets = rets[np.isfinite(rets)]
        if len(rets) < 3:
            return self.cfg.target_vol_annual
        vol = float(np.std(rets, ddof=1) * np.sqrt(252))
        return max(min(vol, 3.0), 0.01)

    def vol_overlay_multiplier(self, equity_vals: List[float]) -> float:
        """Multiplier ∈ [vol_overlay_min, vol_overlay_max] para size_pct."""
        portfolio_vol = self._portfolio_realized_vol(equity_vals)
        ratio = self.cfg.target_vol_annual / max(portfolio_vol, 0.05)
        return float(np.clip(ratio, self.cfg.vol_overlay_min, self.cfg.vol_overlay_max))

    # ── Drawdown / kill-switch ────────────────────────────────────────────
    def update_peak(self, equity: float) -> None:
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

    def current_drawdown(self, equity: float) -> float:
        peak = max(self.state.peak_equity, equity)
        if peak <= 0:
            return 0.0
        return (equity - peak) / peak  # negativo

    def is_paused(self, ds: str) -> bool:
        return (
            self.state.paused_until_ds is not None
            and ds < self.state.paused_until_ds
        )

    def _should_kill_switch(self, equity: float) -> bool:
        return self.current_drawdown(equity) <= -self.cfg.kill_switch_dd_pct

    def _apply_kill_switch(self, ds: str, equity_at_kill: float) -> None:
        # avanzar paused_until_ds = ds + kill_switch_pause_days laborables (aprox calendario)
        import pandas as pd
        anchor = pd.Timestamp(ds)
        self.state.paused_until_ds = (
            anchor + pd.Timedelta(days=self.cfg.kill_switch_pause_days)
        ).strftime('%Y-%m-%d')
        self.state.last_pause_reason = f"kill_switch @ {ds} equity={equity_at_kill:.0f}"
        self.state.triggers['kill_switch'] += 1
        # CRÍTICO: tras un kill, "olvidamos" el peak anterior y empezamos a
        # contar DD desde el equity actual. Sin esto, mientras el equity
        # siga > 12% bajo el peak histórico, cada día post-pausa volvería a
        # disparar kill-switch — loop indeseado.
        self.state.peak_equity = float(equity_at_kill)

    # ── Daily loss limit ──────────────────────────────────────────────────
    def _maybe_register_day_open(self, ds: str, equity: float) -> None:
        if self.state.last_day_ds != ds:
            self.state.last_day_open_equity = equity
            self.state.last_day_ds = ds

    def _daily_loss_breached(self, equity_now: float) -> bool:
        eo = self.state.last_day_open_equity
        if eo is None or eo <= 0:
            return False
        loss = (equity_now - eo) / eo
        return loss <= -self.cfg.daily_loss_limit_pct

    # ── Hooks principales ────────────────────────────────────────────────
    def on_day_start(self, ds: str, equity: float) -> Dict[str, object]:
        """
        Llamar al inicio de cada jornada. Actualiza estado y devuelve dict con:
          - 'force_liquidate' (bool): True si kill-switch acaba de activarse
          - 'paused' (bool): True si la jornada está en pausa post-kill
          - 'daily_lock' (bool): True si daily loss bloquea nuevas entradas
        """
        self.update_peak(equity)
        self._maybe_register_day_open(ds, equity)

        force_liquidate = False
        if self._should_kill_switch(equity) and not self.is_paused(ds):
            self._apply_kill_switch(ds, equity)
            force_liquidate = True

        paused = self.is_paused(ds)
        daily_lock = self._daily_loss_breached(equity)
        if daily_lock:
            self.state.daily_loss_locked_until_ds = ds
            self.state.triggers['daily_loss_freeze'] += 1

        return {
            'force_liquidate': force_liquidate,
            'paused': paused,
            'daily_lock': daily_lock,
            'peak_equity': self.state.peak_equity,
            'current_dd': self.current_drawdown(equity),
        }

    def gate_new_position(
        self,
        ds: str,
        equity: float,
        open_position_value_total: float,
        sym: str,
        candidate_size_pct: float,
        equity_vals: List[float],
    ) -> Tuple[bool, float, str]:
        """
        Decide si abrir una nueva posición y con qué tamaño.

        Returns (allowed, adjusted_size_pct, reason).
        - Si paused o daily-locked → (False, 0.0, reason)
        - Si excede cap por símbolo → trim a max_position_pct_per_sym
        - Si excede cap total exposición → trim a lo que quepa
        - Aplica vol-target overlay como multiplier final
        """
        if self.is_paused(ds):
            return False, 0.0, 'paused_post_kill_switch'

        if self._daily_loss_breached(equity):
            return False, 0.0, 'daily_loss_freeze'

        size = float(candidate_size_pct)

        # Cap por símbolo
        if size > self.cfg.max_position_pct_per_sym:
            self.state.triggers['cap_per_symbol'] += 1
            size = self.cfg.max_position_pct_per_sym

        # Cap exposición total
        current_exposure_pct = (open_position_value_total / equity) if equity > 0 else 1.0
        room = max(0.0, self.cfg.max_total_exposure_pct - current_exposure_pct)
        if size > room:
            self.state.triggers['cap_total_exposure'] += 1
            size = room
        if size <= 0:
            return False, 0.0, 'cap_total_exposure_full'

        # Vol-target overlay (cartera)
        overlay = self.vol_overlay_multiplier(equity_vals)
        if overlay < 1.0:
            self.state.triggers['vol_overlay_clip'] += 1
        size *= overlay

        return True, max(size, 0.0), f"ok overlay={overlay:.2f}"

    def stats(self) -> Dict[str, object]:
        return {
            'peak_equity':   self.state.peak_equity,
            'paused_until':  self.state.paused_until_ds,
            'last_reason':   self.state.last_pause_reason,
            **self.state.triggers,
        }
