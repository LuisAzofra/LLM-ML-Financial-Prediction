"""
================================================================================
PAPER PORTFOLIO - Gestor de Cartera Simulada (Paper Trading)
Persiste el estado de la cartera en un fichero JSON y proporciona métodos
para ejecutar operaciones simuladas, consultar estado con precios en tiempo
real y registrar el historial de trades.
================================================================================
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

# Comisión aplicada en cada lado de la operación (entrada y salida)
COMMISSION_RATE = 0.001  # 0.1 %

# Ruta por defecto del fichero de persistencia
_DEFAULT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'paper_portfolio.json',
)


class PaperPortfolio:
    """
    Cartera de paper trading con persistencia en JSON.

    La cartera mantiene:
    - Efectivo disponible (cash)
    - Posiciones abiertas (positions)
    - Historial completo de trades cerrados (trade_history)

    Para las posiciones SHORT el beneficio se calcula como
    (precio_entrada - precio_actual) * shares, de modo que sube cuando el
    precio baja.
    """

    def __init__(self, portfolio_file: str = None, initial_capital: float = 100_000.0):
        """
        Inicializa la cartera.

        Args:
            portfolio_file: Ruta al fichero JSON de persistencia.
                            Por defecto ``<raiz_proyecto>/paper_portfolio.json``.
            initial_capital: Capital inicial en USD.  Solo se usa si el
                             fichero no existe todavía.
        """
        self.portfolio_file = portfolio_file or _DEFAULT_FILE
        self.initial_capital = initial_capital

        if os.path.exists(self.portfolio_file):
            self.load()
        else:
            self._init_fresh(initial_capital)
            self.save()

    # ─────────────────────────────────────────────────────────────────────
    # Persistencia
    # ─────────────────────────────────────────────────────────────────────

    def _init_fresh(self, initial_capital: float):
        """Inicializa el estado interno a una cartera vacía."""
        self.cash = float(initial_capital)
        self.initial_capital = float(initial_capital)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.positions: dict = {}       # símbolo -> dict de posición abierta
        self.trade_history: list = []   # lista de trades cerrados

    def load(self):
        """Carga el estado de la cartera desde el fichero JSON."""
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self.cash            = float(data.get('cash', self.initial_capital))
            self.initial_capital = float(data.get('initial_capital', self.initial_capital))
            self.created_at      = data.get('created_at', datetime.now(timezone.utc).isoformat())
            self.positions       = data.get('positions', {})
            self.trade_history   = data.get('trade_history', [])
            logger.info(f"[PaperPortfolio] Cartera cargada desde {self.portfolio_file}")
        except Exception as exc:
            logger.error(f"[PaperPortfolio] Error cargando cartera: {exc} — se inicializa en fresco")
            self._init_fresh(self.initial_capital)

    def save(self):
        """Persiste el estado actual de la cartera en el fichero JSON."""
        data = {
            'initial_capital': self.initial_capital,
            'cash':            self.cash,
            'created_at':      self.created_at,
            'positions':       self.positions,
            'trade_history':   self.trade_history,
        }
        try:
            with open(self.portfolio_file, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"[PaperPortfolio] Error guardando cartera: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades internas
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_price(symbol: str, fallback: float) -> float:
        """
        Obtiene el precio de cierre más reciente vía yfinance.
        Devuelve ``fallback`` si se produce algún error.
        """
        try:
            hist = yf.Ticker(symbol).history(period='1d')
            if hist.empty:
                logger.warning(f"[PaperPortfolio] yfinance devolvió datos vacíos para {symbol}")
                return fallback
            return float(hist['Close'].iloc[-1])
        except Exception as exc:
            logger.warning(f"[PaperPortfolio] Error obteniendo precio de {symbol}: {exc}")
            return fallback

    @staticmethod
    def _unrealized_pnl(position: dict, current_price: float) -> tuple[float, float]:
        """
        Calcula el P&L no realizado y su porcentaje para una posición abierta.

        Returns:
            (pnl_absoluto, pnl_porcentaje)
        """
        shares       = float(position['shares'])
        entry_price  = float(position['entry_price'])
        action       = position['action']  # 'LONG' o 'SHORT'

        if action == 'LONG':
            pnl = (current_price - entry_price) * shares
        else:  # SHORT
            pnl = (entry_price - current_price) * shares

        position_cost = entry_price * shares
        pnl_pct = (pnl / position_cost * 100) if position_cost > 0 else 0.0
        return pnl, pnl_pct

    def _realized_pnl_total(self) -> float:
        """Suma del P&L realizado de todos los trades cerrados."""
        return sum(float(t.get('pnl', 0)) for t in self.trade_history)

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def get_status(self, fetch_live_prices: bool = True) -> dict:
        """
        Devuelve el estado completo de la cartera.

        Args:
            fetch_live_prices: Si ``True`` consulta yfinance para obtener
                               precios actuales de las posiciones abiertas.

        Returns:
            Diccionario con cash, equity total, P&L realizado/no realizado,
            lista de posiciones con métricas de mercado y resumen general.
        """
        positions_with_pnl = []
        total_unrealized = 0.0

        for symbol, pos in self.positions.items():
            entry_price = float(pos['entry_price'])

            if fetch_live_prices:
                current_price = self._fetch_price(symbol, fallback=entry_price)
            else:
                current_price = entry_price

            pnl, pnl_pct = self._unrealized_pnl(pos, current_price)
            position_value = current_price * float(pos['shares'])
            total_unrealized += pnl

            positions_with_pnl.append({
                'symbol':             symbol,
                'action':             pos['action'],
                'asset_type':         pos.get('asset_type', 'stock'),
                'shares':             pos['shares'],
                'entry_price':        round(entry_price, 4),
                'entry_date':         pos['entry_date'],
                'current_price':      round(current_price, 4),
                'unrealized_pnl':     round(pnl, 2),
                'unrealized_pnl_pct': round(pnl_pct, 2),
                'position_value':     round(position_value, 2),
                'signal_strength':    pos.get('signal_strength'),
                'predicted_return_pct': pos.get('predicted_return_pct'),
            })

        realized_pnl  = self._realized_pnl_total()
        total_equity  = self.cash + sum(p['position_value'] for p in positions_with_pnl)
        total_return  = ((total_equity - self.initial_capital) / self.initial_capital * 100
                         if self.initial_capital > 0 else 0.0)

        return {
            'cash':              round(self.cash, 2),
            'initial_capital':   round(self.initial_capital, 2),
            'total_equity':      round(total_equity, 2),
            'unrealized_pnl':    round(total_unrealized, 2),
            'realized_pnl':      round(realized_pnl, 2),
            'total_return_pct':  round(total_return, 2),
            'created_at':        self.created_at,
            'num_trades':        len(self.trade_history),
            'num_open':          len(self.positions),
            'positions_with_pnl': positions_with_pnl,
        }

    def execute_trade(
        self,
        symbol: str,
        asset_type: str,
        action: str,
        size_pct: float,
        current_price: float,
        signal_strength: float,
        predicted_return_pct: float,
        reason: str = 'SIGNAL',
    ) -> dict:
        """
        Ejecuta una operación de paper trading y actualiza la cartera.

        Args:
            symbol:               Ticker del activo (p.ej. 'AAPL').
            asset_type:           Tipo de activo ('stock', 'crypto', etc.).
            action:               'LONG' o 'SHORT'.
            size_pct:             Fracción del efectivo a usar (0.0–1.0).
            current_price:        Precio actual de mercado.
            signal_strength:      Fuerza de la señal ML (|pred| * confidence).
            predicted_return_pct: Retorno predicho en porcentaje.
            reason:               Motivo de la operación (para el registro).

        Returns:
            Diccionario con los detalles de la posición abierta.

        Raises:
            ValueError: Si no hay suficiente efectivo, ya existe posición
                        en el símbolo, o el tamaño es inválido.
        """
        symbol = symbol.upper().strip()
        action = action.upper().strip()

        # Validaciones
        if action not in ('LONG', 'SHORT'):
            raise ValueError(f"Acción inválida '{action}'. Debe ser 'LONG' o 'SHORT'.")
        if not (0.0 < size_pct <= 1.0):
            raise ValueError(f"size_pct debe estar en (0, 1]. Recibido: {size_pct}")
        if symbol in self.positions:
            raise ValueError(f"Ya existe una posición abierta para {symbol}. "
                             "Ciérrala antes de abrir una nueva.")

        cash_to_use   = self.cash * size_pct
        commission    = cash_to_use * COMMISSION_RATE
        total_cost    = cash_to_use + commission

        if total_cost > self.cash:
            raise ValueError(
                f"Efectivo insuficiente. Necesario: ${total_cost:,.2f}, "
                f"disponible: ${self.cash:,.2f}."
            )

        shares = cash_to_use / current_price

        position = {
            'symbol':                  symbol,
            'asset_type':              asset_type,
            'action':                  action,
            'shares':                  round(shares, 8),
            'entry_price':             round(float(current_price), 6),
            'entry_date':              datetime.now(timezone.utc).isoformat(),
            'position_value_at_entry': round(cash_to_use, 2),
            'signal_strength':         round(float(signal_strength), 6),
            'predicted_return_pct':    round(float(predicted_return_pct), 4),
            'entry_reason':            reason,
        }

        self.positions[symbol] = position
        self.cash -= total_cost
        self.save()

        logger.info(
            f"[PaperPortfolio] {action} {symbol}: {shares:.4f} acciones @ ${current_price:.4f} "
            f"| coste ${total_cost:,.2f} (comis. ${commission:.2f})"
        )
        return position

    def close_position(
        self,
        symbol: str,
        current_price: float,
        reason: str = 'MANUAL',
    ) -> dict:
        """
        Cierra una posición abierta y registra el trade en el historial.

        Args:
            symbol:        Ticker del activo.
            current_price: Precio actual de cierre.
            reason:        Motivo del cierre ('MANUAL', 'SIGNAL', 'STOP', etc.).

        Returns:
            Registro del trade cerrado con P&L detallado.

        Raises:
            KeyError: Si el símbolo no tiene posición abierta.
        """
        symbol = symbol.upper().strip()

        if symbol not in self.positions:
            raise KeyError(f"No se encontró posición abierta para {symbol}.")

        pos         = self.positions[symbol]
        shares      = float(pos['shares'])
        entry_price = float(pos['entry_price'])
        action      = pos['action']
        entry_date  = pos['entry_date']

        # P&L bruto
        if action == 'LONG':
            gross_pnl = (current_price - entry_price) * shares
        else:  # SHORT
            gross_pnl = (entry_price - current_price) * shares

        # Comisión de salida sobre el valor liquidado
        exit_value  = current_price * shares
        commission  = exit_value * COMMISSION_RATE
        net_proceeds = exit_value - commission
        net_pnl      = gross_pnl - commission

        # Calcular días en posición
        try:
            entry_dt   = datetime.fromisoformat(entry_date)
            exit_dt    = datetime.now(timezone.utc)
            hold_days  = (exit_dt - entry_dt).days
        except Exception:
            hold_days = 0

        cost_basis   = entry_price * shares
        return_pct   = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        trade_record = {
            'id':           str(uuid.uuid4()),
            'symbol':       symbol,
            'asset_type':   pos.get('asset_type', 'stock'),
            'action':       action,
            'entry_date':   entry_date,
            'exit_date':    datetime.now(timezone.utc).isoformat(),
            'entry_price':  round(entry_price, 6),
            'exit_price':   round(float(current_price), 6),
            'shares':       round(shares, 8),
            'pnl':          round(net_pnl, 2),
            'return_pct':   round(return_pct, 2),
            'hold_days':    hold_days,
            'exit_reason':  reason,
        }

        # Actualizar estado
        del self.positions[symbol]
        self.cash += net_proceeds
        self.trade_history.append(trade_record)
        self.save()

        logger.info(
            f"[PaperPortfolio] Cerrada posición {action} {symbol} @ ${current_price:.4f} "
            f"| P&L neto: ${net_pnl:+,.2f} ({return_pct:+.2f}%)"
        )
        return trade_record

    def reset(self, initial_capital: float = None) -> dict:
        """
        Cierra todas las posiciones abiertas al precio de entrada (sin P&L)
        y reinicia la cartera al capital inicial.

        Args:
            initial_capital: Nuevo capital inicial. Si ``None`` se mantiene
                             el capital inicial original.

        Returns:
            Estado de la cartera tras el reinicio.
        """
        capital = float(initial_capital) if initial_capital is not None else self.initial_capital

        # Cerrar posiciones abiertas usando precio de entrada como precio de salida
        # (no tiene sentido consultar mercado en un reset manual)
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                self.close_position(symbol, current_price=float(pos['entry_price']),
                                    reason='RESET')
            except Exception as exc:
                logger.warning(f"[PaperPortfolio] Error cerrando {symbol} en reset: {exc}")

        # Reiniciar estado
        self._init_fresh(capital)
        self.save()

        logger.info(f"[PaperPortfolio] Cartera reiniciada. Capital inicial: ${capital:,.2f}")
        return self.get_status(fetch_live_prices=False)

    # ─────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────

    def _sessions_dir(self) -> str:
        """Returns the directory where named sessions are stored."""
        d = os.path.join(os.path.dirname(self.portfolio_file), 'sessions')
        os.makedirs(d, exist_ok=True)
        return d

    def save_session(self, name: str, description: str = '') -> dict:
        """
        Save the current portfolio state as a named session snapshot.
        The snapshot is stored as a JSON file in the sessions/ directory.
        """
        safe_name = name.strip().replace(' ', '_').replace('/', '-')[:50]
        if not safe_name:
            raise ValueError('El nombre de sesión no puede estar vacío')

        snapshot = {
            'session_name':    safe_name,
            'description':     description,
            'saved_at':        datetime.now(timezone.utc).isoformat(),
            'initial_capital': self.initial_capital,
            'cash':            self.cash,
            'created_at':      self.created_at,
            'positions':       self.positions,
            'trade_history':   self.trade_history,
        }

        path = os.path.join(self._sessions_dir(), f'{safe_name}.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)

        logger.info(f"[PaperPortfolio] Sesión guardada: {safe_name}")
        return {'session_name': safe_name, 'path': path, 'saved_at': snapshot['saved_at']}

    def load_session(self, name: str) -> dict:
        """
        Load a previously saved session snapshot into the active portfolio.
        Overwrites the current portfolio state.
        """
        safe_name = name.strip().replace(' ', '_').replace('/', '-')[:50]
        path = os.path.join(self._sessions_dir(), f'{safe_name}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sesión '{safe_name}' no encontrada")

        with open(path, 'r', encoding='utf-8') as fh:
            snapshot = json.load(fh)

        self.cash            = float(snapshot.get('cash', self.initial_capital))
        self.initial_capital = float(snapshot.get('initial_capital', self.initial_capital))
        self.created_at      = snapshot.get('created_at', datetime.now(timezone.utc).isoformat())
        self.positions       = snapshot.get('positions', {})
        self.trade_history   = snapshot.get('trade_history', [])
        self.save()

        logger.info(f"[PaperPortfolio] Sesión cargada: {safe_name}")
        return self.get_status(fetch_live_prices=False)

    def list_sessions(self) -> list:
        """
        Return metadata for all saved sessions, sorted newest-first.
        """
        sessions = []
        d = self._sessions_dir()
        for fname in os.listdir(d):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(d, fname)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
                # Compute summary stats
                n_pos    = len(meta.get('positions', {}))
                n_trades = len(meta.get('trade_history', []))
                init_cap = float(meta.get('initial_capital', 0))
                cash     = float(meta.get('cash', 0))
                sessions.append({
                    'session_name': meta.get('session_name', fname[:-5]),
                    'description':  meta.get('description', ''),
                    'saved_at':     meta.get('saved_at', ''),
                    'initial_capital': init_cap,
                    'cash':           cash,
                    'open_positions': n_pos,
                    'closed_trades':  n_trades,
                })
            except Exception:
                pass
        sessions.sort(key=lambda s: s.get('saved_at', ''), reverse=True)
        return sessions

    def delete_session(self, name: str) -> bool:
        """Delete a saved session file. Returns True if deleted."""
        safe_name = name.strip().replace(' ', '_').replace('/', '-')[:50]
        path = os.path.join(self._sessions_dir(), f'{safe_name}.json')
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"[PaperPortfolio] Sesión eliminada: {safe_name}")
            return True
        return False
