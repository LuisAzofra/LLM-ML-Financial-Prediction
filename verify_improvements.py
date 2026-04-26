"""
Backtest verificador que corre la pipeline completa (descarga → ML →
`_generate_bot_predictions` → `AutonomousTradingBot.run_backtest`) para los
símbolos indicados, imprimiendo las métricas clave antes/después de las
mejoras. Ejecutar con el venv del proyecto:

    .venv/bin/python verify_improvements.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('LOGLEVEL', 'WARNING')

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from datetime import datetime, timedelta

# Asegurar que los módulos del proyecto se pueden importar al correr desde root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import FinancialDataLoader
from utils.data_utils import DataProcessor
from trading_bot.bot_engine import AutonomousTradingBot, BotConfig
import api as _api  # expone _generate_bot_predictions y run_full_analysis

# Proveedor LLM para los tests:
#   - 'local'      → transformers + Qwen2.5-0.5B (CPU, sin Ollama)
#   - 'ollama'     → Ollama local (requiere ollama serve)
#   - 'huggingface'→ HF Inference API (requiere red + rate limits)
LLM_PROVIDER = os.environ.get('TFG_LLM_PROVIDER', 'local')


SYMBOLS = [
    ('AAPL', 'stock'),
    ('MSFT', 'stock'),
    ('SPY',  'stock'),
    ('BTC-USD', 'crypto'),
]
YEARS = 3


def _run_one(symbol: str, asset_type: str):
    loader = FinancialDataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=YEARS * 365)).strftime('%Y-%m-%d')

    df = loader.download_data(symbol, start_date, end_date, asset_type=asset_type)
    if df is None or df.empty or len(df) < 200:
        print(f"  {symbol}: datos insuficientes ({0 if df is None else len(df)} filas) — saltando")
        return None

    dp = DataProcessor()
    df = dp.clean_data(df)
    df = dp.add_technical_indicators(df)

    # Guardar df para poder calcular B&H en el periodo de test
    _run_one._last_df = df

    preds = _api._generate_bot_predictions(df, asset_type=asset_type)
    if not preds or 'predictions' not in preds:
        print(f"  {symbol}: no se generaron predicciones — saltando")
        return None

    cfg = BotConfig()  # usa los valores mejorados del dataclass
    bot = AutonomousTradingBot(config=cfg)
    result = bot.run_backtest(
        price_df=df,
        predictions=preds['predictions'],
        confidences=preds['confidences'],
        test_start_idx=preds['test_start_idx'],
        symbol=symbol,
    )

    perf = result.get('performance', {}) or {}
    sig_stats = result.get('signal_stats', {}) or {}
    mm_raw = preds.get('model_metrics', {}) or {}
    # La precisión de dirección del ensemble ya la devuelve _generate_bot_predictions.
    direction_accuracy = preds.get('direction_accuracy')

    # Buy & Hold del periodo de test (benchmark pasivo)
    test_start = preds.get('test_start_idx', 0)
    try:
        close = df['Close'].iloc[test_start:]
        buy_hold_ret = float(close.iloc[-1] / close.iloc[0] - 1) * 100
    except Exception:
        buy_hold_ret = None

    return {
        'symbol': symbol,
        'total_return_pct': perf.get('total_return_pct'),
        'buy_hold_pct':     buy_hold_ret,
        'n_trades':         perf.get('total_trades'),
        'win_rate_pct':     perf.get('win_rate'),
        'profit_factor':    perf.get('profit_factor'),
        'sharpe':           perf.get('sharpe_ratio'),
        'max_dd_pct':       perf.get('max_drawdown_pct'),
        'avg_win_pct':      perf.get('avg_win_pct'),
        'avg_loss_pct':     perf.get('avg_loss_pct'),
        'n_signals_over_th': sig_stats.get('n_signals_above_threshold') or sig_stats.get('n_over_threshold'),
        'threshold':        sig_stats.get('threshold'),
        'direction_acc':    direction_accuracy,
        'n_test':           preds.get('n_test'),
    }


def main():
    print("=" * 78)
    print("VERIFICACIÓN DE MEJORAS — backtest completo (ML + bot)")
    print("=" * 78)

    results = []
    for sym, atype in SYMBOLS:
        print(f"\n>>> {sym} ({atype})")
        try:
            r = _run_one(sym, atype)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        if not r:
            continue
        results.append(r)
        def _f(x, fmt='{:.2f}'):
            return fmt.format(x) if x is not None else 'n/a'
        bot = r['total_return_pct'] or 0.0
        bh  = r['buy_hold_pct'] or 0.0
        edge = bot - bh
        print(f"  bot:      {_f(r['total_return_pct'],  '{:+.2f}')}%"
              f"   B&H: {_f(r['buy_hold_pct'], '{:+.2f}')}%"
              f"   alpha: {edge:+.2f}%")
        print(f"  trades:   {r['n_trades']}"
              f"   win_rate: {_f(r['win_rate_pct'], '{:.1f}')}%"
              f"   sharpe: {_f(r['sharpe'])}"
              f"   max_dd: {_f(r['max_dd_pct'], '{:.2f}')}%")
        print(f"  dir_acc:  {_f(r['direction_acc'], '{:.3f}')}"
              f"   n_test: {r['n_test']}")

    print("\n" + "=" * 78)
    print("RESUMEN AGREGADO")
    print("=" * 78)
    if not results:
        print("No hay resultados.")
        return
    def _avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else float('nan')
    avg_ret = _avg('total_return_pct')
    avg_wr  = _avg('win_rate_pct')
    avg_sh  = _avg('sharpe')
    avg_pf  = _avg('profit_factor')
    n_pos   = sum(1 for r in results if (r['total_return_pct'] or 0) > 0)
    avg_bh = _avg('buy_hold_pct')
    n_no_trade = sum(1 for r in results if (r['n_trades'] or 0) == 0)
    print(f"Símbolos evaluados: {len(results)}  |  con retorno positivo: {n_pos}")
    print(f"Símbolos con gate de accuracy activado (sin trades): {n_no_trade}")
    print(f"Retorno bot medio:    {avg_ret:+.2f}%")
    print(f"Retorno B&H medio:    {avg_bh:+.2f}%")
    print(f"Alpha medio:          {avg_ret - avg_bh:+.2f}%")
    print(f"Sharpe medio:         {avg_sh:.2f}")

    # ── Fase 2: análisis híbrido ML+LLM con LLM local ──────────────────────
    # Ejercita la ruta completa de agentes (técnico, sentimiento, riesgo,
    # manager) y el hybrid score. Más lento (~60-90s/símbolo) por la
    # inferencia CPU; por eso lo hacemos sólo sobre los símbolos con edge.
    print("\n" + "=" * 78)
    print(f"FASE 2 — ANÁLISIS HÍBRIDO ML+LLM (provider={LLM_PROVIDER})")
    print("=" * 78)
    traded_syms = [r for r in results if (r['n_trades'] or 0) > 0][:2]
    if not traded_syms:
        traded_syms = results[:2]
    for r in traded_syms:
        sym = r['symbol']
        print(f"\n>>> {sym}")
        try:
            out = _api.run_full_analysis(
                symbol=sym, timeframe='1y',
                asset_type='crypto' if sym.endswith('-USD') else 'stock',
                use_llm=True, llm_provider=LLM_PROVIDER,
            )
            summ = (out or {}).get('summary', {}) or {}
            hy   = (out or {}).get('hybrid',  {}) or {}
            agt  = (out or {}).get('agents',  {}) or {}
            print(f"  ML dir:         {summ.get('ml_direction','?')}")
            print(f"  Agentes (LLM):  {summ.get('agent_decision','?')}"
                  f"   conf={agt.get('final_confidence','?')}")
            print(f"  Híbrido:        {summ.get('recommendation','?')}"
                  f"   score={hy.get('score','?')}"
                  f"   conf={hy.get('confidence','?')}")
            if agt.get('final_reasoning'):
                r_txt = str(agt['final_reasoning'])[:220]
                print(f"  Razonamiento:   {r_txt}")
        except Exception as e:
            print(f"  FALLO: {e}")


if __name__ == '__main__':
    main()
