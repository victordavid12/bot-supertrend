# supertrend_discord_bybit.py
# --------------------------------------------
# Envía alertas a Discord al cierre de cada vela
# cuando el Supertrend cambia de señal (BUY/SELL),
# usando la API PÚBLICA de BYBIT vía CCXT.
# --------------------------------------------

import os
import time
import ccxt
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
SYMBOL = "BTC/USDT"     # Spot: "BTC/USDT"; Perp lineal Bybit: "BTC/USDT:USDT"
INTERVAL = "1m"         # '1m','3m','5m','15m','1h','4h','1d', etc. (Bybit soporta los comunes)
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.3
FETCH_LIMIT = 400       # >= ATR_PERIOD + margen
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
USE_EMBED = True
CHECK_DRIFT_SEC = 1     # margen al despertar tras cierre

# =========================
# TZ utils
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def timeframe_to_seconds(tf: str) -> int:
    t = tf.lower().strip()
    if t.endswith('m'): return int(t[:-1]) * 60
    if t.endswith('h'): return int(t[:-1]) * 3600
    if t.endswith('d'): return int(t[:-1]) * 86400
    raise ValueError(f"Timeframe no soportado: {tf}")

def seconds_to_next_candle_close(now_utc: datetime, tf_seconds: int) -> int:
    epoch = int(now_utc.timestamp())
    remainder = epoch % tf_seconds
    wait = tf_seconds - remainder
    return max(1, wait + CHECK_DRIFT_SEC)

def ts_strings(dt_utc: datetime) -> tuple[str, str]:
    s_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    if ZoneInfo:
        s_mad = dt_utc.astimezone(ZoneInfo("Europe/Madrid")).strftime("%Y-%m-%d %H:%M:%S Europe/Madrid")
    else:
        s_mad = "N/A (instala tzdata)"
    return s_utc, s_mad

# =========================
# Discord
# =========================
def send_discord_message(webhook_url: str, content: str = "", embed: dict | None = None) -> None:
    payload = {"content": content}
    if embed:
        payload = {"embeds": [embed]}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=15)
        if resp.status_code >= 400:
            print(f"[Discord] Error {resp.status_code}: {resp.text}")
        else:
            print("[Discord] Alerta enviada.")
    except Exception as e:
        print(f"[Discord] Excepción al enviar: {e}")

def build_embed(title: str, fields: list[tuple[str, str, bool]], color: int = 0x00ff00) -> dict:
    return {
        "title": title,
        "color": color,
        "fields": [{"name": n, "value": v, "inline": i} for (n, v, i) in fields],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# =========================
# Data
# =========================
def fetch_last_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = FETCH_LIMIT) -> pd.DataFrame:
    """
    Descarga últimas 'limit' velas y quita la última (en formación) para evitar repintado.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) < ATR_PERIOD + 5:
        raise RuntimeError("Datos insuficientes para calcular Supertrend.")
    df = pd.DataFrame(ohlcv, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df.set_index("date", inplace=True)
    if len(df) > 1:
        df = df.iloc[:-1]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

# =========================
# Supertrend
# =========================
def supertrend(df: pd.DataFrame, atr_period: int = ATR_PERIOD, atr_multiplier: float = ATR_MULTIPLIER) -> pd.DataFrame:
    out = df.copy()
    mid = (out["high"] + out["low"]) / 2.0
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], period=atr_period)
    out.dropna(inplace=True)

    out["basicUpperband"] = mid + atr_multiplier * out["atr"]
    out["basicLowerband"] = mid - atr_multiplier * out["atr"]

    upper = [out["basicUpperband"].iloc[0]]
    lower = [out["basicLowerband"].iloc[0]]

    for i in range(1, len(out)):
        bu = out["basicUpperband"].iloc[i]
        bl = out["basicLowerband"].iloc[i]
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]
        prev_close = out["close"].iloc[i-1]

        upper.append(bu if (bu < prev_upper or prev_close > prev_upper) else prev_upper)
        lower.append(bl if (bl > prev_lower or prev_close < prev_lower) else prev_lower)

    out["upperband"] = upper
    out["lowerband"] = lower
    out.drop(columns=["basicUpperband","basicLowerband"], inplace=True)
    return out

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sigs = [0.0]
    ub = out["upperband"]
    lb = out["lowerband"]
    cl = out["close"]
    for i in range(1, len(out)):
        if pd.notna(ub.iloc[i]) and cl.iloc[i] > ub.iloc[i]:
            sigs.append(1.0)
        elif pd.notna(lb.iloc[i]) and cl.iloc[i] < lb.iloc[i]:
            sigs.append(-1.0)
        else:
            sigs.append(sigs[-1])
    out["signal"] = pd.Series(sigs, index=out.index).shift(1).fillna(0.0)
    return out

# =========================
# Loop
# =========================
def run_alerts():
    if not DISCORD_WEBHOOK or DISCORD_WEBHOOK == "PEGAR_AQUI_TU_WEBHOOK":
        raise ValueError("Configura tu webhook de Discord en DISCORD_WEBHOOK o variable de entorno.")

    # ---- Bybit API pública vía CCXT ----
    # Spot:
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}   # para Perp lineal: {"defaultType":"linear"}
    })

    # Si quieres PERP LINEAL (USDT-M), cambia arriba defaultType a "linear"
    # y usa SYMBOL = "BTC/USDT:USDT"

    tf_sec = timeframe_to_seconds(INTERVAL)
    print(f"Inicio de alertas Supertrend (BYBIT) → {SYMBOL} | {INTERVAL} | ATR({ATR_PERIOD}) x {ATR_MULTIPLIER}")

    last_alert_candle_ts = None

    while True:
        try:
            # esperar hasta el cierre exacto según UTC (Bybit también cierra por UTC)
            now_utc = datetime.now(timezone.utc)
            wait = seconds_to_next_candle_close(now_utc, tf_sec)
            print(f"Esperando {wait}s hasta cierre de vela...")
            time.sleep(wait)

            df = fetch_last_ohlcv(exchange, SYMBOL, INTERVAL, limit=FETCH_LIMIT)
            st = supertrend(df, atr_period=ATR_PERIOD, atr_multiplier=ATR_MULTIPLIER)
            st = generate_signals(st)

            last_idx = st.index[-1]
            prev_idx = st.index[-2]
            last_close = float(st.loc[last_idx, "close"])
            sig_now = float(st.loc[last_idx, "signal"])
            sig_prev = float(st.loc[prev_idx, "signal"])

            candle_ts_ms = int(last_idx.timestamp() * 1000)
            s_utc, s_mad = ts_strings(last_idx)

            if (sig_now != sig_prev) and (sig_now in (1.0, -1.0)) and (candle_ts_ms != last_alert_candle_ts):
                direction = "🟢 BUY" if sig_now == 1.0 else "🔴 SELL"
                title = f"{direction} — Supertrend (Bybit)"
                fields = [
                    ("Símbolo", f"`{SYMBOL}`", True),
                    ("Timeframe", f"`{INTERVAL}`", True),
                    ("Precio cierre", f"`{last_close}`", True),
                    ("Cierre UTC", s_utc, False),
                    ("Cierre Europe/Madrid", s_mad, False),
                    ("Parámetros", f"ATR({ATR_PERIOD}) × {ATR_MULTIPLIER}", False),
                ]
                if USE_EMBED:
                    color = 0x00CC66 if sig_now == 1.0 else 0xCC0033
                    embed = build_embed(title, fields, color=color)
                    send_discord_message(DISCORD_WEBHOOK, embed=embed)
                else:
                    msg = (
                        f"**{title}**\n"
                        f"**Símbolo:** {SYMBOL}\n"
                        f"**Timeframe:** {INTERVAL}\n"
                        f"**Precio cierre:** `{last_close}`\n"
                        f"**Vela cerrada:** {s_utc}  |  {s_mad}\n"
                        f"**ATR({ATR_PERIOD}) x {ATR_MULTIPLIER}**"
                    )
                    send_discord_message(DISCORD_WEBHOOK, content=msg)

                last_alert_candle_ts = candle_ts_ms
            else:
                print(f"[{s_utc}] sin cambio de señal (prev={sig_prev}, now={sig_now}).")

        except ccxt.NetworkError as e:
            print(f"[CCXT NetworkError] {e}")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"[CCXT ExchangeError] {e}")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Cerrando por teclado. Hasta luego 👋")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_alerts()

