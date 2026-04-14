"""
screening.py — Multi-market fundamental screening
B3 · S&P 500 · FTSE 100 · DAX · Chinese ADRs (NYSE)
"""

import csv
import json
import math
import random
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RETRIES         = 6
RATE_LIMIT_WAIT     = 60
RATE_LIMIT_BACKOFF  = 2.0     # exponential backoff multiplier
SLEEP_MIN           = 3.0
SLEEP_MAX           = 7.0
BATCH_SIZE          = 15      # tickers per batch before longer pause
BATCH_SLEEP_MIN     = 20
BATCH_SLEEP_MAX     = 40
MIN_VOL_USD        = 1_000_000
B3_MIN_LIQ         = 1_000_000  # Liq.2meses mínima em BRL (R$1M/dia)
MIN_ROE            = 0.10
MAX_DEBT_EBITDA    = 3.0
EBITDA_DECAY_LIMIT = 0.90     # margin may not drop more than 10% YoY (L4)
OUTPUT_CSV         = "screening_results.csv"
OUTPUT_HTML        = "dashboard.html"
CHECKPOINT_FILE    = "screening_checkpoint.json"
CHECKPOINT_EVERY   = 50       # save checkpoint every N tickers

# Smoke test usa apenas tickers do exterior (B3 não precisa de yfinance)
SMOKE_TEST_TICKERS = [
    ("AAPL",  "EUA"),
    ("MSFT",  "EUA"),
    ("SHEL.L","Europa"),
    ("BABA",  "China"),
]


# ─── Custom HTTP session ──────────────────────────────────────────────────────

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    return session


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]


def rotate_session(session: requests.Session) -> requests.Session:
    """Rotate User-Agent and reset cookies to reduce fingerprinting."""
    ua = random.choice(_USER_AGENTS)
    session.cookies.clear()
    session.headers.update({"User-Agent": ua})
    return session


# ─── B3 via fundamentus (sem yfinance) ───────────────────────────────────────

# Mapeamento de colunas do fundamentus para o schema padrão
_FUND_COLS: dict[str, list[str]] = {
    "pl":           ["P/L"],
    "pvp":          ["P/VP"],
    "ev_ebitda":    ["EV/EBITDA"],
    "dy":           ["DY"],
    "roe":          ["ROE"],
    "ebitda_margin":["Mrg Ebit", "Mrg. Ebit"],
    "liq2m":        ["Liq.2meses", "Liq. 2meses"],
    "patrim":       ["Patrim. Liq", "Patrim. Liq."],
    "rec_growth":   ["Cresc. Rec.5a", "Cresc.Rec.5a"],
    "receita":      ["Receita Liq.", "Receita Liq"],
    "lucro":        ["Lucro Liq.", "Lucro Liq"],
}


def _fund_val(row: pd.Series, keys: list[str]):
    for k in keys:
        if k in row.index:
            try:
                v = row[k]
                if pd.isna(v):
                    return None
                f = float(v)
                return None if (math.isinf(f)) else f
            except Exception:
                return None
    return None


def safe_float(val):
    try:
        return None if pd.isna(val) else float(val)
    except Exception:
        return None


def get_b3_records() -> list[dict]:
    """Carrega todos os dados da B3 direto do fundamentus, sem chamar yfinance."""
    try:
        import fundamentus
    except ImportError:
        print("  B3: fundamentus não instalado (pip install fundamentus). Pulando.")
        return []

    try:
        df = fundamentus.get_resultado()
        print("COLUNAS FUNDAMENTUS:", list(df.columns))
        primeiro_ticker = df.iloc[0]
        print("EXEMPLO LINHA B3:", {
            "pl": primeiro_ticker.get("pl"),
            "pvp": primeiro_ticker.get("pvp"),
            "dy": primeiro_ticker.get("dy"),
            "roe": primeiro_ticker.get("roe"),
            "liq2m": primeiro_ticker.get("liq2m"),
            "c5y": primeiro_ticker.get("c5y"),
        })
    except Exception as e:
        print(f"  B3: erro ao chamar fundamentus.get_resultado() ({e}). Pulando.")
        return []

    records: list[dict] = []
    for ticker, row in df.iterrows():
        symbol = f"{ticker}.SA"

        pl            = safe_float(row["pl"])      if "pl"      in row.index else None
        pvp           = safe_float(row["pvp"])     if "pvp"     in row.index else None
        ev_ebitda     = safe_float(row["evebitda"])if "evebitda"in row.index else None
        dy            = safe_float(row["dy"])      if "dy"      in row.index else None
        roe           = safe_float(row["roe"])     if "roe"     in row.index else None
        ebitda_margin = safe_float(row["mrgebit"]) if "mrgebit" in row.index else None
        patrim        = safe_float(row["patrliq"]) if "patrliq" in row.index else None
        rec_growth    = safe_float(row["c5y"])     if "c5y"     in row.index else None
        liq2m         = safe_float(row["liq2m"])   if "liq2m"   in row.index else None
        receita       = None
        lucro         = None

        # fundamentus retorna frações (0.065) mas às vezes percentual (6.5) — normaliza
        if dy            is not None and abs(dy)            > 1: dy            /= 100
        if roe           is not None and abs(roe)           > 1: roe           /= 100
        if ebitda_margin is not None and abs(ebitda_margin) > 2: ebitda_margin /= 100
        if rec_growth    is not None and abs(rec_growth)    > 2: rec_growth    /= 100

        if len(records) == 0:  # só no primeiro ticker
            print("DEBUG liq2m raw:", row.get("liq2m"), type(row.get("liq2m")))
            print("DEBUG _b3_liq2m que vai entrar:", liq2m)

        records.append({
            "symbol":        symbol,
            "region":        "B3",
            "sector":        "B3",  # fundamentus não expõe setor em get_resultado
            "currency":      "BRL",
            "price":         None,
            "mkt_cap":       None,
            "avg_vol_usd":   liq2m,   # Liq.2meses em BRL; usado como proxy de liquidez
            "pl":            pl,
            "pvp":           pvp,
            "ev_ebitda":     ev_ebitda,
            "dy":            dy,
            "roe":           roe,
            "ebitda_margin": ebitda_margin,
            "debt_ebitda":   None,    # Div.Br/Patrim ≠ Dív/EBITDA; omitido
            "total_equity":  patrim,
            "revenue_ttm":   receita,
            # séries temporais não disponíveis no fundamentus
            "rev_series":    [],
            "ni_series":     [],
            "fcf_series":    [],
            "ebitda_series": [],
            "hist_avg_pe":   None,
            "hist_avg_ev":   None,
            # campos auxiliares para filtro L1 no B3
            "_b3_liq2m":      liq2m,
            "_b3_rec_growth": rec_growth,
            "_b3_lucro":      lucro,
            "pass_l1": False, "pass_l2": False, "pass_l3": False, "pass_l4": False,
        })

    print(f"  B3: {len(records)} tickers carregados via fundamentus (sem yfinance)")
    return records


# ─── Ticker collection (exterior only) ───────────────────────────────────────


def _read_html(url: str) -> list[pd.DataFrame]:
    headers = {"User-Agent": random.choice(_USER_AGENTS)}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return pd.read_html(resp.text)
    except Exception:
        return []


def get_sp500_tickers() -> list[str]:
    try:
        import pandas as pd
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = [str(t).replace(".", "-").strip() for t in df["Symbol"].dropna()]
        print(f"  S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  S&P 500: erro ({e})")
        return []


def get_ftse100_tickers() -> list[str]:
    # Lista hardcoded dos 100 principais do FTSE (sufixo .L)
    tickers = [
        "AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","REL.L",
        "NG.L","VOD.L","BATS.L","AAL.L","LLOY.L","BARC.L","NWG.L","GLEN.L",
        "CPG.L","PRU.L","RR.L","IMB.L","DGE.L","EXPN.L","LGEN.L","ABF.L",
        "ADM.L","AHT.L","ANTO.L","AUTO.L","AV.L","AVS.L","BA.L","BNZL.L",
        "BKG.L","BLND.L","BT-A.L","CCH.L","CNA.L","COB.L","CRH.L","DCC.L",
        "DGE.L","EDV.L","ENT.L","FERG.L","FLTR.L","FRES.L","HIK.L","HLMA.L",
        "HLN.L","HWDN.L","IAG.L","IHG.L","III.L","INF.L","ITRK.L","JD.L",
        "KGF.L","LAND.L","LSEG.L","MKS.L","MNDI.L","MNG.L","MRO.L","MTO.L",
        "NXT.L","OCDO.L","PHNX.L","POLYP.L","PSH.L","PSN.L","PSON.L","RKT.L",
        "RMV.L","RS1.L","RSA.L","SBRY.L","SDR.L","SGE.L","SGRO.L","SKG.L",
        "SMDS.L","SMIN.L","SMT.L","SN.L","SPX.L","SSE.L","STAN.L","SVT.L",
        "TSCO.L","TW.L","ULVR.L","UU.L","VTY.L","WEIR.L","WPP.L","WTB.L",
        "XRX.L","ZAL.L"
    ]
    print(f"  FTSE 100: {len(tickers)} tickers (hardcoded)")
    return tickers


def get_dax_tickers() -> list[str]:
    tickers = [
        "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE","BMW.DE",
        "BNR.DE","CBK.DE","CON.DE","1COV.DE","DTG.DE","DBK.DE","DB1.DE",
        "DTE.DE","DPWA.DE","EON.DE","FRE.DE","HNR1.DE","HEI.DE","HEN3.DE",
        "IFX.DE","MBG.DE","MRK.DE","MTX.DE","MUV2.DE","PAH3.DE","P911.DE",
        "QGEN.DE","RHM.DE","RWE.DE","SAP.DE","SRT3.DE","SIE.DE","SHL.DE",
        "SY1.DE","SYM.DE","VOW3.DE","VNA.DE","ZAL.DE"
    ]
    print(f"  DAX: {len(tickers)} tickers (hardcoded)")
    return tickers


def get_chinese_adr_tickers() -> list[str]:
    tickers = [
        "BABA","JD","PDD","BIDU","NIO","XPEV","LI","TME","NTES","ZM",
        "VIPS","TAL","EDU","RLX","DIDI","FUTU","LKNCY","IQ","BILI","QFIN"
    ]
    print(f"  Chinese ADRs: {len(tickers)} tickers (hardcoded)")
    return tickers


def build_exterior_ticker_list() -> list[tuple[str, str]]:
    """Returns deduplicated [(symbol, region)] for S&P500/FTSE100/DAX/ADRs (no B3)."""
    sources = [
        (get_sp500_tickers(),       "EUA"),
        (get_ftse100_tickers(),     "Europa"),
        (get_dax_tickers(),         "Europa"),
        (get_chinese_adr_tickers(), "China"),
    ]
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tickers, region in sources:
        for sym in tickers:
            if sym not in seen:
                result.append((sym, region))
                seen.add(sym)
    return result


# ─── yfinance helpers with retry ──────────────────────────────────────────────

def _retry(fn, symbol: str):
    """Call fn(); retry with exponential backoff on rate-limit errors (yfinance 1.x)."""
    wait = RATE_LIMIT_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "too many" in err:
                if attempt < MAX_RETRIES:
                    print(
                        f"    Rate limit ({symbol}, tentativa {attempt}/{MAX_RETRIES}). "
                        f"Aguardando {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    wait = min(wait * RATE_LIMIT_BACKOFF, 300)  # cap at 5 min
                else:
                    print(f"    Rate limit ({symbol}): máximo de tentativas. Pulando.")
                    return None
            else:
                return None
    return None


def fetch_info(t: yf.Ticker, symbol: str) -> dict:
    result = _retry(lambda: t.info, symbol)
    return result if isinstance(result, dict) else {}


def fetch_income_stmt(t: yf.Ticker, symbol: str) -> pd.DataFrame:
    result = _retry(lambda: t.income_stmt, symbol)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_cashflow(t: yf.Ticker, symbol: str) -> pd.DataFrame:
    result = _retry(lambda: t.cashflow, symbol)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_balance_sheet(t: yf.Ticker, symbol: str) -> pd.DataFrame:
    result = _retry(lambda: t.balance_sheet, symbol)
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_history(t: yf.Ticker, symbol: str) -> pd.DataFrame:
    result = _retry(lambda: t.history(period="4y"), symbol)
    if not isinstance(result, pd.DataFrame) or result.empty:
        return pd.DataFrame()
    result = result.copy()
    result.index = _strip_tz(result.index)
    return result


# ─── Timezone normalisation ───────────────────────────────────────────────────

def _strip_tz(index: pd.Index) -> pd.Index:
    """Convert any DatetimeIndex to tz-naive UTC-equivalent to allow comparisons."""
    if not isinstance(index, pd.DatetimeIndex):
        return index
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _strip_tz_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with both row and column indexes stripped of timezone."""
    out = df.copy()
    out.index   = _strip_tz(out.index)
    out.columns = _strip_tz(out.columns)
    return out


# ─── Financial series extractors ─────────────────────────────────────────────

def _series(df: pd.DataFrame, *labels) -> pd.Series:
    """Return first matching row, sorted oldest→newest (always tz-naive)."""
    for label in labels:
        if label in df.index:
            s = df.loc[label].dropna()
            s.index = _strip_tz(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)


def extract_revenue(inc: pd.DataFrame) -> list[float]:
    s = _series(inc, "Total Revenue", "Revenue", "Revenues")
    return list(s.values)


def extract_net_income(inc: pd.DataFrame) -> list[float]:
    s = _series(
        inc,
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operations",
    )
    return list(s.values)


def extract_ebitda(inc: pd.DataFrame) -> list[float]:
    s = _series(inc, "EBITDA", "Normalized EBITDA")
    if not s.empty:
        return list(s.values)
    # Approximate: EBIT + D&A
    ebit = _series(inc, "EBIT", "Operating Income")
    da   = _series(inc, "Reconciled Depreciation", "Depreciation And Amortization")
    if not ebit.empty and not da.empty:
        combined = ebit.add(da, fill_value=0).dropna().sort_index()
        return list(combined.values)
    return list(ebit.values) if not ebit.empty else []


def extract_fcf(cf: pd.DataFrame) -> list[float]:
    ocf = _series(cf, "Operating Cash Flow", "Cash From Operations")
    if ocf.empty:
        return []
    capex_row = next(
        (r for r in ("Capital Expenditure", "Capital Expenditures", "Purchase Of PPE")
         if r in cf.index),
        None,
    )
    if capex_row:
        capex = _series(cf, capex_row)
        fcf = ocf.add(capex, fill_value=0).dropna().sort_index()  # capex is negative in yf
    else:
        fcf = ocf
    return list(fcf.values)


def compute_hist_pe(
    inc: pd.DataFrame, hist: pd.DataFrame, shares: float
) -> float | None:
    """3-year average trailing P/E from annual NI and year-end prices."""
    if inc.empty or hist.empty or not shares:
        return None
    # hist index already stripped by fetch_history; strip inc columns too
    inc = _strip_tz_df(inc)
    ni_s = _series(inc, "Net Income", "Net Income Common Stockholders")
    if ni_s.empty:
        return None
    pes = []
    for date, ni in ni_s.items():
        if ni <= 0:
            continue
        eps = ni / shares
        # date and hist.index are both tz-naive here
        date_naive = pd.Timestamp(date).tz_localize(None) if getattr(date, "tzinfo", None) else pd.Timestamp(date)
        idx = hist.index.searchsorted(date_naive)
        idx = min(idx, len(hist) - 1)
        price = float(hist.iloc[idx]["Close"])
        if eps > 0 and price > 0:
            pes.append(price / eps)
    return float(np.mean(pes)) if pes else None


def compute_hist_ev_ebitda(
    inc: pd.DataFrame, bs: pd.DataFrame, hist: pd.DataFrame, shares: float
) -> float | None:
    """3-year average EV/EBITDA from annual data."""
    if inc.empty or hist.empty or not shares:
        return None
    inc = _strip_tz_df(inc)
    bs  = _strip_tz_df(bs)
    ebitda_s = _series(inc, "EBITDA", "Normalized EBITDA")
    if ebitda_s.empty:
        return None
    debt_row = next(
        (r for r in ("Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt")
         if r in bs.index),
        None,
    )
    ratios = []
    for date, ebitda in ebitda_s.items():
        if ebitda <= 0:
            continue
        date_naive = pd.Timestamp(date).tz_localize(None) if getattr(date, "tzinfo", None) else pd.Timestamp(date)
        idx = hist.index.searchsorted(date_naive)
        idx = min(idx, len(hist) - 1)
        price = float(hist.iloc[idx]["Close"])
        mktcap = price * shares
        debt = 0.0
        if debt_row and not bs.empty and date_naive in bs.columns:
            try:
                debt = float(bs.loc[debt_row, date_naive])
            except Exception:
                pass
        ev = mktcap + max(debt, 0)
        ratios.append(ev / ebitda)
    return float(np.mean(ratios)) if ratios else None


# ─── Safe value helper ────────────────────────────────────────────────────────

def _safe(d: dict, key: str):
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


# ─── Build single ticker record ───────────────────────────────────────────────

def process_ticker(symbol: str, region: str, session: requests.Session | None = None) -> dict | None:
    t = yf.Ticker(symbol)
    yf.set_tz_cache_location("cache_tz")

    info = fetch_info(t, symbol)
    if not info:
        return None

    inc  = fetch_income_stmt(t, symbol)
    cf   = fetch_cashflow(t, symbol)
    bs   = fetch_balance_sheet(t, symbol)
    hist = fetch_history(t, symbol)

    shares = _safe(info, "sharesOutstanding") or _safe(info, "impliedSharesOutstanding")
    price  = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")

    # ── Spot metrics ──────────────────────────────────────────────────────────
    pl         = _safe(info, "trailingPE")
    pvp        = _safe(info, "priceToBook")
    ev_ebitda  = _safe(info, "enterpriseToEbitda")
    dy_raw     = _safe(info, "dividendYield")
    roe        = _safe(info, "returnOnEquity")
    avg_vol    = _safe(info, "averageVolume")
    mkt_cap    = _safe(info, "marketCap")
    total_debt = _safe(info, "totalDebt")
    ebitda_ttm = _safe(info, "ebitda")
    rev_ttm    = _safe(info, "totalRevenue")
    sector     = info.get("sector") or "Unknown"
    currency   = info.get("currency") or "USD"

    # DY fix: yfinance sometimes returns value >1 for BR stocks
    dy = None
    if dy_raw is not None:
        dy = dy_raw / 100 if dy_raw > 1 else dy_raw

    # Average daily volume in USD
    avg_vol_usd = (avg_vol * price) if avg_vol and price else None

    # EBITDA margin
    ebitda_margin = (ebitda_ttm / rev_ttm) if ebitda_ttm and rev_ttm and rev_ttm > 0 else None

    # Debt / EBITDA
    debt_ebitda = (total_debt / ebitda_ttm) if total_debt and ebitda_ttm and ebitda_ttm > 0 else None

    # Financial series (oldest → newest)
    rev_series    = extract_revenue(inc)
    ni_series     = extract_net_income(inc)
    fcf_series    = extract_fcf(cf)
    ebitda_series = extract_ebitda(inc)

    # Historical average multiples (Layer 2)
    hist_avg_pe = compute_hist_pe(inc, hist, shares) if shares else None
    hist_avg_ev = compute_hist_ev_ebitda(inc, bs, hist, shares) if shares else None

    # Total equity (for display)
    equity_ps = _safe(info, "bookValue")
    total_equity = (equity_ps * shares) if equity_ps and shares else None

    return {
        "symbol":        symbol,
        "region":        region,
        "sector":        sector,
        "currency":      currency,
        "price":         price,
        "mkt_cap":       mkt_cap,
        "avg_vol_usd":   avg_vol_usd,
        "pl":            pl,
        "pvp":           pvp,
        "ev_ebitda":     ev_ebitda,
        "dy":            dy,
        "roe":           roe,
        "ebitda_margin": ebitda_margin,
        "debt_ebitda":   debt_ebitda,
        "total_equity":  total_equity,
        "revenue_ttm":   rev_ttm,
        "rev_series":    rev_series,
        "ni_series":     ni_series,
        "fcf_series":    fcf_series,
        "ebitda_series": ebitda_series,
        "hist_avg_pe":   hist_avg_pe,
        "hist_avg_ev":   hist_avg_ev,
        # filter results (populated in main)
        "pass_l1": False, "pass_l2": False, "pass_l3": False, "pass_l4": False,
    }


# ─── Filters ──────────────────────────────────────────────────────────────────

def filter_l1(r: dict) -> bool:
    """Liquidity, positive PL, revenue growing, no sustained losses."""
    if r["region"] == "B3":
        liq = r.get("_b3_liq2m")
        if not liq or liq < 200_000:
            return False
        return True

    if not r["pl"] or r["pl"] <= 0:
        return False

    # Exterior: usa séries temporais do yfinance
    if not r["avg_vol_usd"] or r["avg_vol_usd"] < MIN_VOL_USD:
        return False
    rev = r["rev_series"]
    if len(rev) < 3 or not (rev[-1] > rev[-2] > rev[-3]):
        return False
    ni = r["ni_series"]
    if len(ni) >= 3 and all(v < 0 for v in ni[-3:]):
        return False
    return True


def filter_l2(r: dict) -> bool:
    """P/L and EV/EBITDA both below their own 3-year historical average."""
    pe_ok = True
    ev_ok = True
    if r["pl"] and r["hist_avg_pe"] and r["hist_avg_pe"] > 0:
        pe_ok = r["pl"] < r["hist_avg_pe"]
    if r["ev_ebitda"] and r["hist_avg_ev"] and r["hist_avg_ev"] > 0:
        ev_ok = r["ev_ebitda"] < r["hist_avg_ev"]
    return pe_ok and ev_ok


def apply_l3(records: list[dict]) -> None:
    """Tag records in cheapest P/L and EV/EBITDA quartile within their sector."""
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["pass_l2"]:
            by_sector[r["sector"]].append(r)

    for group in by_sector.values():
        pls = [r["pl"] for r in group if r["pl"] and r["pl"] > 0]
        evs = [r["ev_ebitda"] for r in group if r["ev_ebitda"] and r["ev_ebitda"] > 0]

        pl_q1 = float(np.percentile(pls, 25)) if len(pls) >= 4 else (max(pls) if pls else None)
        ev_q1 = float(np.percentile(evs, 25)) if len(evs) >= 4 else (max(evs) if evs else None)

        for r in group:
            pl_pass = (r["pl"] is not None and pl_q1 is not None and r["pl"] <= pl_q1) or pl_q1 is None
            ev_pass = (r["ev_ebitda"] is not None and ev_q1 is not None and r["ev_ebitda"] <= ev_q1) or ev_q1 is None
            r["pass_l3"] = pl_pass and ev_pass


def filter_l4(r: dict) -> bool:
    """ROE > 10%, stable/growing EBITDA margin, Debt/EBITDA < 3x, FCF+ 2 years."""
    if not r["roe"] or r["roe"] < MIN_ROE:
        return False
    if r["debt_ebitda"] is not None and r["debt_ebitda"] > MAX_DEBT_EBITDA:
        return False
    fcf = r["fcf_series"]
    if len(fcf) >= 2 and not all(v > 0 for v in fcf[-2:]):
        return False
    ebitda = r["ebitda_series"]
    rev    = r["rev_series"]
    if len(ebitda) >= 2 and len(rev) >= 2:
        margins = [
            e / v
            for e, v in zip(ebitda[-2:], rev[-2:])
            if v > 0
        ]
        if len(margins) == 2 and margins[1] < margins[0] * EBITDA_DECAY_LIMIT:
            return False
    return True


# ─── CSV export ───────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "symbol", "region", "sector", "currency", "price", "mkt_cap",
    "avg_vol_usd", "pl", "pvp", "ev_ebitda", "dy", "roe",
    "ebitda_margin", "debt_ebitda", "total_equity", "revenue_ttm",
    "hist_avg_pe", "hist_avg_ev",
    "pass_l1", "pass_l2", "pass_l3", "pass_l4",
]


def save_csv(records: list[dict]) -> None:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nCSV salvo: {OUTPUT_CSV}")


# ─── HTML dashboard ───────────────────────────────────────────────────────────

def _fmt(val, spec=".2f", suffix="", na="N/A") -> str:
    if val is None:
        return na
    try:
        return f"{val:{spec}}{suffix}"
    except Exception:
        return na


def _badge(r: dict) -> str:
    count = sum([r["pass_l1"], r["pass_l2"], r["pass_l3"], r["pass_l4"]])
    colors = {0: "#555", 1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}
    c = colors.get(count, "#555")
    label = f"L{count}" if count else "—"
    return f'<span class="badge" style="background:{c}">{label}</span>'


def _row_cls(r: dict) -> str:
    for key, cls in (("pass_l4", "l4"), ("pass_l3", "l3"), ("pass_l2", "l2"), ("pass_l1", "l1")):
        if r[key]:
            return cls
    return ""


def generate_html(records: list[dict]) -> None:
    sorted_records = sorted(
        records,
        key=lambda x: (
            not x["pass_l4"], not x["pass_l3"], not x["pass_l2"], not x["pass_l1"]
        ),
    )

    rows = ""
    for r in sorted_records:
        dy_s  = _fmt(r["dy"]  * 100 if r["dy"]  is not None else None, ".2f", "%")
        roe_s = _fmt(r["roe"] * 100 if r["roe"] is not None else None, ".1f", "%")
        em_s  = _fmt(r["ebitda_margin"] * 100 if r["ebitda_margin"] is not None else None, ".1f", "%")
        rows += (
            f'<tr class="{_row_cls(r)}">'
            f"<td>{r['symbol']}</td>"
            f'<td><span class="region r-{r["region"].lower()}">{r["region"]}</span></td>'
            f"<td>{r['sector']}</td>"
            f"<td>{_fmt(r['pl'],   '.2f', 'x')}</td>"
            f"<td>{_fmt(r['pvp'],  '.2f', 'x')}</td>"
            f"<td>{_fmt(r['ev_ebitda'], '.2f', 'x')}</td>"
            f"<td>{dy_s}</td>"
            f"<td>{roe_s}</td>"
            f"<td>{em_s}</td>"
            f"<td>{_fmt(r['debt_ebitda'], '.2f', 'x')}</td>"
            f"<td>{_badge(r)}</td>"
            "</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Screening — {datetime.now():%Y-%m-%d}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#0d1117;color:#c9d1d9;padding:24px;font-size:14px}}
h1{{color:#e6edf3;margin-bottom:16px;font-size:1.3rem}}
.controls{{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}}
input,select{{background:#161b22;border:1px solid #30363d;color:#c9d1d9;
              padding:6px 10px;border-radius:6px;font-size:13px}}
input{{width:240px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#161b22;color:#8b949e;padding:9px 8px;text-align:left;
    border-bottom:1px solid #21262d;cursor:pointer;white-space:nowrap;
    user-select:none}}
th:hover{{color:#e6edf3}}
td{{padding:7px 8px;border-bottom:1px solid #161b22}}
tr:hover td{{background:#161b22}}
tr.l4 td{{background:rgba(46,204,113,.09)}}
tr.l3 td{{background:rgba(241,196,15,.07)}}
tr.l2 td{{background:rgba(230,126,34,.06)}}
tr.l1 td{{background:rgba(231,76,60,.05)}}
.region{{padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700}}
.r-b3    {{background:#0d2137;color:#58a6ff}}
.r-eua   {{background:#0d2b0d;color:#3fb950}}
.r-europa{{background:#1f0d2b;color:#bc8cff}}
.r-china {{background:#2b0d0d;color:#f78166}}
.badge{{padding:2px 9px;border-radius:10px;font-size:11px;font-weight:700;color:#fff}}
.footer{{margin-top:14px;font-size:11px;color:#484f58}}
</style>
</head>
<body>
<h1>Screening Fundamentalista — {datetime.now():%d/%m/%Y %H:%M}</h1>
<div class="controls">
  <input id="q" placeholder="Filtrar ticker, setor..." oninput="render()">
  <select id="reg" onchange="render()">
    <option value="">Todas as regiões</option>
    <option>B3</option><option>EUA</option>
    <option>Europa</option><option>China</option>
  </select>
  <select id="lay" onchange="render()">
    <option value="">Todas as camadas</option>
    <option value="4">L4 — todas camadas</option>
    <option value="3">L3+</option>
    <option value="2">L2+</option>
    <option value="1">L1+</option>
  </select>
</div>
<table>
<thead><tr>
  <th onclick="sort(0)">Ticker ↕</th>
  <th onclick="sort(1)">Região ↕</th>
  <th onclick="sort(2)">Setor ↕</th>
  <th onclick="sort(3)">P/L ↕</th>
  <th onclick="sort(4)">P/VP ↕</th>
  <th onclick="sort(5)">EV/EBITDA ↕</th>
  <th onclick="sort(6)">DY ↕</th>
  <th onclick="sort(7)">ROE ↕</th>
  <th onclick="sort(8)">Mg.EBITDA ↕</th>
  <th onclick="sort(9)">Dív/EBITDA ↕</th>
  <th>Camada</th>
</tr></thead>
<tbody id="tb">{rows}</tbody>
</table>
<div class="footer">
  Gerado em {datetime.now().isoformat()} &nbsp;·&nbsp; {len(records)} tickers processados
</div>
<script>
const allRows = Array.from(document.querySelectorAll('#tb tr'));
let dir = {{}};
function layerOf(row) {{
  const cls = row.className;
  if(cls==='l4') return 4; if(cls==='l3') return 3;
  if(cls==='l2') return 2; if(cls==='l1') return 1;
  return 0;
}}
function render() {{
  const q   = document.getElementById('q').value.toLowerCase();
  const reg = document.getElementById('reg').value;
  const lay = parseInt(document.getElementById('lay').value) || 0;
  allRows.forEach(row => {{
    const text   = row.textContent.toLowerCase();
    const rowReg = row.querySelector('.region')?.textContent.trim() || '';
    const rowLay = layerOf(row);
    row.style.display = (
      (!q   || text.includes(q)) &&
      (!reg || rowReg === reg)   &&
      (!lay || rowLay >= lay)
    ) ? '' : 'none';
  }});
}}
function sort(col) {{
  const tb = document.getElementById('tb');
  const rows = allRows.filter(r => r.style.display !== 'none');
  const d = (dir[col] = -(dir[col]||1));
  rows.sort((a,b) => {{
    const av = a.cells[col].textContent.replace(/[x%]/g,'').trim();
    const bv = b.cells[col].textContent.replace(/[x%]/g,'').trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn)) return (an-bn)*d;
    return av.localeCompare(bv)*d;
  }});
  rows.forEach(r => tb.appendChild(r));
}}
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard salvo: {OUTPUT_HTML}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    total = len(records)
    l1 = sum(1 for r in records if r["pass_l1"])
    l2 = sum(1 for r in records if r["pass_l2"])
    l3 = sum(1 for r in records if r["pass_l3"])
    l4 = sum(1 for r in records if r["pass_l4"])

    sep = "=" * 54
    print(f"\n{sep}")
    print("  RESUMO DO SCREENING")
    print(sep)
    print(f"  Total processado  : {total}")
    print(f"  Camada 1 (liquidez): {l1:>5}  ({l1/total*100:.1f}%)" if total else "")
    print(f"  Camada 2 (histórico): {l2:>4}  ({l2/total*100:.1f}%)" if total else "")
    print(f"  Camada 3 (setor)  : {l3:>5}  ({l3/total*100:.1f}%)" if total else "")
    print(f"  Camada 4 (qualidade): {l4:>4}  ({l4/total*100:.1f}%)" if total else "")

    for region in ("B3", "EUA", "Europa", "China"):
        candidates = [r for r in records if r["region"] == region]
        # Best available layer
        top = [r for r in candidates if r["pass_l4"]]
        if not top:
            top = [r for r in candidates if r["pass_l3"]]
        if not top:
            top = [r for r in candidates if r["pass_l2"]]
        if not top:
            top = [r for r in candidates if r["pass_l1"]]
        if not top:
            continue
        top = sorted(top, key=lambda x: (-(x["roe"] or 0), (x["debt_ebitda"] or 999)))[:10]
        print(f"\n  Top {region} (até 10):")
        for i, r in enumerate(top, 1):
            layer  = "L4" if r["pass_l4"] else "L3" if r["pass_l3"] else "L2" if r["pass_l2"] else "L1"
            roe_s  = f"ROE {r['roe']*100:.1f}%" if r["roe"] else "ROE N/A"
            pl_s   = f"P/L {r['pl']:.1f}x"     if r["pl"]  else "P/L N/A"
            print(f"    {i:2}. {r['symbol']:<14} [{layer}]  {roe_s:<12}  {pl_s}")
    print(f"\n{sep}")


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[list[dict], set[str]]:
    """Returns (records_so_far, processed_symbols)."""
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        processed = set(data.get("processed", []))
        print(f"Checkpoint carregado: {len(records)} registros, {len(processed)} tickers já processados.")
        return records, processed
    except FileNotFoundError:
        return [], set()
    except Exception as e:
        print(f"Aviso: falha ao carregar checkpoint ({e}). Começando do zero.")
        return [], set()


def save_checkpoint(records: list[dict], processed: set[str]) -> None:
    data = {"records": records, "processed": sorted(processed)}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    session = build_session()

    print("Obtendo cookie Yahoo Finance...")
    session.get("https://finance.yahoo.com", timeout=15)
    time.sleep(3)

    # ── B3: carrega tudo de uma vez via fundamentus (sem yfinance) ────────────
    print("=" * 54)
    print("  B3 — carregando via fundamentus")
    print("=" * 54)
    b3_records = get_b3_records()
    print()

    # ── Smoke test (exterior only) ────────────────────────────────────────────
    print("=" * 54)
    print("  SMOKE TEST — tickers do exterior antes do universo completo")
    print("=" * 54)
    smoke_ok = 0
    for sym, reg in SMOKE_TEST_TICKERS:
        print(f"  Testando {sym}...")
        try:
            r = process_ticker(sym, reg, session=session)
            if r and r.get("price"):
                print(f"    OK — preço={r['price']:.2f}  P/L={r['pl'] or 'N/A'}")
                smoke_ok += 1
            else:
                print(f"    AVISO: dados insuficientes para {sym}")
        except Exception as e:
            print(f"    ERRO: {e}")
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if smoke_ok == 0:
        print("\nSmoke test falhou em todos os tickers exteriores. Verifique conexão/yfinance. Abortando.")
        return
    print(f"\nSmoke test: {smoke_ok}/{len(SMOKE_TEST_TICKERS)} tickers OK — continuando com o universo completo.\n")

    # ── Coleta lista de tickers do exterior ───────────────────────────────────
    print("Coletando listas de tickers do exterior...\n")
    ext_tickers = build_exterior_ticker_list()
    ext_total = len(ext_tickers)
    print(f"\nTickers exteriores únicos: {ext_total}\n")

    # ── Retoma do checkpoint (somente exterior — B3 já foi carregado) ─────────
    ext_records_ckpt, processed = load_checkpoint()

    remaining = [(sym, reg) for sym, reg in ext_tickers if sym not in processed]
    offset = ext_total - len(remaining)
    print(f"Tickers exteriores restantes: {len(remaining)}  (retomando do #{offset + 1})\n")

    ext_records: list[dict] = list(ext_records_ckpt)

    for idx, (symbol, region) in enumerate(remaining, offset + 1):
        print(f"Processando ticker {idx} de {ext_total}: {symbol}")
        try:
            record = process_ticker(symbol, region, session=session)
            if record:
                ext_records.append(record)
        except Exception as e:
            print(f"  Erro inesperado em {symbol}: {e}")

        processed.add(symbol)

        if len(processed) % CHECKPOINT_EVERY == 0:
            save_checkpoint(ext_records, processed)
            print(f"  [checkpoint salvo — {len(processed)} tickers exteriores processados]")

        # Pausa aleatória entre tickers
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # Pausa longa entre batches + rotação de sessão
        tickers_done = idx - offset
        if tickers_done % BATCH_SIZE == 0 and tickers_done < len(remaining):
            batch_sleep = random.uniform(BATCH_SLEEP_MIN, BATCH_SLEEP_MAX)
            print(f"  [fim do batch {tickers_done // BATCH_SIZE} — pausa de {batch_sleep:.0f}s + rotação de sessão]")
            session = rotate_session(session)
            time.sleep(batch_sleep)

    save_checkpoint(ext_records, processed)

    # ── Combina B3 + exterior ──────────────────────────────────────────────────
    records: list[dict] = b3_records + ext_records
    print(f"\nDados coletados: {len(b3_records)} B3 + {len(ext_records)} exterior = {len(records)} total")
    print("Aplicando filtros...")

    for r in records:
        r["pass_l1"] = filter_l1(r)

    for r in records:
        if r["pass_l1"]:
            r["pass_l2"] = filter_l2(r)

    apply_l3(records)

    for r in records:
        if r["pass_l3"]:
            r["pass_l4"] = filter_l4(r)

    save_csv(records)
    generate_html(records)
    print_summary(records)


if __name__ == "__main__":
    main()
