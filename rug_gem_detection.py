"""
rug_gem_detection.py — DeFi token scanner: rug detection & gem scoring
Sources: DexScreener · GoPlus Security · Honeypot.is · Etherscan · Rugcheck (Solana)

Usage:
    python rug_gem_detection.py              # scan all chains once
    python rug_gem_detection.py --chain solana  # scan only Solana
    python rug_gem_detection.py --watch 60   # repeat every 60 minutes
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Optional API keys ─────────────────────────────────────────────────────────
ETHERSCAN_API_KEY  = os.getenv("ETHERSCAN_API_KEY", "")
GOPLUSLABS_API_KEY = os.getenv("GOPLUSLABS_API_KEY", "")

# ── Output files ──────────────────────────────────────────────────────────────
OUTPUT_CSV        = "rug_gem_results.csv"
OUTPUT_HTML       = "rug_gem_dashboard.html"
CHECKPOINT_FILE   = "rug_gem_checkpoint.json"

# ── Chains ────────────────────────────────────────────────────────────────────
CHAINS = {
    "ethereum": {"chain_id": "1",    "dex_id": "ethereum", "etherscan": "https://api.etherscan.io/api",  "type": "evm"},
    "bsc":      {"chain_id": "56",   "dex_id": "bsc",      "etherscan": "https://api.bscscan.com/api",   "type": "evm"},
    "base":     {"chain_id": "8453", "dex_id": "base",     "etherscan": "https://api.basescan.org/api",  "type": "evm"},
    "solana":   {"chain_id": "solana","dex_id": "solana",  "etherscan": "",                              "type": "solana"},
}

# ── Solana scoring weights ────────────────────────────────────────────────────
SOL_SCORE_RUGCHECK_HIGH   = 25   # rugcheck score > 80
SOL_SCORE_NO_CRITICAL     = 20   # no critical risks
SOL_SCORE_NOT_RUGGED      = 20   # rugged=false (becomes 0 if true)
SOL_SCORE_HOLDER_SPREAD   = 15   # top10 < 50%
SOL_SCORE_LIQUIDITY_OK    = 10   # liquidity > $50k
SOL_SCORE_LIQUIDITY_LOCKED = 10  # stable price (proxy for locked liq)
SOL_PENALTY_MINTABLE      = -25  # can mint more tokens
SOL_PENALTY_FREEZE        = -20  # freeze authority active

REQUEST_TIMEOUT   = 12
SLEEP_MIN         = 1.0
SLEEP_MAX         = 2.0
CHECKPOINT_EVERY  = 10

# ── Scoring weights ───────────────────────────────────────────────────────────
SCORE_LIQUIDITY_LOCKED   = 20   # locked > 6 months
SCORE_OWNERSHIP_RENOUNCED = 20  # owner = zero address
SCORE_NOT_HONEYPOT       = 15   # honeypot.is clear
SCORE_LOW_TAX            = 10   # buy+sell tax < 10%
SCORE_HOLDER_SPREAD      = 15   # top 10 holders < 50%
SCORE_VERIFIED_CONTRACT  = 10   # source code verified
SCORE_LIQUIDITY_OK       = 10   # liquidity > $50k

PENALTY_MINT_UNLIMITED   = -30
PENALTY_BLACKLIST        = -20
PENALTY_UNVERIFIED_PROXY = -15
PENALTY_LOW_LIQUIDITY    = -20  # < $10k
HONEYPOT_SCORE           = 0    # override to zero

# ── HTTP ──────────────────────────────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
})


def _get(url: str, params: dict = None) -> dict | None:
    try:
        r = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  HTTP error {url}: {e}")
        return None


# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"results": {}, "last_run": None}


def save_checkpoint(data: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 1. Fetch new tokens from DexScreener ─────────────────────────────────────
def _parse_dex_pairs(pairs: list, chain_dex_id: str, seen: set, limit: int = 100) -> list[dict]:
    """Helper: extract token dicts from a DexScreener pairs list."""
    tokens = []
    for pair in pairs:
        if pair.get("chainId") != chain_dex_id:
            continue
        base_token = pair.get("baseToken", {})
        addr = (base_token.get("address") or "").strip()
        if not addr or addr in seen:
            continue
        seen.add(addr)
        liq = pair.get("liquidity", {})
        name = base_token.get("name", "")
        symbol = base_token.get("symbol", "")
        if not name or name.startswith("http"):
            name = symbol if symbol and not symbol.startswith("http") else "?"
        display_name = f"{symbol} ({name})" if name != symbol and name != "?" else symbol
        tokens.append({
            "address":       addr,
            "symbol":        symbol or addr[:8],
            "name":          display_name,
            "liquidity_usd": float(liq.get("usd") or 0),
            "pair_address":  pair.get("pairAddress", ""),
            "created_at":    pair.get("pairCreatedAt", ""),
        })
        if len(tokens) >= limit:
            break
    return tokens


def fetch_new_tokens(chain_key: str) -> list[dict]:
    """Returns list of {address, symbol, name, liquidity_usd, pair_address, created_at}."""
    chain = CHAINS[chain_key]
    seen: set = set()
    tokens = []

    if chain["type"] == "solana":
        # Use token-boosts endpoint for Solana — returns tokens with real liquidity and volume
        data = _get("https://api.dexscreener.com/token-boosts/top/v1")
        if data and isinstance(data, list):
            for item in data:
                if item.get("chainId") != chain["dex_id"]:
                    continue
                addr = (item.get("tokenAddress") or "").strip()
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                # Fetch pair data for this token to get liquidity
                pair_data = _get(f"https://api.dexscreener.com/latest/dex/tokens/{addr}")
                liq_usd = 0.0
                symbol = item.get("description", "") or addr[:8]
                name = symbol
                if pair_data and pair_data.get("pairs"):
                    best = pair_data["pairs"][0]
                    liq_usd = float((best.get("liquidity") or {}).get("usd") or 0)
                    base = best.get("baseToken", {})
                    sym = base.get("symbol", "") or symbol
                    raw_name = base.get("name", "") or sym
                    symbol = sym
                    name_clean = sym if raw_name.startswith("http") else raw_name
                    name = f"{sym} ({name_clean})" if name_clean != sym and name_clean != "?" else sym
                tokens.append({
                    "address":       addr,
                    "symbol":        symbol,
                    "name":          name,
                    "liquidity_usd": liq_usd,
                    "pair_address":  "",
                    "created_at":    "",
                })
                if len(tokens) >= 100:
                    break
        print(f"  DexScreener [{chain_key}]: {len(tokens)} tokens encontrados")
        return tokens

    # EVM chains: token-profiles endpoint, fallback to pairs search
    url_new = "https://api.dexscreener.com/token-profiles/latest/v1"
    data = _get(url_new)
    if data and isinstance(data, list):
        for item in data:
            if item.get("chainId") != chain["dex_id"]:
                continue
            addr = (item.get("tokenAddress") or "").strip()
            if not addr or addr in seen:
                continue
            seen.add(addr)
            tokens.append({
                "address":       addr,
                "symbol":        item.get("header", "?"),
                "name":          item.get("description", ""),
                "liquidity_usd": 0.0,
                "pair_address":  "",
                "created_at":    "",
            })

    if not tokens:
        url_pairs = f"https://api.dexscreener.com/latest/dex/search?q={chain['dex_id']}"
        data2 = _get(url_pairs)
        if data2 and "pairs" in data2:
            tokens = _parse_dex_pairs(data2["pairs"], chain["dex_id"], seen)

    print(f"  DexScreener [{chain_key}]: {len(tokens)} tokens encontrados")
    return tokens[:100]


# ── 2. Analyze token ──────────────────────────────────────────────────────────
def analyze_goplus(address: str, chain_id: str) -> dict:
    """Query GoPlus Security API for token security info."""
    url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"
    params = {"contract_addresses": address}
    if GOPLUSLABS_API_KEY:
        params["apikey"] = GOPLUSLABS_API_KEY
    data = _get(url, params)
    if not data or data.get("code") != 1:
        return {}
    result = data.get("result", {})
    return result.get(address.lower(), result.get(address, {}))


def analyze_honeypot(address: str) -> dict:
    """Query honeypot.is for honeypot detection (Ethereum only)."""
    url = "https://api.honeypot.is/v2/IsHoneypot"
    params = {"address": address}
    data = _get(url, params)
    return data or {}


def analyze_etherscan(address: str, chain_key: str) -> dict:
    """Check if contract source is verified on Etherscan/BscScan/BaseScan."""
    chain = CHAINS[chain_key]
    params = {
        "module":  "contract",
        "action":  "getsourcecode",
        "address": address,
    }
    if ETHERSCAN_API_KEY:
        params["apikey"] = ETHERSCAN_API_KEY
    data = _get(chain["etherscan"], params)
    if not data or data.get("status") != "1":
        return {"verified": False}
    result = data.get("result", [{}])
    src = result[0].get("SourceCode", "") if result else ""
    return {"verified": bool(src and src != "0x")}


def analyze_rugcheck(mint_address: str) -> dict:
    """Query Rugcheck API for Solana token security report (/report endpoint).
    Returns {} silently if the token is not indexed or any error occurs.
    """
    url = f"https://api.rugcheck.xyz/v1/tokens/{mint_address}/report"
    try:
        r = _SESSION.get(url, headers={"accept": "application/json"}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json() or {}
    except Exception:
        pass
    return {}


def analyze_goplus_solana(address: str) -> dict:
    """Query GoPlus Solana token security endpoint."""
    url = "https://api.gopluslabs.io/api/v1/solana/token_security"
    params = {"contract_addresses": address}
    if GOPLUSLABS_API_KEY:
        params["apikey"] = GOPLUSLABS_API_KEY
    data = _get(url, params)
    if not data or data.get("code") != 1:
        return {}
    result = data.get("result", {})
    return result.get(address, result.get(address.lower(), {}))


def analyze_token(address: str, chain_key: str) -> dict:
    """Run all security checks and return merged analysis dict."""
    chain = CHAINS[chain_key]
    analysis = {"address": address, "chain": chain_key}

    if chain["type"] == "solana":
        # Rugcheck — preserva endereço original (Solana é case-sensitive)
        mint_address = address  # nao faz .lower()
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        rc = analyze_rugcheck(mint_address)
        analysis["rugcheck"] = rc

        # GoPlus Solana
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        gp_sol = analyze_goplus_solana(address)
        analysis["goplus_solana"] = gp_sol

        return analysis

    # EVM chains
    # GoPlus
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    gp = analyze_goplus(address, chain["chain_id"])
    analysis["goplus"] = gp

    # Honeypot.is (only reliable on ethereum)
    if chain_key == "ethereum":
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        hp = analyze_honeypot(address)
        analysis["honeypot"] = hp
    else:
        analysis["honeypot"] = {}

    # Etherscan
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
    es = analyze_etherscan(address, chain_key)
    analysis["etherscan"] = es

    return analysis


# ── 3. Score token ────────────────────────────────────────────────────────────
def _gp_flag(gp: dict, key: str) -> str:
    """Returns GoPlus field value as string, default '0'."""
    return str(gp.get(key, "0")).strip()


def score_token_solana(token: dict, analysis: dict) -> tuple[int, list[str], list[str]]:
    """Solana-specific scoring using Rugcheck + GoPlus Solana."""
    rc  = analysis.get("rugcheck", {})
    gps = analysis.get("goplus_solana", {})
    liq = token.get("liquidity_usd", 0.0)
    has_rugcheck = bool(rc)

    score     = 50  # base always starts at 50 regardless of Rugcheck availability
    positives = []
    negatives = []

    # ── Rugged check (instant disqualifier) ──────────────────────────────
    if rc.get("rugged", False):
        return 0, [], ["TOKEN MARCADO COMO RUGGED"]

    if has_rugcheck:
        # ── Not rugged ───────────────────────────────────────────────────
        score += SOL_SCORE_NOT_RUGGED
        positives.append("Nao foi marcado como rugged")

        # ── Rugcheck score ────────────────────────────────────────────────
        rc_score = rc.get("score", 0) or 0
        if rc_score > 80:
            score += SOL_SCORE_RUGCHECK_HIGH
            positives.append(f"Rugcheck score alto: {rc_score}")
        else:
            negatives.append(f"Rugcheck score baixo: {rc_score}")

        # ── Critical risks ────────────────────────────────────────────────
        risks = rc.get("risks", []) or []
        critical = [r for r in risks if str(r.get("level", "")).lower() in ("critical", "danger", "high")]
        if not critical:
            score += SOL_SCORE_NO_CRITICAL
            positives.append("Sem risks criticos no Rugcheck")
        else:
            for r in critical:
                negatives.append(f"Risk [{r.get('level','?')}]: {r.get('name', r.get('description', '?'))}")

        # ── Locked liquidity (rugcheck) ───────────────────────────────────
        lp_locked = rc.get("lp_locked", False) or rc.get("lpLockedPct", 0) or 0
        if lp_locked and float(lp_locked) > 0.5:
            score += SOL_SCORE_LIQUIDITY_LOCKED
            positives.append(f"Liquidez travada {float(lp_locked)*100:.0f}%")
        else:
            negatives.append("Liquidez nao confirmada como travada")
    else:
        negatives.append("Dados insuficientes: Rugcheck nao respondeu")

    # ── Mintable (GoPlus Solana) ──────────────────────────────────────────
    if gps:
        if str(gps.get("mintable", "0")) == "1" or str(gps.get("mint_authority", "")) not in ("", "0", "null", "None"):
            score += SOL_PENALTY_MINTABLE
            negatives.append("Mint authority ativa (pode emitir mais tokens)")
        else:
            positives.append("Mint authority inativa")

        # ── Freeze authority ──────────────────────────────────────────────
        freeze = str(gps.get("freeze_authority", "")).strip()
        if freeze and freeze not in ("0", "null", "None", ""):
            score += SOL_PENALTY_FREEZE
            negatives.append("Freeze authority ativa (pode congelar carteiras)")
        else:
            positives.append("Freeze authority inativa")

        # ── Holder concentration ──────────────────────────────────────────
        holders = gps.get("holders", []) or []
        top10_pct = sum(float(h.get("percent", 0) or 0) for h in holders[:10])
        if top10_pct < 0.50:
            score += SOL_SCORE_HOLDER_SPREAD
            positives.append(f"Top10 holders {top10_pct*100:.0f}% do supply")
        else:
            negatives.append(f"Concentrado: top10 tem {top10_pct*100:.0f}%")

    # ── Liquidity tiers ───────────────────────────────────────────────────
    if liq == 0:
        score = min(score, 40)
        negatives.append("Sem liquidez (score cap 40)")
    elif liq >= 100_000:
        score += 15
        positives.append(f"Liquidez alta ${liq:,.0f}")
    elif liq >= 50_000:
        score += 10
        positives.append(f"Liquidez boa ${liq:,.0f}")
    elif liq >= 10_000:
        score += 5
        positives.append(f"Liquidez minima ${liq:,.0f}")
    else:
        score -= 20
        negatives.append(f"Liquidez baixa ${liq:,.0f}")

    score = max(0, min(100, score))
    return score, positives, negatives


def score_token(token: dict, analysis: dict) -> tuple[int, list[str], list[str]]:
    """
    Returns (score, positives, negatives).
    score is clamped to [0, 100].
    Routes to chain-specific scorer.
    """
    chain_key = analysis.get("chain", "ethereum")
    if CHAINS.get(chain_key, {}).get("type") == "solana":
        return score_token_solana(token, analysis)
    gp  = analysis.get("goplus", {})
    hp  = analysis.get("honeypot", {})
    es  = analysis.get("etherscan", {})
    liq = token.get("liquidity_usd", 0.0)

    score     = 50  # neutral base
    positives = []
    negatives = []

    # ── Honeypot check (instant disqualifier) ─────────────────────────────
    is_honeypot_hp = hp.get("isHoneypot", False)
    is_honeypot_gp = _gp_flag(gp, "is_honeypot") == "1"
    if is_honeypot_hp or is_honeypot_gp:
        return HONEYPOT_SCORE, [], ["HONEYPOT CONFIRMADO"]

    score += SCORE_NOT_HONEYPOT
    positives.append("Nao e honeypot")

    # ── Liquidity locked ──────────────────────────────────────────────────
    lp_locked = _gp_flag(gp, "lp_lock_detail")
    lock_ratio = float(gp.get("lp_holder_analysis", {}).get("lock_rate", 0) or 0) if isinstance(gp.get("lp_holder_analysis"), dict) else 0.0
    # GoPlus returns lock info in lp_holders list
    locked_pct = 0.0
    for holder in gp.get("lp_holders", []):
        if str(holder.get("is_locked", "0")) == "1":
            locked_pct += float(holder.get("percent", 0) or 0)
    if locked_pct >= 0.80:
        score += SCORE_LIQUIDITY_LOCKED
        positives.append(f"Liquidez travada {locked_pct*100:.0f}%")
    else:
        negatives.append(f"Liquidez nao travada ({locked_pct*100:.0f}%)")

    # ── Ownership renounced ───────────────────────────────────────────────
    owner = _gp_flag(gp, "owner_address")
    renounced = (
        _gp_flag(gp, "owner_change_balance") == "0"
        or owner in ("", "0x0000000000000000000000000000000000000000")
        or _gp_flag(gp, "can_take_back_ownership") == "0"
    )
    if renounced:
        score += SCORE_OWNERSHIP_RENOUNCED
        positives.append("Ownership renunciado")
    else:
        negatives.append("Owner ainda ativo")

    # ── Tax check ─────────────────────────────────────────────────────────
    try:
        buy_tax  = float(gp.get("buy_tax",  0) or 0)
        sell_tax = float(gp.get("sell_tax", 0) or 0)
    except (ValueError, TypeError):
        buy_tax = sell_tax = 0.0

    # GoPlus returns tax as fraction (0.05 = 5%) or percent — normalize
    if buy_tax > 1:
        buy_tax /= 100
    if sell_tax > 1:
        sell_tax /= 100

    if buy_tax + sell_tax < 0.10:
        score += SCORE_LOW_TAX
        positives.append(f"Taxa baixa (buy {buy_tax*100:.1f}% / sell {sell_tax*100:.1f}%)")
    else:
        negatives.append(f"Taxa alta (buy {buy_tax*100:.1f}% / sell {sell_tax*100:.1f}%)")

    # ── Holder concentration ──────────────────────────────────────────────
    holders = gp.get("holders", [])
    top10_pct = sum(float(h.get("percent", 0) or 0) for h in holders[:10])
    if top10_pct < 0.50:
        score += SCORE_HOLDER_SPREAD
        positives.append(f"Top10 holders {top10_pct*100:.0f}% do supply")
    else:
        negatives.append(f"Concentrado: top10 tem {top10_pct*100:.0f}%")

    # ── Verified contract ─────────────────────────────────────────────────
    verified = es.get("verified", False) or _gp_flag(gp, "is_open_source") == "1"
    if verified:
        score += SCORE_VERIFIED_CONTRACT
        positives.append("Contrato verificado")
    else:
        negatives.append("Contrato nao verificado")

    # ── Liquidity size ────────────────────────────────────────────────────
    if liq == 0:
        score = min(score, 40)
        negatives.append("Sem liquidez (score cap 40)")
    elif liq >= 50_000:
        score += SCORE_LIQUIDITY_OK
        positives.append(f"Liquidez ${liq:,.0f}")
    elif liq < 10_000:
        score += PENALTY_LOW_LIQUIDITY
        negatives.append(f"Liquidez baixa ${liq:,.0f}")
    else:
        negatives.append(f"Liquidez media ${liq:,.0f}")

    # ── Rug penalties ─────────────────────────────────────────────────────
    if _gp_flag(gp, "is_mintable") == "1":
        score += PENALTY_MINT_UNLIMITED
        negatives.append("Funcao mint sem limite")

    if _gp_flag(gp, "is_blacklisted") == "1":
        score += PENALTY_BLACKLIST
        negatives.append("Blacklist de carteiras ativa")

    if _gp_flag(gp, "is_proxy") == "1" and not verified:
        score += PENALTY_UNVERIFIED_PROXY
        negatives.append("Proxy nao verificado")

    score = max(0, min(100, score))
    return score, positives, negatives


# ── 4. HTML dashboard ─────────────────────────────────────────────────────────
def color_for_score(score: int) -> tuple[str, str]:
    """Returns (bg_color, label)."""
    if score >= 70:
        return "#1a3a2a", "GEM"
    elif score >= 40:
        return "#3a3010", "NEUTRO"
    else:
        return "#3a1010", "RUG"


def generate_html(results: list[dict], generated_at: str) -> str:
    rows = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        bg, label = color_for_score(r["score"])
        score_color = "#2ecc71" if r["score"] >= 70 else ("#f1c40f" if r["score"] >= 40 else "#e74c3c")
        positives_html = "".join(f'<li style="color:#2ecc71">{p}</li>' for p in r.get("positives", []))
        negatives_html = "".join(f'<li style="color:#e74c3c">{n}</li>' for n in r.get("negatives", []))
        sources = r.get("data_sources", "DexScreener")
        rows.append(f"""
        <tr style="background:{bg}">
          <td style="font-family:monospace;font-size:11px">{r['address'][:10]}...{r['address'][-6:]}</td>
          <td><strong>{r['symbol']}</strong></td>
          <td>{r['chain']}</td>
          <td style="color:{score_color};font-size:1.4rem;font-weight:700;text-align:center">{r['score']}</td>
          <td style="color:{score_color};font-weight:700">{label}</td>
          <td style="font-size:11px">${r.get('liquidity_usd', 0):,.0f}</td>
          <td style="font-size:10px;color:#8b949e">{sources}</td>
          <td style="font-size:10px"><ul style="margin:0;padding-left:14px">{positives_html}{negatives_html}</ul></td>
        </tr>""")

    rows_html = "\n".join(rows) if rows else '<tr><td colspan="8" style="text-align:center;color:#8b949e">Nenhum token analisado ainda.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rug/Gem Detection</title>
<style>
  :root {{ --bg:#0d1117; --surface:#161b22; --border:#21262d; --text:#c9d1d9; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:24px; }}
  h1 {{ color:#e6edf3; margin-bottom:6px; }}
  .sub {{ color:#8b949e; font-size:13px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:var(--surface); color:#8b949e; font-weight:600; padding:10px 12px; text-align:left; border-bottom:2px solid var(--border); position:sticky; top:0; }}
  td {{ padding:10px 12px; border-bottom:1px solid #21262d; vertical-align:top; }}
  tr:hover td {{ background:rgba(255,255,255,.03); }}
  .legend {{ display:flex; gap:20px; margin-bottom:16px; font-size:12px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:5px; }}
</style>
</head>
<body>
<h1>Rug / Gem Detection</h1>
<div class="sub">Gerado em {generated_at} &nbsp;·&nbsp; {len(results)} tokens analisados</div>
<div class="legend">
  <span><span class="dot" style="background:#2ecc71"></span>Score &gt;= 70: Gem potencial</span>
  <span><span class="dot" style="background:#f1c40f"></span>Score 40-69: Investigar</span>
  <span><span class="dot" style="background:#e74c3c"></span>Score &lt; 40: Rug provavel</span>
</div>
<table>
  <thead>
    <tr>
      <th>Endereco</th><th>Token</th><th>Chain</th>
      <th style="text-align:center">Score</th><th>Status</th>
      <th>Liquidez</th><th>Fonte de Dados</th><th>Detalhes</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
<div style="color:#484f58;font-size:11px;margin-top:20px;text-align:center">
  Dados: DexScreener · GoPlus Security · Honeypot.is · Etherscan &nbsp;|&nbsp; Apenas para fins educacionais.
</div>
</body>
</html>"""


# ── 5. CSV export ─────────────────────────────────────────────────────────────
CSV_FIELDS = ["address", "symbol", "name", "chain", "score", "label",
              "liquidity_usd", "positives", "negatives", "analyzed_at"]


def save_csv(results: list[dict]):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["positives"] = " | ".join(r.get("positives", []))
            row["negatives"] = " | ".join(r.get("negatives", []))
            _, row["label"] = color_for_score(r["score"])
            writer.writerow(row)
    print(f"  CSV salvo: {OUTPUT_CSV}")


def _data_sources(chain_key: str, analysis: dict) -> str:
    """Returns a human-readable string of which APIs responded."""
    parts = []
    if CHAINS.get(chain_key, {}).get("type") == "solana":
        if analysis.get("goplus_solana"):
            parts.append("GoPlus")
        if analysis.get("rugcheck"):
            parts.append("Rugcheck")
    else:
        if analysis.get("goplus"):
            parts.append("GoPlus")
        if analysis.get("honeypot"):
            parts.append("Honeypot")
        if analysis.get("etherscan", {}).get("verified") is not None:
            parts.append("Etherscan")
    return " | ".join(parts) if parts else "DexScreener"


# ── Main ──────────────────────────────────────────────────────────────────────
def run_scan(target_chains: list[str]):
    checkpoint = load_checkpoint()
    cached: dict = checkpoint.get("results", {})
    all_results: list[dict] = list(cached.values())
    new_this_run = 0

    for chain_key in target_chains:
        print(f"\n[{chain_key.upper()}] Buscando tokens...")
        tokens = fetch_new_tokens(chain_key)

        for idx, token in enumerate(tokens, 1):
            addr = token["address"]
            cache_key = f"{chain_key}:{addr}"

            if cache_key in cached:
                print(f"  [{idx}/{len(tokens)}] {token['symbol']} ({addr[:8]}...) — cache")
                continue

            # Ignora tokens Solana sem liquidez mínima
            if chain_key == "solana" and token.get("liquidity_usd", 0) < 10_000:
                print(f"  [{idx}/{len(tokens)}] {token['symbol']} — liquidez insuficiente (${token.get('liquidity_usd',0):,.0f}), pulando")
                continue

            print(f"  [{idx}/{len(tokens)}] Analisando {token['symbol']} ({addr[:8]}...)...")
            analysis = analyze_token(addr, chain_key)
            score, positives, negatives = score_token(token, analysis)

            label_color = "GEM" if score >= 70 else ("NEUTRO" if score >= 40 else "RUG")
            print(f"    Score: {score:3d}  [{label_color}]  liq=${token.get('liquidity_usd',0):,.0f}")

            result = {
                "address":       addr,
                "symbol":        token["symbol"],
                "name":          token["name"],
                "chain":         chain_key,
                "score":         score,
                "liquidity_usd": token.get("liquidity_usd", 0.0),
                "positives":     positives,
                "negatives":     negatives,
                "data_sources":  _data_sources(chain_key, analysis),
                "analyzed_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            cached[cache_key] = result
            all_results.append(result)
            new_this_run += 1

            if new_this_run % CHECKPOINT_EVERY == 0:
                checkpoint["results"] = cached
                checkpoint["last_run"] = datetime.now(timezone.utc).isoformat()
                save_checkpoint(checkpoint)
                print(f"  Checkpoint salvo ({new_this_run} tokens novos)")

    # Final checkpoint
    checkpoint["results"] = cached
    checkpoint["last_run"] = datetime.now(timezone.utc).isoformat()
    save_checkpoint(checkpoint)

    # Outputs
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    save_csv(all_results)
    html = generate_html(all_results, ts)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard: {OUTPUT_HTML}")
    print(f"\n  Total: {len(all_results)} tokens | Novos esta rodada: {new_this_run}")

    gems    = sum(1 for r in all_results if r["score"] >= 70)
    neutros = sum(1 for r in all_results if 40 <= r["score"] < 70)
    rugs    = sum(1 for r in all_results if r["score"] < 40)
    print(f"  GEM: {gems}  |  NEUTRO: {neutros}  |  RUG: {rugs}")


def main():
    parser = argparse.ArgumentParser(description="Rug/Gem Detection Scanner")
    parser.add_argument("--chain", choices=list(CHAINS.keys()) + ["all"], default="all",
                        help="Chain to scan (default: all)")
    parser.add_argument("--watch", type=int, nargs="?", const=60, default=None,
                        metavar="MINUTES", help="Repeat scan every N minutes")
    args = parser.parse_args()

    target = list(CHAINS.keys()) if args.chain == "all" else [args.chain]

    # Limpa checkpoint para forçar reanálise limpa
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        print(f"  Checkpoint anterior removido ({CHECKPOINT_FILE})")

    if args.watch:
        print(f"Modo watch: escaneando a cada {args.watch} minutos. Ctrl+C para parar.")
        while True:
            print(f"\n=== SCAN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            run_scan(target)
            print(f"Aguardando {args.watch} minutos...")
            time.sleep(args.watch * 60)
    else:
        print(f"=== RUG/GEM SCAN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        run_scan(target)


if __name__ == "__main__":
    main()
