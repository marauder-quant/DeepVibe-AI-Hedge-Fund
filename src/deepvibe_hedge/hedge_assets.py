"""Risk-off / hedge universe (Alpaca symbols). Edit here; ``deepvibe_hedge.config`` imports this tuple.

These tickers are the pool the MAD engine rotates into when the regime filter
(``MAD_REGIME_TICKERS``) flips risk-off, OR — in multi-index mode — when one
or more index slots fail their per-ETF trend gate (progressive risk-off).

Each leg is independently trend-gated against its own 200D SMA
(``MAD_SLEEVE_MRAT_LONG``): legs below trend get 0% weight and their share
rotates to ``MAD_REGIME_OFF_SAFE_HARBOR`` (BIL). The ``mrat_distance`` sleeve
scheme then sizes trend-passing legs by how far each is stretched above its
200D, so the book piles into whatever is actually working.

Design principle: cover every "what if this asset class pops off" scenario.
Adding more legs is essentially free — losers get zero weight automatically,
so the downside is tiny (a little extra DB storage + minor MRAT compute)
while the upside is capturing any regime winner the market throws at us.

Groupings are by asset-class role:

* Treasury duration curve — SHY / IEI / IEF / TLH / TLT / TIP
  (rotate between durations as the curve shapes; all failed together in 2022)
* Precious metals + miners — GLD / SLV / GDX
  (GDX = leveraged gold via miners; shine in currency-debasement regimes)
* FX + broad commodities — UUP / DBC / DBA
  (DBC = crude-heavy; DBA = softs/ag; UUP = USD flight-to-quality)
* Specialized commodities — URA / IBIT
  (URA = uranium / nuclear thematic; IBIT = spot Bitcoin, new-asset-class)
* Commodity-sensitive equities — XLE
  (#1 sector in 2022 at +54%; essential for inflationary / energy shocks)
* Regional equities — EWZ / EWW / FXI / KWEB / VGK / EWJ / INDA / VWO
  (geographic diversification — anything from Brazil commodity booms to
  Japan re-rating to China stimulus to Indian structural growth)
* Defensive equity sectors — XLU / XLP / XLV / USMV
  (rate-sensitive utilities, staples, healthcare, and min-vol quality)
* Real estate — VNQ
  (REITs: rate-sensitive but different driver than treasuries)
* Thematic growth — PAVE
  (US infrastructure: multi-year trender from 2021 bill onward)
* Factor tilts — SCHD / MTUM
  (dividend quality + momentum: capture factor-level regimes)
* Credit / risk appetite — HYG
  (high-yield corporate bonds: trend up when credit spreads tighten)
* Managed futures / "crisis alpha" — DBMF
  (explicit regime-flip ETF: systematically long trends / short anti-trends;
  +21% in 2022 when both stocks AND bonds crashed)
* Anti-beta equity hedge — BTAL
  (long low-vol / short high-vol; +17% in 2022, clean equity-crash hedge)
* Safe harbor (cash proxy) — BIL
  (1-3mo T-bills: 100% weight when every other leg fails its trend gate)
"""

hedge_assets = (
    # Treasury duration curve — let trend filters pick whichever end is working
    "SHY",   # 1-3yr Treasury
    "IEI",   # 3-7yr Treasury
    "IEF",   # 7-10yr Treasury
    "TLH",   # 10-20yr Treasury
    "TLT",   # 20+yr Treasury (deflationary / growth-scare hedge)
    "TIP",   # TIPS (inflation-protected)
    # Precious metals + miners
    "GLD",   # gold
    "SLV",   # silver (higher-beta precious metal)
    "GDX",   # gold miners (leveraged gold exposure, amplifies trend)
    # FX + commodities
    "UUP",   # USD bullish (flight-to-quality)
    "DBC",   # broad commodities (crude + metals heavy — stagflation hedge)
    "DBA",   # agriculture (softs/grains — decorrelated from DBC)
    # Specialized commodities
    "URA",   # uranium miners (nuclear renaissance thematic)
    "IBIT",  # spot Bitcoin ETF (new asset class; 2024+ trender, short history)
    # Commodity-sensitive equities
    "XLE",   # Energy sector SPDR (+54% in 2022; the missing piece)
    # Regional equities — capture geographic desyncs
    "EWZ",   # Brazil (commodity-currency EM; +15% in 2022)
    "EWW",   # Mexico (commodity-exporter, nearshoring beneficiary)
    "FXI",   # China large cap (mainland stimulus / regime play)
    "KWEB",  # China internet (higher-beta China thematic)
    "VGK",   # Europe broad (developed international diversifier)
    "EWJ",   # Japan (multi-year re-rating story from 2023+)
    "INDA",  # India (structural growth, often decoupled from DM)
    "VWO",   # broad emerging markets (diversified EM complement)
    # Defensive equity sectors
    "XLU",   # utilities (rate-sensitive defensive)
    "XLP",   # consumer staples
    "XLV",   # healthcare (defensive, different driver than XLU/XLP)
    "USMV",  # min-volatility equities
    # Real estate
    "VNQ",   # US REITs (real-estate rate play, different from treasuries)
    # Thematic growth
    "PAVE",  # US infrastructure (post-2021 multi-year trender)
    # Factor tilts
    "SCHD",  # dividend quality (outperforms in sideways / mid-vol regimes)
    "MTUM",  # momentum factor (captures factor-level trends)
    # Credit / risk appetite gauge
    "HYG",   # high-yield corporate bonds (credit-spread tightening regimes)
    # Managed futures (explicit regime-flip hedge)
    "DBMF",  # iM DBi Managed Futures (long trending assets / short anti-trending)
    # Anti-beta equity (crash hedge)
    "BTAL",  # long low-vol / short high-vol (+17% in 2022)
    # Safe harbor (cash proxy)
    "BIL",   # 1-3mo T-bills — default when nothing else passes the trend gate
)
