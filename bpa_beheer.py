# -*- coding: utf-8 -*-
"""
BPA Jaarlijks Beheer Tool
=========================
Eenmaal per jaar kan BPA via dit script:
  1. Het aantal subscripties (klanten) per component aanpassen.
  2. Nieuwe componenten toevoegen met bijbehorende lambda en levertijd.
  3. Een overzicht opslaan van de bijgewerkte basisvoorraden.

Instellingen worden opgeslagen in bpa_config.json naast dit script.
Bij de volgende run worden die automatisch geladen.

Gebruik:
    python bpa_beheer.py
"""

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

# ── pad zoeken naar model.py ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import BPAOptimizationModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ══════════════════════════════════════════════════════════════════════════════
#  PADEN
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(__file__)
CONFIG_PATH   = os.path.join(SCRIPT_DIR, 'bpa_config.json')
HISTORY_PATH  = os.path.join(SCRIPT_DIR, 'bpa_history.json')
EXCEL_PATH  = os.path.join(SCRIPT_DIR,
                           'annual_use_abc_met_artikeldata_complete_europa.xlsx')
SHEET_NAME  = 'Filtered '

# Subscriptie-dataset (zonder 231-AS: RSPL). Bevat per component het aantal
# klantlocaties (Aantal_klantlocaties_5jr) en per (component, klant) de
# regionale adoption rate. Standaardbron voor de tab 'Verwachte subscripties'.
SUBSCRIPTIES_PATH = os.path.join(
    SCRIPT_DIR,
    'annual_use_abc_met_artikeldata_subscripties_europa_zonder_rspl.xlsx')

# Tabs voor de subscriptie-simulatie (binomiale adoptie per component/klant).
SHEET_ADOPTIE      = 'Adoptie'
# Naamvarianten van de adoptie-tab; de subscriptie-dataset gebruikt
# 'Adoption_rate_per_klant', de oude complete-Excel gebruikt 'Adoptie'.
ADOPTIE_SHEET_CANDIDATES = ('Adoptie', 'Adoption_rate_per_klant')
SHEET_BESTELLINGEN = 'bestellingen_per_klant'

# Selectiebestand geproduceerd door classificatie_scoring.py.
# Aanwezigheid activeert de classificatie-koppeling (whitelist + LT-bron).
SELECTIE_PATH = os.path.join(SCRIPT_DIR, 'bpa_selectie.json')

# ── Filters (zelfde als base_stock_overview.py) ───────────────────────────────
FILTER_ARTICLE_TYPES = ['Critical', 'Onbekend']
FILTER_ABC           = ['A']
FILTER_INCOURANT     = ['Uitgeschakeld']
FILTER_MIN_VP        = 1000
FILTER_MIN_KLANTEN   = 5

# Standaard serviceniveaus voor het overzicht
SERVICE_LEVELS = [0.980, 0.985, 0.990, 0.995, 0.999]

# Default MTBF (jaren) wanneer MTBF onbekend is in zowel Excel als classificatie.
# λ valt dan terug op n_klanten / DEFAULT_MTBF_JR i.p.v. historische orders.
DEFAULT_MTBF_JR = 10.0


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG – laden / opslaan
# ══════════════════════════════════════════════════════════════════════════════

def _leeg_config() -> dict:
    return {
        "aangemaakt":          str(date.today()),
        "aangepast":           str(date.today()),
        # code → aantal subscripties (overschrijft het aantal klantlocaties)
        "n_klanten_overrides": {},
        # code → inkoopprijs override (€)
        "ip_overrides": {},
        # code → verkoopprijs override (€)
        "vp_overrides": {},
        # code → levertijd override (dagen)
        "lt_overrides": {},
        # handmatig toegevoegde componenten
        # code → {"descr": str, "lambda_per_jaar": float, "lt_dagen": int,
        #          "n_klanten": int, "ip": float}
        "handmatige_componenten": {},
        # codes die volledig uit het model worden uitgesloten (Excel én handmatig)
        "uitgesloten_componenten": [],
        # klantnaam → lijst van {"code": str, "descr": str, "ingangsdatum": str (ISO-datum)}
        "klanten": {},
        # code → werkelijke fysieke voorraad (handmatig bijgewerkt door BPA)
        "voorraad_actueel": {},
    }


def laad_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        # voeg ontbrekende sleutels toe bij oudere bestanden
        basis = _leeg_config()
        for k, v in basis.items():
            cfg.setdefault(k, v)
        return cfg
    return _leeg_config()


def sla_config_op(cfg: dict, excel_file=None) -> None:
    """
    Bewaar eerst een snapshot van de oude situatie en schrijf daarna
    de gewijzigde configuratie atomisch naar bpa_config.json.
    """
    oude_cfg = laad_config()

    try:
        _sla_history_snapshot(oude_cfg, excel_file)
    except Exception as exc:
        print(f"History-snapshot kon niet worden opgeslagen: {exc}")

    cfg["aangepast"] = str(date.today())

    tijdelijk_pad = CONFIG_PATH + ".tmp"

    with open(tijdelijk_pad, "w", encoding="utf-8") as bestand:
        json.dump(
            cfg,
            bestand,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tijdelijk_pad, CONFIG_PATH)


def _sla_history_snapshot(cfg: dict, excel_file=None) -> None:
    """Voeg een snapshot toe aan bpa_history.json na elke config-update."""
    try:
        df = bereken_overzicht(cfg, excel_file)
    except Exception:
        return
    if df.empty:
        return

    sl_cols = [c for c in df.columns if c.startswith('s@')]
    snapshot = {
        'datum':      str(date.today()),
        'n_klanten':  int(df['n_klanten'].median()) if not df.empty else 0,
        'n_actief':   len(df),
        'totalen':    {c: int(df[c].sum()) for c in sl_cols},
        'componenten': {
            str(code): {c: int(df.at[code, c]) for c in sl_cols}
            for code in df.index
        },
    }

    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding='utf-8') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    # Vervang bestaande snapshot van vandaag (meest recente wint)
    history = [h for h in history if h.get('datum') != snapshot['datum']]
    history.append(snapshot)
    history.sort(key=lambda h: h['datum'])

    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LADEN UIT EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dutch_price(val):
    if pd.isna(val):
        return 0.0
    # Pandas leest numerieke cellen al als float – direct teruggeven
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lstrip('\x80€ ').strip()
    # Dutch format: punt = duizendteken, komma = decimaalteken
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_lt_dagen(val) -> int:
    """Parseer levertijd in dagen, robuust voor int/float/strings.

    Spiegelt classificatie._parse_lt_dagen zodat de tool exact dezelfde
    levertijd overneemt als de classificatie. Accepteert numerieke cellen
    (30.0), '30', '30 dagen', NL-decimalen ('30,0') en waarden met een
    leidende spatie. Geeft 0 bij leeg/onparseerbaar; 0 wordt downstream
    aangevuld tot de default van 30 dagen (LT_bron 'nul→30').

    De oude implementatie (``int(str(v).split()[0])`` met
    ``str(v)[0].isdigit()``) faalde op float-cellen ('30.0' → ValueError,
    waardoor de héle Excel-load afbrak) en op NL-decimalen of een leidende
    spatie (→ stil 0), waardoor de juiste levertijd vaak niet meekwam.
    """
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return int(val) if val > 0 else 0
    s = str(val).strip()
    if not s:
        return 0
    head = s.split()[0].replace(',', '.')
    try:
        n = int(float(head))
        return n if n > 0 else 0
    except ValueError:
        return 0


def laad_excel_onderdelen(excel_file=None, *, skip_hard_filter: bool = True) -> pd.DataFrame:
    """Laad onderdelen uit de Excel-spreadsheet.
    excel_file: bestandspad (str) of file-like object (BytesIO / UploadedFile).
    Valt terug op EXCEL_PATH als niet opgegeven.
    skip_hard_filter: standaard True — alle BPA-hardfilters (ABC/VP/n_cust/...)
    worden overgeslagen. De selectie wordt volledig bepaald door de
    classificatie-tool. Zet expliciet op False om de oude hardfilter toe te passen."""
    bron = excel_file if excel_file is not None else EXCEL_PATH
    df = pd.read_excel(bron, sheet_name=SHEET_NAME)
    df = df.rename(columns={
        'Verkooporderregel artikel.Artikel.Artikelcode': 'Code',
        'Omschrijving_standaard_artikelen':             'Descr',
        'Standaard verkoopprijs':                       'VP',
        'Inkoopprijs (standaard)':                      '_IP_raw',
        'Totaal_orders_5jr':                            'Totaal_orders_5jr',
        'Aantal_klantlocaties_5jr':                     'n_cust',
        'Hoofdleverancier.Levertijd':                   '_LT_raw',
        'MTBF(years)':                                  'MTBF',
    })
    df = df[df['Code'].notna()].copy()
    df['Code'] = df['Code'].astype(str).str.strip()
    df = df[~df['Code'].str.lower().isin(('', 'nan', 'none', 'null'))]
    df = df.drop_duplicates(subset='Code', keep='first')
    df['IP']      = df['_IP_raw'].apply(_parse_dutch_price)
    df['LT_days'] = df['_LT_raw'].apply(_parse_lt_dagen)
    df['MTBF'] = pd.to_numeric(
        df.get('MTBF', pd.Series(np.nan, index=df.index)), errors='coerce'
    )
    if skip_hard_filter:
        mask = pd.Series(True, index=df.index)
    else:
        mask = (
            df['ArticleType'].isin(FILTER_ARTICLE_TYPES) &
            df['ABC_categorie'].isin(FILTER_ABC) &
            df['Incourant'].isin(FILTER_INCOURANT) &
            (df['VP'] >= FILTER_MIN_VP) &
            (df['n_cust'] >= FILTER_MIN_KLANTEN)
        )
    return df[mask].set_index('Code')[['Descr', 'VP', 'IP', 'Totaal_orders_5jr', 'n_cust', 'LT_days', 'MTBF']]


def _lambda_voor_rij(row, n_klanten: int) -> float:
    """λ = n_klanten / MTBF(jaar).

    Fallback: als MTBF ontbreekt of ≤ 0, gebruik DEFAULT_MTBF_JR (= 10 jr)
    in plaats van de historische orders-berekening. Dit geeft een
    voorspelbare λ ook voor componenten zonder MTBF-data.
    """
    mtbf = row['MTBF']
    if pd.notna(mtbf) and mtbf > 0:
        return n_klanten / mtbf
    return n_klanten / DEFAULT_MTBF_JR


def _n_uit_klantlocaties(raw) -> int:
    """Aantal subscripties uit het werkelijke aantal klantlocaties (n_cust).

    Vervangt de oude globale standaardwaarde: bij ontbrekende of ongeldige
    data valt het aantal terug op 1 (minimaal één potentiële klant).
    """
    try:
        if raw is None or pd.isna(raw):
            return 1
        n = int(float(raw))
        return n if n >= 1 else 1
    except (TypeError, ValueError):
        return 1


def extra_klanten_tot_voorraadstap(
    n_klanten: int,
    lambda_per_jaar: float,
    lt_dagen: int,
    service_level: float = 0.99,
    max_extra: int = 100_000,
) -> int | None:
    """Aantal extra klanten waarbij de adviesvoorraad voor het eerst stijgt."""
    if n_klanten <= 0 or lambda_per_jaar <= 0 or lt_dagen <= 0:
        return None

    lambda_per_klant = lambda_per_jaar / n_klanten
    lt_jaren = lt_dagen / 365
    huidige_voorraad = BPAOptimizationModel.inverse_service_level(
        service_level, lambda_per_jaar, lt_jaren
    )
    laag, hoog = n_klanten + 1, n_klanten + max_extra
    if BPAOptimizationModel.inverse_service_level(
        service_level, lambda_per_klant * hoog, lt_jaren
    ) <= huidige_voorraad:
        return None

    while laag < hoog:
        midden = (laag + hoog) // 2
        voorraad = BPAOptimizationModel.inverse_service_level(
            service_level, lambda_per_klant * midden, lt_jaren
        )
        if voorraad > huidige_voorraad:
            hoog = midden
        else:
            laag = midden + 1
    return laag - n_klanten


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATIE-KOPPELING
# ══════════════════════════════════════════════════════════════════════════════

def laad_classificatie_selectie() -> dict:
    """
    Lees bpa_selectie.json (geproduceerd door classificatie_scoring.py).

    Returns:
        dict met:
          'items'       : {code(str) → {'score','lt_dagen','lt_bron','abc'}}
          'gegenereerd' : timestamp-string of None
          'lt_overzicht': samenvatting of {}
        Lege dict als het bestand niet bestaat of corrupt is.
    """
    if not os.path.exists(SELECTIE_PATH):
        return {}
    try:
        with open(SELECTIE_PATH, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠  bpa_selectie.json niet leesbaar: {e}")
        return {}
    return {
        'items':        {str(it['code']): it for it in data.get('items', [])},
        'selectie_actief': bool(data.get('selectie_actief', data.get('items', []))),
        'gegenereerd':  data.get('gegenereerd'),
        'lt_overzicht': data.get('lt_overzicht', {}),
        'threshold':    data.get('threshold'),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BEREKEN OVERZICHT
# ══════════════════════════════════════════════════════════════════════════════

def bereken_overzicht(cfg: dict, excel_file=None) -> pd.DataFrame:
    """
    Geeft een DataFrame terug met per component:
      Code, Descr, n_klanten, lambda_per_jaar, LT_dagen, mu, en s per serviceniveau.
    Combineert Excel-onderdelen met handmatige toevoegingen.
    excel_file: optioneel bestandspad of file-like object; valt terug op EXCEL_PATH.
    """
    rijen = []

    uitgesloten = set(cfg.get('uitgesloten_componenten', []))
    ip_ov = cfg.get('ip_overrides', {})
    vp_ov = cfg.get('vp_overrides', {})
    lt_ov = cfg.get('lt_overrides', {})

    # ── Classificatie-koppeling ───────────────────────────────────────────
    # Als bpa_selectie.json bestaat: gebruik als whitelist (alleen die codes
    # uit de Excel) en haal LT-bron + classificatie-score op. Bij actieve
    # whitelist slaan we de BPA-hardfilter over zodat ALLE "Opnemen"-codes
    # zichtbaar worden (ook B/C-categorieën of items met lage VP).
    _cls = laad_classificatie_selectie()
    _cls_items = _cls.get('items', {})
    _gebruik_cls_whitelist = _cls.get('selectie_actief', bool(_cls_items))
    if _gebruik_cls_whitelist:
        print(f"  ✓ Classificatie-selectie actief: {len(_cls_items)} codes "
              f"(gegenereerd {_cls.get('gegenereerd')})")

    # 1. Excel-onderdelen — hardfilter is altijd uit; selectie loopt via classificatie.
    try:
        excel_parts = laad_excel_onderdelen(excel_file, skip_hard_filter=True)
    except Exception as e:
        print(f"  ⚠  Excel niet geladen: {e}")
        excel_parts = pd.DataFrame()

    for code, row in excel_parts.iterrows():
        if str(code) in uitgesloten:
            continue
        # Whitelist: sla over als niet in classificatie-selectie
        if _gebruik_cls_whitelist and str(code) not in _cls_items:
            continue
        n     = cfg['n_klanten_overrides'].get(str(code),
                                               _n_uit_klantlocaties(row.get('n_cust')))
        # VP komt uit Excel (standaard verkoopprijs) tenzij override; IP = VP / 2 tenzij override.
        vp    = vp_ov.get(str(code), float(row.get('VP', 0.0) or 0.0))
        ip    = ip_ov.get(str(code), vp / 2)
        # LT: configuratie-override telt als 'bevestigd' (gebruiker heeft 'm
        # zelf gezet); anders: bron volgens classificatie; anders Excel-waarde.
        if str(code) in lt_ov:
            lt_d    = lt_ov[str(code)]
            lt_bron = 'override'
        else:
            lt_d    = int(row['LT_days'])
            lt_bron = _cls_items.get(str(code), {}).get('lt_bron', 'onbekend')
        # LT=0 → vul aan met 30 dagen en markeer met aparte bron (blauwe cel in UI)
        if lt_d == 0 and lt_bron != 'override':
            lt_d = 30
            lt_bron = 'nul→30'
        lam   = _lambda_voor_rij(row, n)
        lt_jr = lt_d / 365
        mu    = lam * lt_jr
        stocks = {
            sl: BPAOptimizationModel.inverse_service_level(sl, lam, lt_jr)
            for sl in SERVICE_LEVELS
        }
        rij = {
            'Code':      code,
            'Descr':     str(row['Descr'])[:40],
            'n_klanten': n,
            'lambda_jr': round(lam, 4),
            'LT_dagen':  lt_d,
            'LT_bron':   lt_bron,
            'Cls_score': _cls_items.get(str(code), {}).get('score'),
            'IP':        round(ip, 2),
            'VP':        round(vp, 2),
            'mu':        round(mu, 4),
            'bron':      'excel',
        }
        for sl in SERVICE_LEVELS:
            rij[f's@{sl:.1%}'] = stocks[sl]
        rijen.append(rij)

    # ── Diagnostiek: classificatie-codes die niet in de BPA-Excel staan ──
    if _gebruik_cls_whitelist:
        _excel_codes = {str(c) for c in excel_parts.index} if not excel_parts.empty else set()
        _missing = [c for c in _cls_items.keys() if c not in _excel_codes and c not in uitgesloten]
        if _missing:
            print(f"  ⚠  {len(_missing)} classificatie-codes ontbreken in de BPA-Excel "
                  f"(eerste 5: {_missing[:5]}) — toegevoegd vanuit classificatie-metadata")

        # Bouw synthetische rijen vanuit de classificatie-payload (descr, ip, vp,
        # mtbf, totaal_orders_5jr, n_cust, lt_dagen). Zo verschijnen ALLE
        # "Opnemen in lijst"-codes in het overzicht.
        for code in _missing:
            cls_meta = _cls_items.get(code, {})
            n   = cfg['n_klanten_overrides'].get(
                code, _n_uit_klantlocaties(cls_meta.get('n_cust')))
            # VP komt uit classificatie-metadata tenzij override; IP = VP/2 (of override).
            vp_meta = cls_meta.get('vp')
            vp  = vp_ov.get(code, float(vp_meta) if vp_meta is not None else 0.0)
            ip  = ip_ov.get(code, vp / 2)

            # LT: override > classificatie > default 30
            if code in lt_ov:
                lt_d    = int(lt_ov[code])
                lt_bron = 'override'
            else:
                lt_raw  = cls_meta.get('lt_dagen')
                if lt_raw is None or int(lt_raw) == 0:
                    lt_d    = 30
                    lt_bron = 'nul→30'
                else:
                    lt_d    = int(lt_raw)
                    lt_bron = cls_meta.get('lt_bron', 'onbekend')

            # Lambda: MTBF heeft voorrang; bij ontbreken valt λ terug op
            # n / DEFAULT_MTBF_JR (= 10 jr). Geen fallback meer op historische
            # orders, zodat het gedrag consistent is met _lambda_voor_rij().
            mtbf = cls_meta.get('mtbf')
            if mtbf is not None and mtbf > 0:
                lam = n / mtbf
            else:
                lam = n / DEFAULT_MTBF_JR

            lt_jr = lt_d / 365
            mu    = lam * lt_jr
            stocks = {
                sl: BPAOptimizationModel.inverse_service_level(sl, lam, lt_jr)
                for sl in SERVICE_LEVELS
            }
            rij = {
                'Code':      code,
                'Descr':     str(cls_meta.get('descr', ''))[:40],
                'n_klanten': n,
                'lambda_jr': round(lam, 4),
                'LT_dagen':  lt_d,
                'LT_bron':   lt_bron,
                'Cls_score': cls_meta.get('score'),
                'IP':        round(ip, 2),
                'VP':        round(vp, 2),
                'mu':        round(mu, 4),
                'bron':      'classificatie',
            }
            for sl in SERVICE_LEVELS:
                rij[f's@{sl:.1%}'] = stocks[sl]
            rijen.append(rij)

    # 2. Handmatige componenten
    for code, hcomp in cfg['handmatige_componenten'].items():
        if code in uitgesloten:
            continue
        if _gebruik_cls_whitelist and str(code) not in _cls_items:
            continue
        lam   = hcomp['lambda_per_jaar']
        lt_d  = hcomp['lt_dagen']
        lt_jr = lt_d / 365
        mu    = lam * lt_jr
        n     = cfg['n_klanten_overrides'].get(str(code), hcomp.get('n_klanten', 1))
        ip    = hcomp.get('ip', 0.0)
        stocks = {
            sl: BPAOptimizationModel.inverse_service_level(sl, lam, lt_jr)
            for sl in SERVICE_LEVELS
        }
        rij = {
            'Code':      code,
            'Descr':     hcomp.get('descr', '')[:40],
            'n_klanten': n,
            'lambda_jr': round(lam, 4),
            'LT_dagen':  lt_d,
            'LT_bron':   'handmatig',
            'Cls_score': None,
            'IP':        round(ip, 2),
            'VP':        round(ip * 2, 2),
            'mu':        round(mu, 4),
            'bron':      'handmatig',
        }
        for sl in SERVICE_LEVELS:
            rij[f's@{sl:.1%}'] = stocks[sl]
        rijen.append(rij)

    if not rijen:
        return pd.DataFrame()
    overzicht = pd.DataFrame(rijen).set_index('Code')
    return overzicht[~overzicht.index.duplicated(keep='first')]


# ══════════════════════════════════════════════════════════════════════════════
#  KOSTENMODEL
# ══════════════════════════════════════════════════════════════════════════════

def bouw_model_kosten(
    overzicht_df: pd.DataFrame,
    alpha:         float,
    kappa_bpa:     float,
    kappa_c:       float,
    service_level: float,
    n_klanten_override=None,   # int | None – overschrijft alle n_klanten en schaalt lambda_jr
) -> tuple:
    """
    Bouwt een BPAOptimizationModel vanuit de overzicht DataFrame en geeft
    (model, result) terug.  Demand-keys: (part, customer).
    Elk component krijgt zijn eigen n_klanten; de klantenpool is de unie
    van alle klanten (grootte = max(n_klanten)).
    n_klanten_override: overschrijft n_klanten voor alle componenten en
    schaalt lambda_jr proportioneel (Λ_new = N_new · λ_per_cust).
    """
    if overzicht_df.empty:
        raise ValueError("overzicht_df is leeg – laad eerst het overzicht.")

    df = overzicht_df.copy()
    if n_klanten_override is not None:
        _n_new = int(n_klanten_override)
        df['lambda_jr'] = df.apply(
            lambda r: _n_new * (r['lambda_jr'] / r['n_klanten'])
            if r['n_klanten'] > 0 else r['lambda_jr'],
            axis=1,
        )
        df['n_klanten'] = _n_new

    n_max = int(df['n_klanten'].max())
    customers = {f'Klant_{i+1}': 1 for i in range(n_max)}

    demand_data: dict = {}
    for code, row in df.iterrows():
        n_i = int(row['n_klanten'])
        lam_per_cust = row['lambda_jr'] / n_i if n_i > 0 else row['lambda_jr']
        for i in range(n_max):
            demand_data[(code, f'Klant_{i+1}')] = lam_per_cust if i < n_i else 0.0

    vp_col = 'VP' if 'VP' in df.columns else 'IP'
    cost_data = {
        'purchase_price': {code: row['IP']      for code, row in df.iterrows()},
        'sales_price':    {code: row[vp_col]    for code, row in df.iterrows()},
        'lead_time':      {code: row['LT_dagen'] / 365 for code, row in df.iterrows()},
    }

    model = BPAOptimizationModel()
    model.add_customers_and_machines(customers)
    model.add_spare_parts(list(df.index))
    model.add_parameters(
        service_level=service_level,
        alpha=alpha,
        kappa_bpa=kappa_bpa,
        kappa_c=kappa_c,
        demand_data=demand_data,
        cost_data=cost_data,
    )
    return model, model.check_feasibility()


# ══════════════════════════════════════════════════════════════════════════════
#  SUBSCRIPTIE-SIMULATIE  (binomiale adoptie per component)
# ══════════════════════════════════════════════════════════════════════════════

def regionale_adoptie_parameter(
    p0: float,
    alpha: float, X: float,
    alpha0: float, X0: float,
    gamma_alpha: float = 1.0, gamma_X: float = 0.5,
    alpha_max: float = None, s_alpha: float = 0.02,
) -> float:
    """Regionale adoptieparameter p_r(α, X) volgens de logit-specificatie.

    Het baseline-niveau p0 = p_{r,0} geldt bij de referentie-instelling
    (α0, X0). Prijs en service level schuiven de adoptie op de logit-schaal:

        p_r(α, X) = σ( logit(p0)
                       − γ_α · (α − α0) / α0
                       + γ_X · [g(X) − g(X0)] )

    met σ(y) = 1/(1+e^{−y}),  logit(p) = ln(p/(1−p))  en  g(X) = −ln(1−X).

    Een hogere prijs (α > α0) verlaagt de adoptie; een hoger service level
    (X > X0) verhoogt de adoptie. Bij (α0, X0) geldt p_r = p0. Het resultaat
    blijft begrensd tussen 0 en 1.

    WTP-plafond op α (= α_U = κ_c)
    ------------------------------
    Een klant abonneert alleen als α·U_i^c ≤ κ_c·U_i^c, dus α ≤ κ_c (U_i^c
    valt weg). Als ``alpha_max`` (= κ_c) is opgegeven, wordt de logit-kans
    vermenigvuldigd met een gladde poort die de adoptie naar 0 brengt naarmate
    α → α_max:

        gate(α) = 1 − e^{−(α_max − α)/s}   voor α < α_max,   anders 0

    Dit is gate(α) = P(κ_{c,n} > α) met klant-specifieke breakeven
    κ_{c,n} = κ_max − E, E ~ Exp(gemiddelde s). De bovengrens α_max = κ_c is
    dus de maximale breakeven: bij α = α_max geldt gate = 0, zodat Z_i exact 0
    wordt zodra de bovengrens α_U bereikt wordt. Kleine s → bijna harde cutoff;
    grotere s → geleidelijker aflopen door spreiding in de klant-κ_c.
    """
    _eps = 1e-9
    p0 = float(min(max(p0, _eps), 1.0 - _eps))
    if alpha0 == 0:
        alpha0 = _eps

    def _g(x: float) -> float:
        x = float(min(max(x, _eps), 1.0 - _eps))
        return -np.log(1.0 - x)

    _logit_p0 = np.log(p0 / (1.0 - p0))
    _y = (
        _logit_p0
        - gamma_alpha * (alpha - alpha0) / alpha0
        + gamma_X * (_g(X) - _g(X0))
    )
    p = float(1.0 / (1.0 + np.exp(-_y)))

    # WTP-plafond: adoptie zakt glad naar 0 naarmate α de bovengrens α_U nadert.
    if alpha_max is not None:
        s = float(s_alpha)
        if alpha >= alpha_max:
            gate = 0.0
        elif s <= 0:
            gate = 1.0
        else:
            gate = 1.0 - np.exp(-(alpha_max - alpha) / s)
        p *= gate
    return float(p)


def adoptie_kans(alpha, kappa_c, q_eq, beta_r) -> float:
    """Globale logit-adoptiekans q(α) = σ(logit(q_eq) + η_r·ln(κ_c/α)).

    Discrete-keuze (logit) specificatie: een klant abonneert met kans
    q(α) = 1/(1+e^(-y)) waarin y = η_0 + η_r·ln(κ_c/α) en de intercept
    η_0 = logit(q_eq) = ln(q_eq/(1-q_eq)) geijkt is op kostenpariteit
    (bij α = κ_c geldt ln(κ_c/α) = 0 ⇒ q = q_eq). De kostenratio
    C_i^c/(α·v_i^c) ≈ κ_c/α (zelf-voorraadkosten ≈ κ_c·v_i^c). Eén globale q
    voor alle klanten (geen regio-onderscheid); het service level X zit in het
    onwaargenomen nut/η_0 en beïnvloedt q niet direct.

    Parameters
    ----------
    alpha   : prijspercentage α (> 0).
    kappa_c : bovengrens/kostenpariteit κ_c (uit de kostenanalyse).
    q_eq    : adoptiekans bij kostenpariteit (0 < q_eq < 1).
    beta_r  : gevoeligheid voor de kostenratio ln(κ_c/α).
    """
    eps = 1e-9
    q_eq    = float(min(max(q_eq, eps), 1.0 - eps))
    alpha   = max(float(alpha), eps)
    kappa_c = max(float(kappa_c), eps)
    beta_0  = np.log(q_eq / (1.0 - q_eq))
    y       = beta_0 + float(beta_r) * np.log(kappa_c / alpha)
    return float(1.0 / (1.0 + np.exp(-y)))


def aantal_klanten_per_component(excel_file=None, codes=None) -> pd.Series:
    """N_i = aantal verschillende historische klanten per component.

    Telt de unieke klanten per Code in de Adoptie-tab (één rij = één
    (component, klant)-combinatie). Beperkt tot ``codes`` of, bij None, tot de
    classificatie-selectie.

    Returns een Series geïndexeerd op Code met N_i (float).
    """
    adoptie = laad_adoptie_data(excel_file, rate_overrides=None)
    codes_set = {str(c).strip() for c in codes} if codes is not None else classificatie_codes()
    if codes_set:
        adoptie = adoptie[adoptie['Code'].isin(codes_set)]
    if adoptie.empty:
        return pd.Series(dtype=float)
    _n = adoptie.groupby('Code')['Klant'].nunique().astype(float)
    _n.index.name = 'Code'
    return _n


def binomiale_verdeling(N, q, k_min=None, k_max=None):
    """Kansverdeling (PMF) van de binomiale verdeling Binomiaal(N, q).

    Berekent P(Z = k) = C(N, k)·q^k·(1-q)^(N-k) numeriek stabiel via log-gamma
    (zonder externe afhankelijkheden zoals scipy). Standaard wordt alleen een
    relevant venster rond het gemiddelde (μ ± ~5σ) berekend zodat het ook voor
    grote N snel blijft.

    Dit is de verdeling achter de analytische verwachtingswaarde: per component
    geldt Z_i ~ Binomiaal(N_i, q(α)); omdat alle klanten dezelfde globale
    adoptiekans q delen is het totaal Z_tot ~ Binomiaal(Σ N_i, q).

    Parameters
    ----------
    N : aantal trials (historische klanten N_i, of Σ N_i voor het totaal).
    q : slaagkans per klant (de globale adoptiekans q(α)).
    k_min, k_max : optioneel expliciet k-bereik; anders automatisch (μ ± ~5σ).

    Returns (k_array (int), pmf_array (float)).
    """
    import math
    N = int(round(float(N)))
    q = float(min(max(float(q), 0.0), 1.0))
    if N <= 0:
        return np.array([0]), np.array([1.0])
    mean = N * q
    sd   = math.sqrt(max(N * q * (1.0 - q), 0.0))
    if k_min is None:
        k_min = max(0, int(math.floor(mean - 5.0 * sd - 1)))
    if k_max is None:
        k_max = min(N, int(math.ceil(mean + 5.0 * sd + 1)))
    if k_min > k_max:
        k_min, k_max = 0, N
    ks = np.arange(int(k_min), int(k_max) + 1)
    if q <= 0.0:
        return ks, np.where(ks == 0, 1.0, 0.0)
    if q >= 1.0:
        return ks, np.where(ks == N, 1.0, 0.0)
    _lg   = math.lgamma
    _lnq  = math.log(q)
    _ln1q = math.log(1.0 - q)
    _logc = _lg(N + 1)
    logpmf = np.array([
        _logc - _lg(k + 1) - _lg(N - k + 1) + k * _lnq + (N - k) * _ln1q
        for k in ks
    ], dtype=float)
    return ks, np.exp(logpmf)


def binomiale_quantile(N: int, q: float, level: float) -> int:
    """(level)-kwantiel van Binomiaal(N, q): kleinste z met P(Z ≤ z) ≥ level.

    Gebruikt voor de chance-constrained base-stock berekening:

        Z_i^{1-ε} = min{ z ∈ Z≥0 : P(Z_i ≤ z) ≥ 1-ε },
        Z_i ~ Binomiaal(M_i, q^s).

    Bij level = 0.5 geeft dit de mediaan (≈ gemiddelde voor grote N), bij
    level = 0.90 / 0.95 geeft dit de robuuste bovengrens waarbij het
    service level X met kans 1-ε ook bij hogere adoptie wordt gehaald.
    """
    import math
    N = int(round(float(N)))
    q = float(min(max(float(q), 0.0), 1.0))
    level = float(min(max(float(level), 0.0), 1.0))
    if N <= 0:
        return 0
    if q <= 0.0:
        return 0
    if q >= 1.0:
        return N
    _lg   = math.lgamma
    _lnq  = math.log(q)
    _ln1q = math.log(1.0 - q)
    _logc = _lg(N + 1)
    _cdf  = 0.0
    for _k in range(N + 1):
        _logp = _logc - _lg(_k + 1) - _lg(N - _k + 1) + _k * _lnq + (N - _k) * _ln1q
        _cdf += math.exp(_logp)
        if _cdf >= level:
            return _k
    return N


def verwacht_subscripties_per_component(
    excel_file = None,
    codes      = None,
    alpha:   float = 0.15,
    kappa_c: float = 0.25,
    q_eq:    float = 0.5,
    beta_r:  float = 1.0,
) -> pd.Series:
    """Verwacht aantal subscripties E[Z_i(α)] per component.

    Z_i(α) ~ Binomiaal(N_i, q(α)) met N_i = aantal verschillende historische
    klanten van component i en q(α) de globale logit-adoptiekans
    (:func:`adoptie_kans`). Dus E[Z_i(α)] = N_i · q(α). Eén globale q voor alle
    klanten (geen regio-onderscheid); q hangt alleen van α af via κ_c/α. Het
    service level X beïnvloedt de adoptie niet, alleen de voorraad/kostenkant.

    Returns een Series geïndexeerd op Code met het verwachte aantal subs.
    """
    _n = aantal_klanten_per_component(excel_file, codes)
    if _n.empty:
        return pd.Series(dtype=float)
    _q  = adoptie_kans(alpha, kappa_c, q_eq, beta_r)
    _ez = _n * _q
    _ez.index.name = 'Code'
    return _ez.sort_values(ascending=False)


def gevoeligheid_verwachte_z(
    waarden,
    parameter: str,
    alpha:   float,
    kappa_c: float,
    q_eq:    float,
    beta_r:  float,
    excel_file = None,
    codes      = None,
):
    """Totaal verwacht aantal subscripties Σ_i E[Z_i] als functie van één parameter.

    ``parameter`` ∈ {'alpha', 'kappa_c', 'q_eq', 'beta_r'}. Voor elke waarde in
    ``waarden`` wordt die parameter gevarieerd (de overige blijven vast) en het
    totaal Σ_i E[Z_i] = q(·) · Σ_i N_i berekend, met N_i het aantal historische
    klanten per component en q de globale logit-adoptiekans. De Adoptie-data
    wordt één keer geladen.

    Returns een lijst totalen, parallel aan ``waarden``.
    """
    _toegestaan = ('alpha', 'kappa_c', 'q_eq', 'beta_r')
    if parameter not in _toegestaan:
        raise ValueError(f"parameter moet een van {_toegestaan} zijn")
    _n = aantal_klanten_per_component(excel_file, codes)
    _n_tot = float(_n.sum())
    if _n_tot <= 0:
        return [0.0 for _ in waarden]
    _out = []
    for _v in waarden:
        _a  = float(_v) if parameter == 'alpha'   else float(alpha)
        _kc = float(_v) if parameter == 'kappa_c' else float(kappa_c)
        _qe = float(_v) if parameter == 'q_eq'    else float(q_eq)
        _br = float(_v) if parameter == 'beta_r'  else float(beta_r)
        _out.append(adoptie_kans(_a, _kc, _qe, _br) * _n_tot)
    return _out


def pareto_alpha_X(
    overzicht_df,
    alpha_waarden,
    X_waarden,
    q_eq:      float,
    beta_r:    float,
    kappa_bpa: float,
    kappa_c:   float,
    excel_file       = None,
    codes            = None,
    n_series         = None,
    epsilon: float   = None,
) -> pd.DataFrame:
    """Pareto-analyse over (α, X): BPA-marge vs. totaal klantsurplus.

    Voor elk paar (α, X) uit het cartesisch product van ``alpha_waarden`` en
    ``X_waarden`` wordt de keten doorgerekend:

      1. globale logit-adoptiekans q(α) = σ(logit(q_eq) + η_r·ln(κ_c/α));
      2. E[Z_i(α)] = N_i · q(α) per component (N_i = aantal historische
         klanten). X beïnvloedt de adoptie niet, alleen de voorraad/kosten;
      3. een overzicht met n_klanten = Z_i (λ proportioneel meegeschaald);
      4. het kostenmodel → BPA-marge en het totale klantsurplus
         Σ_klant (zelf-voorraadkosten − abonnementskosten).

    ``n_series`` : optioneel vooraf geladen N_i (Series op Code). Wordt gebruikt
    om herhaald inlezen van de Adoptie-tab te vermijden wanneer deze functie
    vaak wordt aangeroepen (bv. per η_r-rasterpunt in de onzekerheidsband).
    ``epsilon``  : geaccepteerde kans dat de service-belofte niet gehaald wordt
    door hogere adoptie dan verwacht. Wanneer opgegeven wordt de base-stock
    berekend op het (1-ε)-kwantiel Z_i^{1-ε} ~ Binomiaal(M_i, q), terwijl de
    omzet (revenue) op E[Z_i] = M_i·q blijft (klanten betalen ongeacht gebruik).
    Hiermee implementeert de keten Eq. (chance-constraint) uit de thesis:
    P(β_i(S_i; Z_i) ≥ X) ≥ 1-ε. Standaard None = gemiddelde benadering.

    Returns een DataFrame met kolommen:
        alpha, X, margin, surplus, total_Z, feasible, revenue, costs,
        stock_level (totale base-stock Σ_i S_i* over alle componenten).
    """
    if overzicht_df is None or overzicht_df.empty:
        return pd.DataFrame()

    _n = n_series if n_series is not None else aantal_klanten_per_component(excel_file, codes)
    if _n is None or _n.empty:
        return pd.DataFrame()

    # Beperk het overzicht tot componenten waarvoor adoptie-data beschikbaar
    # is. Componenten zonder data zouden anders een vaste (α-onafhankelijke)
    # bijdrage geven aan de marge, wat de curve en total_Z inconsistent maakt.
    base = overzicht_df.loc[overzicht_df.index.isin(_n.index)]
    if base.empty:
        return pd.DataFrame()
    base_n    = base['n_klanten'].astype(float)
    base_lam  = base['lambda_jr'].astype(float)
    lam_per_cust = (base_lam / base_n.replace(0, np.nan)).fillna(0.0)

    _n_base = _n.reindex(base.index).fillna(0.0)  # M_i uitgelijnd op base
    rijen = []
    for _a in alpha_waarden:
        # E[Z_i(α)] = M_i · q(α); q hangt alleen van α af (niet van X).
        _q       = adoptie_kans(_a, kappa_c, q_eq, beta_r)
        _ez      = _n_base * _q                      # E[Z_i] – voor omzet
        _total_z = float(_ez.sum())
        _n_int   = _ez.round().clip(lower=0).astype(int)
        # Chance-constrained stock: Z_i^{1-ε}-kwantiel per component.
        # Omzet blijft op E[Z_i]; alleen de stock wordt robuust berekend.
        # Invariant: CC kan stock alleen verhogen, nooit verlagen.
        # Voor N=1-componenten met q < ε geeft het kwantiel 0 terug; door
        # onderaan te clippen op E[Z_i] = _ez geldt altijd _z_cc ≥ _ez,
        # zodat margin_CC ≤ margin_no_CC bij vaste α en portfolio.
        if epsilon is not None:
            _level = 1.0 - float(epsilon)
            _z_cc = pd.Series(
                np.maximum(
                    [float(binomiale_quantile(int(round(float(_mi))), _q, _level))
                     for _mi in _n_base.values],
                    _ez.values,   # ondergrens: nooit minder dan verwachte waarde
                ),
                index=base.index, dtype=float,
            )
        else:
            _z_cc = _ez   # geen CC: stock op verwachte waarde
        for _x in X_waarden:
            mod = base.copy()
            mod['n_klanten'] = _n_int.values              # E[Z_i]: bepaalt omzet
            mod['lambda_jr'] = (_z_cc * lam_per_cust).values  # Z_cc of E[Z_i]: bepaalt stock
            if int(mod['n_klanten'].sum()) <= 0:
                _margin  = 0.0
                _surplus = 0.0
                _feas    = False
                _revenue = 0.0
                _costs   = 0.0
                _stock   = 0.0
            else:
                try:
                    _model, _res = bouw_model_kosten(mod, _a, kappa_bpa, kappa_c, _x)
                    _margin  = float(_res['bpa_margin'])
                    _surplus = float(sum(
                        v['savings'] for v in _res['customer_benefits'].values()))
                    _feas    = bool(_res['feasible'])
                    _revenue = float(_res['total_revenue'])
                    _costs   = float(_res['bpa_costs'])
                    _stock   = float(sum(_model.calculate_base_stock_levels().values()))
                except Exception:
                    _margin = _surplus = float('nan')
                    _feas   = False
                    _revenue = _costs = _stock = float('nan')
            rijen.append({
                'alpha':       float(_a),
                'X':           float(_x),
                'margin':      _margin,
                'surplus':     _surplus,
                'total_Z':     _total_z,
                'feasible':    _feas,
                'revenue':     _revenue,
                'costs':       _costs,
                'stock_level': _stock,
            })
    return pd.DataFrame(rijen)


def metrieken_voor_wtp_grid(
    overzicht_df,
    param_dicts,
    kappa_bpa: float,
    kappa_c:   float,
    excel_file       = None,
    codes            = None,
    budget: float    = None,
):
    """Meerdere keten-uitkomsten per parametercombinatie (α, X, q_eq, η_r).

    Generaliseert de keten van ``pareto_alpha_X``: elke dict in ``param_dicts``
    bevat de sleutels ``alpha``, ``X``, ``q_eq``, ``beta_r`` en optioneel
    ``kappa_c`` (anders de meegegeven ``kappa_c``). Per combinatie wordt de
    globale logit-adoptiekans q(α), daaruit E[Z_i] = N_i·q(α), en via het
    kostenmodel de BPA-marge en het totale klantsurplus berekend.

    Returns een lijst dicts (parallel aan ``param_dicts``), elk met sleutels:
        ``bpa_margin``  – totale BPA-marge (€);
        ``surplus``     – totaal klantsurplus (€);
        ``total_Z``     – verwacht totaal aantal subscripties E[Z];
        ``q``           – globale adoptiekans q(α);
        ``feasible``    – of het kostenmodel haalbaar was.
    NaN bij een niet-doorgerekende combinatie.
    """
    def _leeg():
        return {
            'bpa_margin': float('nan'), 'surplus': float('nan'),
            'total_Z': float('nan'), 'q': float('nan'), 'feasible': False,
            'revenue': float('nan'), 'costs': float('nan'),
            'stock_level': float('nan'), 'inv_total': float('nan'),
        }

    if overzicht_df is None or overzicht_df.empty or not param_dicts:
        return [_leeg() for _ in (param_dicts or [])]

    _n = aantal_klanten_per_component(excel_file, codes)
    if _n.empty:
        return [_leeg() for _ in param_dicts]

    # Beperk het overzicht tot componenten waarvoor adoptie-data beschikbaar is.
    base = overzicht_df.loc[overzicht_df.index.isin(_n.index)]
    if base.empty:
        return [_leeg() for _ in param_dicts]
    base_n    = base['n_klanten'].astype(float)
    base_lam  = base['lambda_jr'].astype(float)
    lam_per_cust = (base_lam / base_n.replace(0, np.nan)).fillna(0.0)

    resultaten = []
    for p in param_dicts:
        _a  = float(p['alpha'])
        _x  = float(p['X'])
        _qe = float(p['q_eq'])
        _br = float(p['beta_r'])
        _kc      = float(p.get('kappa_c', kappa_c))
        _kb      = float(p.get('kappa_bpa', kappa_bpa))
        _ip_mult = float(p.get('ip_mult', 1.0))
        _n_mult  = float(p.get('n_mult', 1.0))
        _lt_mult = float(p.get('lt_mult', 1.0))
        _q       = adoptie_kans(_a, _kc, _qe, _br)
        # Effective customer base: scale M_i by n_mult (default 1.0 = unchanged).
        _n_eff   = (_n * _n_mult).reindex(base.index).fillna(0.0)
        _ez      = _n_eff * _q
        _total_z = float(_ez.sum())

        _bgt = p.get('budget', budget)
        if _bgt is not None:
            _bgt = float(_bgt)
        if _bgt is not None:
            # Greedy modus: run greedy voor deze (α, X, q_eq, η_r)-combinatie.
            _eps = p.get('epsilon', None)
            if _eps is not None and float(_eps) <= 0:
                _eps = None
            _ov_mod = overzicht_df
            if _ip_mult != 1.0:
                _ov_mod = _ov_mod.assign(IP=_ov_mod['IP'] * _ip_mult)
            if _lt_mult != 1.0:
                _ov_mod = _ov_mod.assign(LT_dagen=_ov_mod['LT_dagen'] * _lt_mult)
            _n_ser = (_n * _n_mult) if _n_mult != 1.0 else _n
            _gs = greedy_alpha_sweep(
                _ov_mod, [_a], _bgt, _x,
                _qe, _br, float(_kb), _kc,
                n_series=_n_ser,
                epsilon=_eps,
            )
            if _gs.empty:
                resultaten.append({**_leeg(), 'total_Z': _total_z, 'q': float(_q)})
            else:
                _row = _gs.iloc[0]
                resultaten.append({
                    'bpa_margin': float(_row['total_margin']),
                    'surplus':    float('nan'),   # niet beschikbaar bij greedy
                    'total_Z':    float(_row['total_Z']),
                    'q':          float(_row['q']),
                    'feasible':   float(_row['total_margin']) >= 0,
                    'revenue':    float(_row.get('total_rev',   float('nan'))),
                    'costs':      float(_row.get('total_cbpa',  float('nan'))),
                    'stock_level': float(_row.get('total_stock', float('nan'))),
                    'inv_total':  float(_row.get('total_inv',   float('nan'))),
                })
            continue

        # n_klanten voor modelstructuur: ceil zodat componenten met E[Z_i]>0
        # nooit worden weggerond naar 0. Omzet/marge worden post-hoc berekend
        # met continue _ez zodat afronden de financiële uitkomsten niet verstoort.
        _n_ceil = np.ceil(_ez).clip(lower=0).astype(int)
        # Chance-constrained stock (spiegelt pareto_alpha_X):
        # omzet op E[Z_i], stock op (1-ε)-kwantiel Z_i^{1-ε}.
        # Uses _n_eff (already scaled by n_mult) as the M_i population.
        _eps_v = p.get('epsilon', None)
        if _eps_v is not None and float(_eps_v) > 0:
            _level = 1.0 - float(_eps_v)
            _z_cc = pd.Series(
                np.maximum(
                    [float(binomiale_quantile(int(round(float(_mi))), float(_q), _level))
                     for _mi in _n_eff.values],
                    _ez.values,
                ),
                index=base.index, dtype=float,
            )
        else:
            _z_cc = _ez  # geen CC: stock op verwachte waarde
        mod = base.copy()
        mod['n_klanten'] = _n_ceil.values
        mod['lambda_jr'] = (_z_cc * lam_per_cust).values  # Z_cc bepaalt stock
        if _ip_mult != 1.0:
            mod['IP'] = base['IP'].values * _ip_mult
        if _lt_mult != 1.0:
            mod['LT_dagen'] = base['LT_dagen'].values * _lt_mult
        # Continue E[Z_i]-waarden voor omzetberekening (nooit afronden).
        _ez_mod  = _ez.reindex(mod.index).fillna(0.0)
        _vp_col  = 'VP' if 'VP' in mod.columns else 'IP'
        _rec = {'total_Z': _total_z, 'q': float(_q)}
        if _ez_mod.sum() <= 0:
            _rec.update({'bpa_margin': 0.0, 'surplus': 0.0, 'feasible': False,
                         'revenue': 0.0, 'costs': 0.0, 'stock_level': 0.0, 'inv_total': 0.0})
        else:
            try:
                _model, _res = bouw_model_kosten(mod, _a, _kb, _kc, _x)
                _stk     = _model.calculate_base_stock_levels()
                _stk_tot = float(sum(_stk.values()))
                _inv_tot = float(sum(
                    _stk.get(code, 0) * float(mod.at[code, 'IP'])
                    for code in mod.index
                ))
                # Post-hoc omzet/marge met continue E[Z_i] (geen afrondingsfout).
                _rev_cont    = float((_ez_mod * _a * mod[_vp_col]).sum())
                _costs_cont  = float(_res.get('bpa_costs', float('nan')))
                _margin_cont = (_rev_cont - _costs_cont
                                if np.isfinite(_costs_cont) else float('nan'))
                _rec.update({
                    'bpa_margin':  _margin_cont,
                    'surplus':     float(sum(
                        v['savings'] for v in _res['customer_benefits'].values())),
                    'feasible':    (_margin_cont >= 0
                                    if np.isfinite(_margin_cont) else False),
                    'revenue':     _rev_cont,
                    'costs':       _costs_cont,
                    'stock_level': _stk_tot,
                    'inv_total':   _inv_tot,
                })
            except Exception:
                _rec.update({'bpa_margin': float('nan'), 'surplus': float('nan'),
                             'feasible': False, 'revenue': float('nan'),
                             'costs': float('nan'), 'stock_level': float('nan'),
                             'inv_total': float('nan')})
        resultaten.append(_rec)
    return resultaten


def winst_voor_wtp_grid(
    overzicht_df,
    param_dicts,
    kappa_bpa: float,
    kappa_c:   float,
    excel_file       = None,
    codes            = None,
):
    """Totale BPA-marge (€) voor een lijst WTP-parametercombinaties.

    Dunne wrapper rond :func:`metrieken_voor_wtp_grid` die alleen de
    BPA-marge per parameter-dict teruggeeft (lijst van floats, parallel aan
    ``param_dicts``; NaN bij een niet-doorgerekende combinatie).
    """
    _recs = metrieken_voor_wtp_grid(
        overzicht_df, param_dicts, kappa_bpa, kappa_c,
        excel_file=excel_file, codes=codes,
    )
    return [r.get('bpa_margin', float('nan')) for r in _recs]


def optimale_alpha_bij_X(
    overzicht_df,
    X_fix:     float,
    alpha_grid,
    q_eq:      float,
    beta_r:    float,
    kappa_bpa: float,
    kappa_c:   float,
    excel_file            = None,
    codes                 = None,
    alleen_haalbaar: bool = False,
    epsilon: float        = None,
    margin_ratio_min: float = None,
):
    """Zoek het prijspercentage α dat de BPA-marge maximaliseert bij vast X.

    Houdt het service level X vast op ``X_fix`` en evalueert de keten
    α → q(α) → E[Z_i] → kostenmodel voor elke α in ``alpha_grid``. Omdat een
    hogere α de omzet per klant verhoogt maar de adoptie q(α) verlaagt, bestaat
    doorgaans een inwendig marge-optimum.

    Parameters
    ----------
    alleen_haalbaar : als True wordt het optimum alleen gezocht onder de
                      haalbare combinaties (marge ≥ 0 én alle klanten
                      profiteren); bij geen enkele haalbare α valt de functie
                      terug op de marge-maximalisatie over alle α.

    Returns (curve_df, best_row).
    """
    curve = pareto_alpha_X(
        overzicht_df, list(alpha_grid), [float(X_fix)],
        q_eq, beta_r, kappa_bpa, kappa_c,
        excel_file=excel_file, codes=codes,
        epsilon=epsilon,
    )
    if curve.empty:
        return curve, None
    _geldig = curve.dropna(subset=['margin'])
    if _geldig.empty:
        return curve, None
    _kandidaten = _geldig
    if alleen_haalbaar:
        _haalbaar = _geldig[_geldig['feasible']]
        if not _haalbaar.empty:
            _kandidaten = _haalbaar
    if margin_ratio_min is not None and float(margin_ratio_min) > 0:
        if 'revenue' in _kandidaten.columns:
            _rev = _kandidaten['revenue'].fillna(0.0)
            _mrm = (_rev > 0) & (
                _kandidaten['margin'] / _rev >= float(margin_ratio_min))
            if _mrm.any():
                _kandidaten = _kandidaten[_mrm]
    best = _kandidaten.loc[_kandidaten['margin'].idxmax()]
    return curve, best


def beta_r_winstband(
    overzicht_df,
    X_fix:      float,
    alpha_grid,
    q_eq:       float,
    beta_r_min: float,
    beta_r_max: float,
    kappa_bpa:  float,
    kappa_c:    float,
    n_samples:  int = 200,
    excel_file        = None,
    codes             = None,
    seed              = None,
    alleen_haalbaar:  bool  = False,
    percentielen           = (5, 50, 95),
    epsilon: float         = None,
    budget: float          = None,
    margin_ratio_min: float = None,
):
    """η_r-parameteronzekerheid: bandbreedte voor winst-vs-α en optimale α.

    η_r kan niet uit historische subscriptie-data worden geschat. Het is
    **geen kansverdeling** (er is geen kansmodel over de aannemelijkheid van
    een bepaalde waarde) maar een **onzekere parameter met een plausibel
    bereik** [beta_r_min, beta_r_max]. Om de gevoeligheid van de uitkomst
    voor deze onzekerheid te tonen, wordt het volledige bereik **deterministisch
    en gelijkmatig doorlopen** via een raster (geen willekeurige trekkingen,
    geen aanname over een onderliggende verdeling)::

        η_r^(k) = beta_r_min + k·(beta_r_max − beta_r_min)/(n_samples − 1),
        k = 0 … n_samples − 1.

    Voor elk rasterpunt wordt (bij vast service level ``X_fix``) de volledige
    keten α → q(α) → E[Z_i] → kostenmodel over ``alpha_grid`` doorgerekend
    (via :func:`pareto_alpha_X`). Dit levert per α een bandbreedte van de
    verwachte BPA-winst E[Π_BPA(α)] over het η_r-bereik en per rasterpunt een
    winst-maximaliserende α*. De N_i worden één keer geladen en over alle
    rasterpunten hergebruikt.

    Parameters
    ----------
    n_samples : aantal η_r-rasterpunten (K) waarmee het bereik gelijkmatig
                wordt doorlopen.
    seed      : ongebruikt (geen willekeurige trekkingen meer); alleen nog
                aanwezig voor achterwaartse compatibiliteit van de signatuur.
    alleen_haalbaar : bepaal α* per rasterpunt alleen over haalbare α
                      (marge ≥ 0 én alle klanten profiteren), met terugval op
                      alle α.
    percentielen : welke posities over het doorlopen η_r-bereik te
                   rapporteren (default 5/50/95; 0/100 geven exact het
                   minimum/maximum van de band).

    Returns
    -------
    dict met sleutels:
        alpha_grid     : np.array van gebruikte α-waarden;
        beta_r_samples : np.array van gelijkmatig verdeelde η_r-rasterwaarden
                         (deterministisch, geen trekkingen);
        margin_matrix  : (n_samples × len(alpha_grid)) marges E[Π_BPA];
        margin_pct     : {p: array over α} band van de winst over het
                         doorlopen η_r-bereik (p=0/100 = min/max);
        margin_mean    : np.array gemiddelde winst per α over het raster;
        opt_alpha      : np.array optimale α* per rasterpunt;
        opt_margin     : np.array optimale winst per rasterpunt;
        opt_alpha_pct  : {p: float} band van de optimale α* over het bereik;
        opt_margin_pct : {p: float} band van de optimale winst over het bereik.
    Of ``None`` wanneer er geen data/overzicht beschikbaar is.
    """
    _alpha = np.asarray(list(alpha_grid), dtype=float)
    if overzicht_df is None or overzicht_df.empty or _alpha.size == 0:
        return None
    if float(beta_r_max) < float(beta_r_min):
        beta_r_min, beta_r_max = beta_r_max, beta_r_min

    # N_i één keer laden (dure Excel-read) en over alle rasterpunten hergebruiken.
    _n = aantal_klanten_per_component(excel_file, codes)
    if _n is None or _n.empty:
        return None

    # Deterministisch, gelijkmatig raster over het η_r-bereik — η_r is een
    # onzekere parameter met een gedefinieerd bereik, geen kansverdeling, dus
    # geen willekeurige trekkingen (``seed`` wordt niet meer gebruikt).
    _br_samples = np.linspace(float(beta_r_min), float(beta_r_max), int(n_samples))

    _margin  = np.full((int(n_samples), _alpha.size), np.nan)
    _revenue = np.full((int(n_samples), _alpha.size), np.nan)  # R per (grid-punt, α)
    _opt_a   = np.full(int(n_samples), np.nan)
    _opt_m   = np.full(int(n_samples), np.nan)

    for _k, _br in enumerate(_br_samples):
        if budget is not None:
            # Greedy modus: per trekking de volledige greedy uitvoeren.
            _gs = greedy_alpha_sweep(
                overzicht_df, list(_alpha), float(budget),
                float(X_fix), float(q_eq), float(_br),
                float(kappa_bpa), float(kappa_c),
                n_series=_n,
                epsilon=epsilon,
            )
            if _gs.empty:
                continue
            _m_series = _gs.set_index('alpha')['total_margin']
            _margin[_k, :] = _m_series.reindex(_alpha).to_numpy(dtype=float)
            if 'total_rev' in _gs.columns:
                _revenue[_k, :] = _gs.set_index('alpha')['total_rev'].reindex(_alpha).to_numpy(dtype=float)
            _valid = _gs.dropna(subset=['total_margin'])
            if _valid.empty:
                continue
            _kandidaten = _valid
            if alleen_haalbaar:
                _pos = _valid[_valid['total_margin'] > 0]
                if not _pos.empty:
                    _kandidaten = _pos
            if margin_ratio_min is not None and float(margin_ratio_min) > 0:
                if 'total_rev' in _kandidaten.columns:
                    _rev = _kandidaten['total_rev'].fillna(0.0)
                    _mrm = (_rev > 0) & (
                        _kandidaten['total_margin'] / _rev
                        >= float(margin_ratio_min))
                    if _mrm.any():
                        _kandidaten = _kandidaten[_mrm]
            _best = _kandidaten.loc[_kandidaten['total_margin'].idxmax()]
            _opt_a[_k] = float(_best['alpha'])
            _opt_m[_k] = float(_best['total_margin'])
        else:
            _curve = pareto_alpha_X(
                overzicht_df, list(_alpha), [float(X_fix)],
                float(q_eq), float(_br), float(kappa_bpa), float(kappa_c),
                codes=codes, n_series=_n,
                epsilon=epsilon,
            )
            if _curve is None or _curve.empty:
                continue
            # Uitlijnen op _alpha (identieke floats uit dezelfde grid).
            _m_series = _curve.set_index('alpha')['margin']
            _margin[_k, :] = _m_series.reindex(_alpha).to_numpy(dtype=float)
            if 'revenue' in _curve.columns:
                _revenue[_k, :] = _curve.set_index('alpha')['revenue'].reindex(_alpha).to_numpy(dtype=float)
            _valid = _curve.dropna(subset=['margin'])
            if _valid.empty:
                continue
            _kandidaten = _valid
            if alleen_haalbaar:
                _haalbaar = _valid[_valid['feasible']]
                if not _haalbaar.empty:
                    _kandidaten = _haalbaar
            if margin_ratio_min is not None and float(margin_ratio_min) > 0:
                if 'revenue' in _kandidaten.columns:
                    _rev = _kandidaten['revenue'].fillna(0.0)
                    _mrm = (_rev > 0) & (
                        _kandidaten['margin'] / _rev
                        >= float(margin_ratio_min))
                    if _mrm.any():
                        _kandidaten = _kandidaten[_mrm]
            _best = _kandidaten.loc[_kandidaten['margin'].idxmax()]
            _opt_a[_k] = float(_best['alpha'])
            _opt_m[_k] = float(_best['margin'])

    # Percentiel-band alleen berekenen wanneer er ten minste één geldige rij is.
    def _safe_pct(arr2d, p):
        _col_ok = ~np.all(np.isnan(arr2d), axis=0)
        _out = np.full(arr2d.shape[1], np.nan)
        if _col_ok.any():
            _out[_col_ok] = np.nanpercentile(arr2d[:, _col_ok], p, axis=0)
        return _out

    _margin_pct  = {int(p): _safe_pct(_margin, p) for p in percentielen}
    _revenue_pct = {int(p): _safe_pct(_revenue, p) for p in percentielen}
    with np.errstate(invalid='ignore'):
        _margin_mean = np.where(
            np.all(np.isnan(_margin), axis=0), np.nan, np.nanmean(_margin, axis=0))

    _opt_valid = _opt_a[~np.isnan(_opt_a)]
    _optm_valid = _opt_m[~np.isnan(_opt_m)]
    _opt_a_pct = {
        int(p): (float(np.percentile(_opt_valid, p)) if _opt_valid.size else float('nan'))
        for p in percentielen
    }
    _opt_m_pct = {
        int(p): (float(np.percentile(_optm_valid, p)) if _optm_valid.size else float('nan'))
        for p in percentielen
    }

    return {
        'alpha_grid':     _alpha,
        'beta_r_samples': _br_samples,
        'margin_matrix':  _margin,
        'margin_pct':     _margin_pct,
        'margin_mean':    _margin_mean,
        'revenue_pct':    _revenue_pct,
        'opt_alpha':      _opt_a,
        'opt_margin':     _opt_m,
        'opt_alpha_pct':  _opt_a_pct,
        'opt_margin_pct': _opt_m_pct,
        'budget':         budget,
    }


def greedy_alpha_sweep(
    overzicht_df,
    alpha_grid,
    budget: float,
    service_level: float,
    q_eq: float,
    beta_r: float,
    kappa_bpa: float,
    kappa_c: float,
    excel_file=None,
    codes=None,
    n_series=None,
    epsilon: float = None,
) -> pd.DataFrame:
    """Adoptie-bewuste greedy budget-selectie voor elk α in alpha_grid.

    De bestaande greedy-tab gebruikt een vaste Z (uit het overzicht) en een
    vaste α. Dat is niet consistent met het adoptiemodel: zowel de revenue
    (Z_i·α·VP_i) als de investering (S_i*(X, Λ_i(α))·IP_i) hangen af van α
    via de adoptiekans q(α). Een hogere α verlaagt q(α) → minder abonnees →
    lagere Λ_i → lagere S_i* én lagere Inv_i, maar ook lagere omzet per α-punt.
    De greedy-rangschikking (ROI = Margin_i/Inv_i) en dus de geselecteerde set
    kunnen daardoor per α verschillen.

    Deze functie herschrijft voor elk α de volledige keten::

        q(α) → Z_i(α) → S_i*(X, Z_i·λ/N·L) → Inv_i / Rev_i / Margin_i → greedy

    en laat zo zien hoe de geselecteerde portfolio én de totale marge van de
    greedy-selectie variëren met α (en impliciet met η_r).

    Parameters
    ----------
    budget        : maximaal investeringsbudget (€).
    service_level : doel-fillrate X voor S_i*-berekening via Poisson-inverse.
    q_eq          : adoptiekans bij kostenpariteit (logit-intercept).
    beta_r        : kostenratio-gevoeligheid (vaste waarde; voor een band over
                    η_r run deze functie meerdere keren).
    epsilon       : kans-constraint niveau (None = verwachte Z_i; anders
                    CC-kwantiel Z_i^{1-ε}).

    Returns
    -------
    DataFrame met per α-waarde:
        alpha, q, total_Z, total_inv, total_stock, total_rev, total_margin,
        total_cbpa, n_selected, n_total.
    Leeg DataFrame als er geen data of overzicht beschikbaar is.
    """
    if overzicht_df is None or overzicht_df.empty:
        return pd.DataFrame()
    _alpha_arr = np.asarray(list(alpha_grid), dtype=float)
    if _alpha_arr.size == 0:
        return pd.DataFrame()

    # N_i laden (eenmalig)
    _n = n_series if n_series is not None else aantal_klanten_per_component(excel_file, codes)
    if _n is None or _n.empty:
        # Fallback: gebruik n_klanten uit het overzicht als vaste M_i.
        base = overzicht_df.copy()
        _n_base_v = base['n_klanten'].astype(float).values
    else:
        base = overzicht_df.loc[overzicht_df.index.isin(_n.index)].copy()
        if base.empty:
            return pd.DataFrame()
        _n_base_v = _n.reindex(base.index).fillna(0.0).values

    _base_n   = base['n_klanten'].astype(float).replace(0, np.nan)
    _base_lam = base['lambda_jr'].astype(float)
    _lam_pc   = (_base_lam / _base_n).fillna(0.0).values  # λ per klant
    _lt_yr    = base['LT_dagen'].astype(float).values / 365.0
    _ip       = base['IP'].astype(float).values
    _vp_col   = 'VP' if 'VP' in base.columns else 'IP'
    _vp       = base[_vp_col].astype(float).values
    _N        = len(base)

    rijen = []
    for _a in _alpha_arr:
        _q = adoptie_kans(float(_a), float(kappa_c), float(q_eq), float(beta_r))

        # E[Z_i] = M_i · q — altijd de grondslag voor omzet en total_Z.
        _z_mean = _n_base_v * _q

        # Z_i voor stock-sizing: kwantiel bij CC, anders identiek aan mean.
        # Invariant: CC kan stock alleen verhogen, nooit verlagen.
        # Voor N=1-componenten met q < ε geeft het kwantiel 0 terug, wat de
        # investering elimineert en de marge kunstmatig opblaast. Door het
        # kwantiel onderaan te clippen op E[Z_i] geldt altijd:
        #   _z_stock ≥ _z_mean  →  costs_CC ≥ costs_no_CC  →  margin_CC ≤ margin_no_CC.
        if epsilon is not None:
            _lvl = 1.0 - float(epsilon)
            _z_stock = np.maximum(
                np.array([
                    float(binomiale_quantile(int(round(float(_n_base_v[_j]))), _q, _lvl))
                    for _j in range(_N)
                ], dtype=float),
                _z_mean,   # ondergrens: nooit minder dan verwachte waarde
            )
        else:
            _z_stock = _z_mean

        # S_i* per component via Poisson inverse (stock-sizing op _z_stock)
        _s_star = np.array([
            float(BPAOptimizationModel.inverse_service_level(
                float(service_level),
                float(_z_stock[_j]) * float(_lam_pc[_j]),
                float(_lt_yr[_j]),
            ))
            for _j in range(_N)
        ], dtype=float)

        _inv_v    = _s_star * _ip
        _rev_v    = _z_mean * float(_a) * _vp          # omzet op E[Z_i]
        _cbpa_v   = float(kappa_bpa) * _ip * _s_star   # kosten op CC-stock
        _margin_v = _rev_v - _cbpa_v
        _roi_v    = np.where(_inv_v > 0, _margin_v / _inv_v, np.inf)

        # Greedy: primair ROI desc, secundair inv asc (stabiel, idem UI-logica)
        _order = np.lexsort((_inv_v, -_roi_v))
        _cum   = np.cumsum(_inv_v[_order])
        _sel   = np.zeros(_N, dtype=bool)
        _sel[_order[_cum <= float(budget)]] = True
        # Backfill: kleinere items die nog passen
        _rest = float(budget) - float(_inv_v[_sel].sum())
        for _pos in _order:
            if not _sel[_pos] and float(_inv_v[_pos]) <= _rest:
                _sel[_pos] = True
                _rest -= float(_inv_v[_pos])

        rijen.append({
            'alpha':          float(_a),
            'q':              float(_q),
            'total_Z':        float(_z_mean.sum()),   # verwachte abonnees (niet kwantiel)
            'total_inv':      float(_inv_v[_sel].sum()),
            'total_stock':    float(_s_star[_sel].sum()),  # Σ S_i* (units) van de geselecteerde set
            'total_rev':      float(_rev_v[_sel].sum()),
            'total_margin':   float(_margin_v[_sel].sum()),
            'total_cbpa':     float(_cbpa_v[_sel].sum()),
            'n_selected':     int(_sel.sum()),
            'n_total':        _N,
            'selected_codes': list(base.index[_sel]),
        })

    return pd.DataFrame(rijen)


def greedy_detail_for_params(
    overzicht_df,
    alpha: float,
    service_level: float,
    q_eq: float,
    beta_r: float,
    kappa_bpa: float,
    kappa_c: float,
    budget: float,
    n_series=None,
    excel_file=None,
    codes=None,
    epsilon: float = None,
) -> pd.DataFrame:
    """Per-component greedy detail voor één parametercombinatie.

    Geeft een DataFrame terug met per component:
        code, omschrijving, Z_i (verwachte abonnees), S* (voorraadniveau),
        Inv (€), Rev (€), Marge (€), ROI, geselecteerd (bool).

    Gesorteerd: geselecteerde componenten eerst, daarna op ROI aflopend.
    """
    if overzicht_df is None or overzicht_df.empty:
        return pd.DataFrame()

    _n = n_series if n_series is not None else aantal_klanten_per_component(excel_file, codes)
    if _n is None or _n.empty:
        base = overzicht_df.copy()
        _n_base_v = base['n_klanten'].astype(float).values
    else:
        base = overzicht_df.loc[overzicht_df.index.isin(_n.index)].copy()
        if base.empty:
            return pd.DataFrame()
        _n_base_v = _n.reindex(base.index).fillna(0.0).values

    _base_n   = base['n_klanten'].astype(float).replace(0, np.nan)
    _base_lam = base['lambda_jr'].astype(float)
    _lam_pc   = (_base_lam / _base_n).fillna(0.0).values
    _lt_yr    = base['LT_dagen'].astype(float).values / 365.0
    _ip       = base['IP'].astype(float).values
    _vp_col   = 'VP' if 'VP' in base.columns else 'IP'
    _vp       = base[_vp_col].astype(float).values
    _N        = len(base)

    _q      = adoptie_kans(float(alpha), float(kappa_c), float(q_eq), float(beta_r))
    _z_mean = _n_base_v * _q

    if epsilon is not None:
        _lvl     = 1.0 - float(epsilon)
        _z_stock = np.maximum(
            np.array([
                float(binomiale_quantile(int(round(float(_n_base_v[_j]))), _q, _lvl))
                for _j in range(_N)
            ], dtype=float),
            _z_mean,
        )
    else:
        _z_stock = _z_mean

    _s_star = np.array([
        float(BPAOptimizationModel.inverse_service_level(
            float(service_level),
            float(_z_stock[_j]) * float(_lam_pc[_j]),
            float(_lt_yr[_j]),
        ))
        for _j in range(_N)
    ], dtype=float)

    _inv_v    = _s_star * _ip
    _rev_v    = _z_mean * float(alpha) * _vp
    _cbpa_v   = float(kappa_bpa) * _ip * _s_star
    _margin_v = _rev_v - _cbpa_v
    _roi_v    = np.where(_inv_v > 0, _margin_v / _inv_v, np.inf)

    # Greedy selectie (zelfde logica als greedy_alpha_sweep)
    _order = np.lexsort((_inv_v, -_roi_v))
    _cum   = np.cumsum(_inv_v[_order])
    _sel   = np.zeros(_N, dtype=bool)
    _sel[_order[_cum <= float(budget)]] = True
    _rest = float(budget) - float(_inv_v[_sel].sum())
    for _pos in _order:
        if not _sel[_pos] and float(_inv_v[_pos]) <= _rest:
            _sel[_pos] = True
            _rest -= float(_inv_v[_pos])

    _omsch_col = next(
        (c for c in ('omschrijving', 'Omschrijving', 'description') if c in base.columns),
        None,
    )
    records = []
    for _j, _code in enumerate(base.index):
        records.append({
            'code':         str(_code),
            'omschrijving': str(base[_omsch_col].iloc[_j]) if _omsch_col else '',
            'Z_i':          round(float(_z_mean[_j]), 2),
            'S*':           int(_s_star[_j]),
            'Inv (€)':      round(float(_inv_v[_j]), 2),
            'Rev (€)':      round(float(_rev_v[_j]), 2),
            'Marge (€)':    round(float(_margin_v[_j]), 2),
            'ROI':          (round(float(_roi_v[_j]), 4)
                             if np.isfinite(_roi_v[_j]) else float('inf')),
            'geselecteerd': bool(_sel[_j]),
        })

    _df = pd.DataFrame(records)
    _df = _df.sort_values(
        ['geselecteerd', 'ROI'], ascending=[False, False]
    ).reset_index(drop=True)
    return _df


def _hermap_adoption_rates(rates: pd.Series, rate_overrides) -> pd.Series:
    """Hermap de twee adoption-rate niveaus (Benelux hoog / overig laag).

    De data bevat per (component, klant) een vooraf bepaalde adoption rate met
    twee niveaus: een hoog niveau voor Benelux-klanten en een laag niveau voor
    de overige klanten. Met deze functie kunnen die twee niveaus in de tool
    worden overschreven zonder de data opnieuw te genereren.

    rate_overrides : dict met optionele sleutels 'benelux' en/of 'overig'
                     (waarden tussen 0 en 1). De twee oorspronkelijke niveaus
                     worden gedetecteerd als de twee meest voorkomende positieve
                     waarden; rijen op het hoge niveau krijgen de Benelux-rate,
                     rijen op het lage niveau de overig-rate. Nul-waarden (geen
                     data) blijven ongemoeid.
    """
    if not rate_overrides:
        return rates
    p_ben = rate_overrides.get('benelux')
    p_ov  = rate_overrides.get('overig')
    if p_ben is None and p_ov is None:
        return rates
    _pos = rates[rates > 0]
    if _pos.empty:
        return rates
    # Twee meest voorkomende positieve niveaus = Benelux (hoog) en overig (laag).
    _niveaus = sorted(_pos.round(4).value_counts().index.tolist()[:2])
    if not _niveaus:
        return rates
    _laag = _niveaus[0]
    _hoog = _niveaus[-1]
    _r = rates.round(4)
    out = rates.copy()
    if p_ben is not None:
        out = out.mask(_r == round(_hoog, 4), float(p_ben))
    if p_ov is not None and _hoog != _laag:
        out = out.mask(_r == round(_laag, 4), float(p_ov))
    return out.clip(0.0, 1.0)


def laad_adoptie_data(excel_file=None, rate_overrides=None) -> pd.DataFrame:
    """
    Laad de 'Adoptie'-tab met per (component, klant)-combinatie het aantal
    orders en de adoption rate.

    excel_file: bestandspad (str) of file-like object; valt terug op EXCEL_PATH.

    Returns een DataFrame met kolommen:
        Code, Klant, Orders_component_klant, Orders_klant_totaal, Adoption_rate
    """
    bron = excel_file if excel_file is not None else EXCEL_PATH
    # Reset de leespositie van een geüploade buffer: deze functie kan meerdere
    # keren op hetzelfde file-like object worden aangeroepen (bv. runs +
    # metadata), waardoor een eerder gelezen buffer aan het einde zou staan.
    if excel_file is not None:
        try:
            bron.seek(0)
        except (AttributeError, ValueError):
            pass
    # Zoek de adoptie-tab onder de bekende naamvarianten (nieuwe subscriptie-
    # dataset: 'Adoption_rate_per_klant'; oude complete-Excel: 'Adoptie').
    _xls = pd.ExcelFile(bron)
    _sheet = next((s for s in ADOPTIE_SHEET_CANDIDATES if s in _xls.sheet_names), None)
    if _sheet is None:
        raise ValueError(
            f"Geen adoptie-tab gevonden (gezocht: {ADOPTIE_SHEET_CANDIDATES}; "
            f"aanwezig: {_xls.sheet_names})")
    df = _xls.parse(_sheet)
    df = df.rename(columns={
        'Verkooporderregel artikel.Artikel.Artikelcode': 'Code',
        'Order.Relatie':                                 'Klant',
        'Aantal_orders_5jr':                             'Orders_component_klant',
        'Totaal_bestellingen_klant_5jr':                 'Orders_klant_totaal',
        'Adoption_rate':                                 'Adoption_rate',
    })
    df['Code']  = df['Code'].astype(str).str.strip()
    df['Klant'] = df['Klant'].astype(str).str.strip()
    df['Adoption_rate'] = (
        pd.to_numeric(df['Adoption_rate'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    )
    # Optioneel: de twee adoption-rate niveaus (Benelux/overig) overschrijven.
    df['Adoption_rate'] = _hermap_adoption_rates(df['Adoption_rate'], rate_overrides)
    df['Orders_component_klant'] = (
        pd.to_numeric(df['Orders_component_klant'], errors='coerce')
        .fillna(0).clip(lower=0).astype(int)
    )
    df['Orders_klant_totaal'] = (
        pd.to_numeric(df['Orders_klant_totaal'], errors='coerce')
        .fillna(0).clip(lower=0).astype(int)
    )
    return df[['Code', 'Klant', 'Orders_component_klant',
               'Orders_klant_totaal', 'Adoption_rate']]


def classificatie_codes() -> set:
    """Geef de set artikelcodes uit de classificatie-selectie (bpa_selectie.json).

    Lege set als er geen selectiebestand is. Wordt gebruikt om de
    subscriptie-simulatie standaard te beperken tot de componenten die uit
    de classificatie komen.
    """
    sel = laad_classificatie_selectie()
    return {str(c).strip() for c in sel.get('items', {}).keys()}


def simuleer_subscripties_per_component(
    n_runs:      int   = 500,
    seed:        int   = 42,
    excel_file         = None,
    codes              = None,
    rate_overrides     = None,
) -> pd.DataFrame:
    """
    Monte Carlo simulatie van het aantal subscripties per component.

    Per run en per (component, klant) wordt de abonneerkans q_klant bepaald en
    een Bernoulli-trekking gedaan voor de binaire abonneer-beslissing:

        q_klant   = 1 - (1 - adoption_rate) ** aantal_orders
        X_klant   ~ Bernoulli(q_klant)   (1 = abonneert, 0 = niet)

    De kans dat een klant een subscriptie neemt stijgt zo met het aantal orders
    voor het component:
        P(X_klant = 1) = 1 - (1 - adoption_rate) ** aantal_orders.
    Het aantal subscripties voor een component in een run is de som van deze
    binaire beslissingen over alle klanten (maximaal het aantal klanten). Over
    alle runs wordt de verdeling samengevat.

    Parameters
    ----------
    n_runs : aantal Monte Carlo runs (stochastische trekkingen).
    seed   : seed voor reproduceerbaarheid (None = willekeurig).
    excel_file : bestandspad of file-like object; valt terug op EXCEL_PATH.
    codes : optionele iterable van artikelcodes om de simulatie te beperken.
            Bij None worden standaard alleen de classificatie-componenten
            (bpa_selectie.json) gesimuleerd.

    Returns
    -------
    DataFrame geïndexeerd op Code met kolommen:
        gem_subs, std_subs, p05, p50, p95, min_subs, max_subs,
        n_klanten, n_orders, runs
    """
    runs = simuleer_subscripties_runs(
        n_runs=n_runs, seed=seed, excel_file=excel_file, codes=codes,
        rate_overrides=rate_overrides,
    )
    if not runs:
        return pd.DataFrame()

    adoptie = laad_adoptie_data(excel_file)
    codes_set = {str(c).strip() for c in codes} if codes is not None else classificatie_codes()
    if codes_set:
        adoptie = adoptie[adoptie['Code'].isin(codes_set)]
    n_klanten = adoptie.groupby('Code').size()
    n_orders  = adoptie.groupby('Code')['Orders_component_klant'].sum()

    resultaten: dict = {}
    for code, subs_per_run in runs.items():
        resultaten[code] = {
            'gem_subs':  float(subs_per_run.mean()),
            'std_subs':  float(subs_per_run.std(ddof=0)),
            'p05':       float(np.percentile(subs_per_run, 5)),
            'p50':       float(np.percentile(subs_per_run, 50)),
            'p95':       float(np.percentile(subs_per_run, 95)),
            'min_subs':  int(subs_per_run.min()),
            'max_subs':  int(subs_per_run.max()),
            'n_klanten': int(n_klanten.get(code, 0)),
            'n_orders':  int(n_orders.get(code, 0)),
            'runs':      int(n_runs),
        }
    out = pd.DataFrame.from_dict(resultaten, orient='index')
    out.index.name = 'Code'
    return out.sort_values('gem_subs', ascending=False)


def simuleer_subscripties_runs(
    n_runs:      int   = 500,
    seed:        int   = 42,
    excel_file         = None,
    codes              = None,
    rate_overrides     = None,
) -> dict:
    """
    Voer de subscriptie-simulatie uit en geef de ruwe trekkingen terug.

    Per klant wordt eerst de abonneerkans q_klant = 1 - (1 - adoption_rate)^{orders}
    bepaald (= kans op ≥ 1 conversie over de orders). Daarna volgt een
    Bernoulli-trekking voor de binaire abonneer-beslissing:
        q_klant = 1 - (1 - adoption_rate) ** orders
        X_klant ~ Bernoulli(q_klant)   (1 = abonneert, 0 = niet)
    Het aantal subscripties per component is de som van deze binaire
    beslissingen over de klanten (Poisson-binomiaal, maximaal het aantal
    klanten).

    Bij codes=None wordt standaard beperkt tot de classificatie-componenten.

    Returns een dict {code → np.ndarray van lengte n_runs} met per run het
    gesimuleerde aantal subscripties voor dat component. Handig voor het
    plotten van de verdeling (histogram) per component.
    """
    adoptie = laad_adoptie_data(excel_file, rate_overrides=rate_overrides)
    codes_set = {str(c).strip() for c in codes} if codes is not None else classificatie_codes()
    if codes_set:
        adoptie = adoptie[adoptie['Code'].isin(codes_set)]
    if adoptie.empty:
        return {}

    rng = np.random.default_rng(seed)
    runs: dict = {}
    for code, grp in adoptie.groupby('Code', sort=True):
        n_arr = grp['Orders_component_klant'].to_numpy()
        p_arr = grp['Adoption_rate'].to_numpy()
        # Bernoulli-keuze per klant: X_klant ~ Bernoulli(q_klant) met
        # q_klant = 1 - (1 - adoption_rate)^{orders} = P(≥ 1 conversie over de
        # orders van de klant). Zo stijgt de abonneerkans met het aantal orders.
        # Z_component = Σ_klant X_klant (Poisson-binomiaal over de klanten).
        # (n_runs, n_klanten) Bernoulli-trekkingen → som per run.
        q_arr = 1.0 - (1.0 - p_arr) ** n_arr
        draws = rng.random((n_runs, q_arr.shape[0])) < q_arr[None, :]
        runs[code] = draws.sum(axis=1)
    return runs


# ══════════════════════════════════════════════════════════════════════════════
#  MENU-FUNCTIES
# ══════════════════════════════════════════════════════════════════════════════

def menu_toon_overzicht(cfg: dict) -> None:
    print("\nOverzicht berekenen…")
    df = bereken_overzicht(cfg)
    if df.empty:
        print("  Geen onderdelen gevonden.")
        return

    sl_cols = [c for c in df.columns if c.startswith('s@')]
    hdr = f"{'Code':<22}  {'Omschrijving':<40}  {'N':>4}  {'λ/jr':>8}  {'LT(d)':>6}  {'μ':>7}"
    for c in sl_cols:
        hdr += f"  {c:>8}"
    hdr += f"  {'bron':<10}"
    print(hdr)
    print('─' * len(hdr))

    totals = {c: 0 for c in sl_cols}
    for code, row in df.iterrows():
        line = (f"{code:<22}  {row['Descr']:<40}  {int(row['n_klanten']):>4}  "
                f"{row['lambda_jr']:>8.4f}  {int(row['LT_dagen']):>6}  {row['mu']:>7.4f}")
        for c in sl_cols:
            val = int(row[c])
            line += f"  {val:>8}"
            totals[c] += val
        line += f"  {row['bron']:<10}"
        print(line)

    print('─' * len(hdr))
    totline = f"{'TOTAAL':<22}  {'':40}  {'':>4}  {'':>8}  {'':>6}  {'':>7}"
    for c in sl_cols:
        totline += f"  {totals[c]:>8}"
    print(totline)

    # Optioneel opslaan als CSV
    antw = input("\nCSV opslaan? (j/n) [n]: ").strip().lower()
    if antw == 'j':
        pad = os.path.join(SCRIPT_DIR, f"bpa_overzicht_{date.today()}.csv")
        df.to_csv(pad, sep=';', decimal=',')
        print(f"  ✓ Opgeslagen als {pad}")


def menu_pas_subscripties_aan(cfg: dict) -> None:
    cfg.setdefault('ip_overrides', {})
    cfg.setdefault('lt_overrides', {})

    print("\nOverride per component – N, inkoopprijs en levertijd (laat code leeg om te stoppen):")
    print("  Laat een veld leeg om het ongewijzigd te laten. Typ 'x' om een override te verwijderen.")
    while True:
        code = input("  Artikelcode (of Enter om te stoppen): ").strip()
        if not code:
            break

        # N
        huidig_n = cfg['n_klanten_overrides'].get(code, '(uit data)')
        antw = input(f"    N subscripties (huidig: {huidig_n}, 'x'=verwijder): ").strip()
        if antw.lower() == 'x':
            cfg['n_klanten_overrides'].pop(code, None)
            print(f"    ✓ N-override verwijderd.")
        elif antw:
            try:
                cfg['n_klanten_overrides'][code] = int(antw)
                print(f"    ✓ N → {int(antw)}")
            except ValueError:
                print("    ✗ Ongeldige invoer, overgeslagen.")

        # Inkoopprijs
        huidig_ip = cfg['ip_overrides'].get(code, '(uit Excel)')
        antw = input(f"    Inkoopprijs € (huidig: {huidig_ip}, 'x'=verwijder): ").strip().replace(',', '.')
        if antw.lower() == 'x':
            cfg['ip_overrides'].pop(code, None)
            print(f"    ✓ IP-override verwijderd.")
        elif antw:
            try:
                cfg['ip_overrides'][code] = float(antw)
                print(f"    ✓ IP → €{float(antw):.2f}")
            except ValueError:
                print("    ✗ Ongeldige invoer, overgeslagen.")

        # Levertijd
        huidig_lt = cfg['lt_overrides'].get(code, '(uit Excel)')
        antw = input(f"    Levertijd dagen (huidig: {huidig_lt}, 'x'=verwijder): ").strip()
        if antw.lower() == 'x':
            cfg['lt_overrides'].pop(code, None)
            print(f"    ✓ LT-override verwijderd.")
        elif antw:
            try:
                cfg['lt_overrides'][code] = int(antw)
                print(f"    ✓ LT → {int(antw)} dagen")
            except ValueError:
                print("    ✗ Ongeldige invoer, overgeslagen.")

    sla_config_op(cfg)


def menu_voeg_component_toe(cfg: dict) -> None:
    print("\n── Nieuw component toevoegen ──")
    code = input("  Artikelcode: ").strip()
    if not code:
        print("  Geannuleerd.")
        return
    if code in cfg['handmatige_componenten']:
        print(f"  ⚠  {code} bestaat al. Gebruik 'Subscripties aanpassen' of verwijder het eerst.")
        return

    descr = input("  Omschrijving: ").strip()
    try:
        lam   = float(input("  Lambda (vraag per jaar): ").replace(',', '.'))
        lt    = int(input("  Levertijd leverancier→BPA (dagen): "))
        n     = int(input("  Aantal subscripties [1]: ").strip() or 1)
        ip_raw = input("  Inkoopprijs (optioneel, bijv. 1234.56): ").strip().replace(',', '.')
        ip    = float(ip_raw) if ip_raw else 0.0
    except ValueError:
        print("  ✗ Ongeldige invoer, component niet toegevoegd.")
        return

    cfg['handmatige_componenten'][code] = {
        'descr':           descr,
        'lambda_per_jaar': lam,
        'lt_dagen':        lt,
        'n_klanten':       n,
        'ip':              ip,
    }
    print(f"  ✓ Component '{code}' toegevoegd.")
    sla_config_op(cfg)


def menu_verwijder_component(cfg: dict) -> None:
    """Verwijder een component uit het model (handmatig of Excel)."""
    handmatig  = cfg['handmatige_componenten']
    uitgesloten = cfg.setdefault('uitgesloten_componenten', [])

    # Laad Excel-codes voor weergave
    try:
        excel_codes = list(laad_excel_onderdelen().index)
    except FileNotFoundError:
        excel_codes = []

    alle_codes = list(handmatig.keys()) + [c for c in excel_codes if c not in handmatig]
    actief     = [c for c in alle_codes if c not in uitgesloten]

    if not actief:
        print("\n  Geen actieve componenten om te verwijderen.")
        return

    print("\nActieve componenten:")
    for c in actief:
        bron = 'handmatig' if c in handmatig else 'excel'
        descr = handmatig[c].get('descr', '') if c in handmatig else ''
        print(f"  {c:<22}  [{bron}]  {descr}")

    code = input("Artikelcode om te verwijderen (Enter = annuleren): ").strip()
    if not code:
        return
    if code in handmatig:
        del handmatig[code]
        print(f"  ✓ '{code}' (handmatig) verwijderd.")
        sla_config_op(cfg)
    elif code in excel_codes:
        if code not in uitgesloten:
            uitgesloten.append(code)
        print(f"  ✓ '{code}' (Excel) uitgesloten van het model.")
        sla_config_op(cfg)
    else:
        print(f"  ✗ '{code}' niet gevonden.")


def menu_toon_config(cfg: dict) -> None:
    print(f"\nConfiguratiebestand: {CONFIG_PATH}")
    print(f"  Aangemaakt   : {cfg['aangemaakt']}")
    print(f"  Aangepast    : {cfg['aangepast']}")
    overrides = cfg['n_klanten_overrides']
    if overrides:
        print(f"\n  Overrides ({len(overrides)}):")
        for code, n in overrides.items():
            print(f"    {code:<22}  →  {n} subscripties")
    else:
        print("  Geen per-component overrides.")
    handmatig = cfg['handmatige_componenten']
    if handmatig:
        print(f"\n  Handmatige componenten ({len(handmatig)}):")
        for code, v in handmatig.items():
            print(f"    {code:<22}  λ={v['lambda_per_jaar']:.4f}/jr  "
                  f"LT={v['lt_dagen']}d  N={v.get('n_klanten','std')}  "
                  f"IP=€{v.get('ip',0):.2f}  – {v.get('descr','')}")
    else:
        print("  Geen handmatige componenten.")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT BIJ AFSLUITING
# ══════════════════════════════════════════════════════════════════════════════

def exporteer_afsluiting(cfg: dict) -> None:
    """Sla bij afsluiting automatisch een CSV op met alle componenten en basisvoorraden."""
    print("\nOverzicht opslaan…")
    df = bereken_overzicht(cfg)
    if df.empty:
        print("  Geen onderdelen om op te slaan.")
        return

    pad = os.path.join(SCRIPT_DIR, f"bpa_base_stock_{date.today()}.csv")
    df.to_csv(pad, sep=';', decimal=',')
    print(f"  ✓ Basisvoorraden opgeslagen in {pad}")
    print(f"     {len(df)} componenten  |  kolommen: {', '.join(df.columns.tolist())}")


# ══════════════════════════════════════════════════════════════════════════════
#  HOOFDMENU
# ══════════════════════════════════════════════════════════════════════════════

MENU = """
╔══════════════════════════════════════════════════╗
║         BPA Jaarlijks Beheer Tool                ║
╠══════════════════════════════════════════════════╣
║  1. Toon overzicht basisvoorraden                ║
║  2. Pas subscripties aan (per code)               ║
║  3. Voeg nieuw component toe                     ║
║  4. Verwijder component uit model                ║
║  5. Toon huidige configuratie                    ║
║  0. Afsluiten                                    ║
╚══════════════════════════════════════════════════╝
Keuze: """


def main() -> None:
    cfg = laad_config()
    print(f"\nBPA Beheer Tool – configuratie geladen (aangepast: {cfg['aangepast']})")

    while True:
        keuze = input(MENU).strip()
        if keuze == '1':
            menu_toon_overzicht(cfg)
        elif keuze == '2':
            menu_pas_subscripties_aan(cfg)
        elif keuze == '3':
            menu_voeg_component_toe(cfg)
        elif keuze == '4':
            menu_verwijder_component(cfg)
        elif keuze == '5':
            menu_toon_config(cfg)
        elif keuze == '0':
            exporteer_afsluiting(cfg)
            print("Tot de volgende jaarlijkse update!")
            break
        else:
            print("  Ongeldige keuze, probeer opnieuw.")


if __name__ == '__main__':
    main()
