# -*- coding: utf-8 -*-
"""
Classificatie-module
====================
Pure scoring-logica voor spare parts classificatie. Wordt zowel door de
CLI (`classificatie_scoring.py`) als door de Streamlit-tab gebruikt.

Belangrijkste functies:
    laad_ruwe_dataset(path)              → DataFrame
    bereken_scores(df, params)           → DataFrame met Score_*/Gewogen_Score/Beslissing
    pas_harde_filters_toe(df, params)    → DataFrame
    bouw_selectie_payload(df, params)    → dict (klaar voor json.dump)
    schrijf_selectie_json(payload, path) → None
    weight_sensitivity(df, params, step) → (per-artikel DataFrame[, combo DataFrame])
"""

from __future__ import annotations

import io
import json
import os
from collections import Counter
from dataclasses import dataclass, field, asdict, replace
from typing import Iterable

import numpy as np
import pandas as pd

# ── Kolomnamen in de bron-Excel ───────────────────────────────────────────────
COL_ABC          = "ABC_categorie"
COL_PRICE        = "Standaard verkoopprijs"
COL_LOCATIONS    = "Aantal_klantlocaties_met_orders_5jr"
COL_ORDERS       = "Gem_orders_per_klantlocatie_5jr"
COL_ARTICLE_TYPE = "ArticleType"

CODE_COL_CANDIDATES = [
    "Verkooporderregel artikel.Artikel.Artikelcode",
    "Artikelcode",
    "Code",
]
LT_COL_CANDIDATES = [
    "Hoofdleverancier.Levertijd",
    "Levertijd",
    "LT_dagen",
]
DESCR_COL_CANDIDATES = [
    "Omschrijving_standaard_artikelen",
    "Omschrijving",
    "Descr",
]
IP_COL_CANDIDATES = [
    "Inkoopprijs (standaard)",
    "Inkoopprijs",
]
MTBF_COL_CANDIDATES = [
    "MTBF(years)",
    "MTBF (years)",
    "MTBF_years",
    "MTBF(jaren)",
    "MTBF (jaren)",
    "MTBF (dagen)",
    "MTBF(dagen)",
    "MTBF (days)",
    "MTBF(days)",
    "MTBF_days",
    "MTBF",
]
TOTAAL_ORDERS_COL_CANDIDATES = [
    "Totaal_orders_5jr",
]
N_CUST_COL_CANDIDATES = [
    "Aantal_klantlocaties_5jr",
    "Aantal_klantlocaties_met_orders_5jr",
]

VERPLICHTE_KOLOMMEN = [COL_PRICE, COL_LOCATIONS, COL_ORDERS, COL_ARTICLE_TYPE]


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class ClassificatieParams:
    threshold:               float = 55.0
    weight_prijs:            float = 1 / 3
    weight_locaties:         float = 1 / 3
    weight_orders:           float = 1 / 3
    orders_power:            float = 2.0
    min_prijs:               float = 0.0   # minimum verkoopprijs (EUR), harde filter
    min_orders:              float = 0.0   # minimum gem. orders/locatie, harde filter
    selectie_modus:          str = "threshold"   # "threshold", "top_n" of "top_pct_all"
    top_n:                   int = 100            # alleen gebruikt bij modus "top_n"
    top_pct:                 float = 20.0         # alleen gebruikt bij modus "top_pct_all"
    min_klantlocaties:       int = 5
    article_type_filter:     tuple = ("critical", "onbekend")  # case-insensitief
    lt_default_waarden:      tuple = ("30", "30 dagen", "30,0", "30.0")
    score_methode:           str = "arithmetisch"  # "arithmetisch" of "geometrisch"
    epsilon:                 float = 1.0            # verschuiving bij geometrisch gemiddelde

    def normaliseer_weights(self) -> "ClassificatieParams":
        """Zorg dat de gewichten optellen tot 1."""
        s = self.weight_prijs + self.weight_locaties + self.weight_orders
        if s <= 0:
            return self
        return ClassificatieParams(
            **{**asdict(self),
               "weight_prijs":    self.weight_prijs    / s,
               "weight_locaties": self.weight_locaties / s,
               "weight_orders":   self.weight_orders   / s}
        )


# ── Laden ─────────────────────────────────────────────────────────────────────

def laad_ruwe_dataset(bron, sheet_name="Filtered ") -> pd.DataFrame:
    """Lees de volledige (ongefilterde) dataset.

    `bron` mag een pad of file-like object zijn. Standaard wordt de
    'Filtered '-sheet gelezen (gelijk aan bpa_beheer.SHEET_NAME); die bevat de
    correcte levertijden en MTBF(years). De eerste sheet 'Final_data' bevat
    placeholder-waarden (levertijd '0 dagen', MTBF in dagen) en mag hier niet
    gebruikt worden. Geef sheet_name=None om expliciet de eerste sheet te lezen.

    Snelheidsoptimalisatie: als `bron` een bestandspad is en er een CSV-sidecar
    ``<basisnaam>_filtered.csv`` naast het Excel-bestand staat, wordt die
    geladen in plaats van het Excel-bestand (~500x sneller). De sidecar wordt
    genegeerd bij file-like objecten (uploads) en bij een afwijkende sheet_name.
    """
    import os as _os
    # CSV-sidecar check: alleen voor padnamen + default sheet "Filtered "
    if isinstance(bron, (_os.PathLike, str)) and sheet_name == "Filtered ":
        _csv = _os.path.splitext(str(bron))[0] + "_filtered.csv"
        if _os.path.exists(_csv):
            df = pd.read_csv(_csv)
        else:
            df = pd.read_excel(bron, sheet_name=sheet_name)
    else:
        df = pd.read_excel(bron, sheet_name=sheet_name) if sheet_name else pd.read_excel(bron)

    code_col = next((c for c in CODE_COL_CANDIDATES if c in df.columns), None)
    if code_col is not None:
        df = df[df[code_col].notna()].copy()
        df[code_col] = df[code_col].astype(str).str.strip()
        df = df[~df[code_col].str.lower().isin(("", "nan", "none", "null"))]
        df = df.drop_duplicates(subset=code_col, keep="first")
    return df


def _lees_csv_flexibel(bron) -> pd.DataFrame:
    """ERP-exports zijn niet consistent UTF-8; val terug op cp1252 met
    foutentolerantie voor onherkenbare bytes en misvormde regels."""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(bron, sep=";", dtype=str, encoding=enc)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
            if hasattr(bron, "seek"):
                bron.seek(0)
    if hasattr(bron, "seek"):
        bron.seek(0)
    return pd.read_csv(
        bron, sep=";", dtype=str, encoding="cp1252",
        encoding_errors="replace", on_bad_lines="skip",
    )


def _parse_eur(v) -> float:
    if pd.isna(v):
        return 0.0
    s = (
        str(v)
        .replace("€", "").replace("\u0080", "").replace("\u00A0", "")
        .replace(" ", "").replace(".", "").replace(",", ".").strip()
    )
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_dagen_erp(v) -> int:
    if pd.isna(v):
        return 0
    s = str(v).strip().split()
    return int(s[0]) if s and s[0].isdigit() else 0


def _parse_mtbf_dagen(v) -> float:
    if pd.isna(v):
        return 0.0
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return 0.0


# Zelfde landenset als complete_dataverwerking_europa_only(_zonder_rspl).py,
# zodat 'ERP-documenten' en de losse pipeline-scripts consistent filteren.
EUROPESE_LANDEN = {
    "NL-NEDERLAND", "DE-GERMANY", "FR-FRANCE", "BE-BELGIUM", "AT-AUSTRIA",
    "CH-SWITZERLAND", "PL-POLAND", "IT-ITALY", "ES-SPAIN", "PT-PORTUGAL",
    "SE-SWEDEN", "NO-NORWAY", "DK-DENMARK", "FI-FINLAND", "CZ-CZECH REPUBLIC",
    "SK-SLOVAKIA", "HU-HUNGARY", "RO-ROMANIA", "BG-BULGARIA", "HR-CROATIA",
    "SI-SLOVENIA", "GR-GREECE", "IE-IRELAND", "GB-UNITED KINGDOM", "LU-LUXEMBOURG",
    "LV-LATVIA", "LT-LITHUANIA", "EE-ESTONIA", "MT-MALTA", "CY-CYPRUS",
    "IS-ICELAND", "TR-TURKEY",
}
ORDERS_VANAF = pd.Timestamp("2021-01-01")


def laad_erp_documenten(document_bron, document1_bron) -> pd.DataFrame:
    """Bouw de ruwe dataset op uit twee ERP-exports i.p.v. de centrale Excel.

    `document_bron`  = artikelstamdata-export (prijs, levertijd, ArticleType).
    `document1_bron` = orderregels-export (voor klantlocaties/orders).
    Ververst prijs/levertijd/orders altijd uit de nieuwe export; bestaande
    selecties in het overzicht blijven behouden (zie `_bestaande_codes` in de
    UI) en worden hier niet aangeraakt.
    """
    code_col = CODE_COL_CANDIDATES[0]

    art = _lees_csv_flexibel(document_bron)
    art.columns = [c.strip() for c in art.columns]
    artikelen = pd.DataFrame({
        code_col: art["Artikelcode"].astype(str).str.strip(),
        "Omschrijving_standaard_artikelen": art.get("Omschrijving"),
        COL_PRICE: art["Standaard verkoopprijs"].apply(_parse_eur),
        "Inkoopprijs (standaard)": art.get(
            "Inkoopprijs (standaard)", pd.Series(dtype=str)
        ).apply(_parse_eur),
        "Hoofdleverancier.Levertijd": art.get(
            "Hoofdleverancier.Levertijd", pd.Series(dtype=str)
        ).apply(_parse_dagen_erp),
        "Incourant": art.get("Incourant"),
        COL_ARTICLE_TYPE: art.get("Aanbeveling", pd.Series(dtype=str)).astype(str).str.strip(),
        "MTBF(years)": art.get(
            "MTBF (dagen)", pd.Series(dtype=str)
        ).apply(_parse_mtbf_dagen) / 365,
    }).drop_duplicates(subset=[code_col], keep="first")

    ord_df = _lees_csv_flexibel(document1_bron)
    ord_df.columns = [c.strip() for c in ord_df.columns]
    art_col     = code_col
    order_col   = "Order"
    relatie_col = "Order.Bestemmingsadres.Relatie"
    land_col    = "Order.Bestemmingsadres.Adres.Land"
    datum_col   = "Order.Orderdatum"
    ord_df = ord_df[
        ord_df[art_col].notna() & (ord_df[art_col].astype(str).str.strip() != "")
    ].copy()
    ord_df[art_col] = ord_df[art_col].astype(str).str.strip()

    # Filter: alleen klanten binnen Europa
    if land_col in ord_df.columns:
        ord_df = ord_df[
            ord_df[land_col].astype(str).str.strip().isin(EUROPESE_LANDEN)
        ].copy()

    # Filter: alleen orders vanaf 1 januari 2021
    if datum_col in ord_df.columns:
        _orderdatum = pd.to_datetime(
            ord_df[datum_col], dayfirst=True, errors="coerce"
        )
        ord_df = ord_df[_orderdatum >= ORDERS_VANAF].copy()

    ord_df["Klantlocatie"] = (
        ord_df.get(relatie_col, "").astype(str).str.strip() + " | "
        + ord_df.get(land_col, "").astype(str).str.strip()
    )
    orders = ord_df.groupby(art_col).agg(
        Totaal_orders_5jr=(order_col, "nunique"),
        Aantal_klantlocaties_met_orders_5jr=("Klantlocatie", "nunique"),
    ).reset_index()
    orders[COL_ORDERS] = (
        orders["Totaal_orders_5jr"]
        / orders["Aantal_klantlocaties_met_orders_5jr"].clip(lower=1)
    ).round(3)

    merged = artikelen.merge(orders, on=code_col, how="left")
    merged["Totaal_orders_5jr"] = merged["Totaal_orders_5jr"].fillna(0)
    merged[COL_LOCATIONS] = merged[COL_LOCATIONS].fillna(0)
    merged[COL_ORDERS] = merged[COL_ORDERS].fillna(0.0)
    merged["Aantal_klantlocaties_5jr"] = merged[COL_LOCATIONS]
    return merged


def df_naar_filtered_excel_bytes(df: pd.DataFrame) -> bytes:
    """Schrijf een reeds opgebouwde dataset weg als in-memory Excel-workbook
    (sheet 'Filtered '), zodat de rest van de tool ongewijzigd blijft werken."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Filtered ", index=False)
    return buf.getvalue()


def bouw_filtered_excel_bytes(document_bron, document1_bron) -> bytes:
    """Bouw een in-memory Excel-workbook (sheet 'Filtered ') uit de twee
    ERP-documenten, zodat de rest van de tool ongewijzigd blijft werken."""
    df = laad_erp_documenten(document_bron, document1_bron)
    return df_naar_filtered_excel_bytes(df)


def controleer_kolommen(df: pd.DataFrame) -> list[str]:
    """Geef lijst terug van ontbrekende verplichte kolommen."""
    return [c for c in VERPLICHTE_KOLOMMEN if c not in df.columns]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_lt_dagen(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return int(v) if v > 0 else None
    s = str(v).strip()
    if not s:
        return None
    head = s.split()[0].replace(",", ".")
    try:
        n = int(float(head))
        return n if n > 0 else None
    except ValueError:
        return None


def _lt_bron(v, defaults: Iterable[str]) -> str:
    if pd.isna(v) or str(v).strip() == "":
        return "ontbreekt"
    # Een waarde die naar 0/None parseert (bv. "0 dagen") is geen bevestigde
    # levertijd maar een leeg/placeholder-veld → behandel als ontbrekend,
    # consistent met _parse_lt_dagen (die hier None teruggeeft).
    if _parse_lt_dagen(v) is None:
        return "ontbreekt"
    if str(v).strip() in set(defaults):
        return "default"
    return "geupdate"


def _mtbf_naar_jaren(raw, col_name: str | None) -> float | None:
    """Parseer MTBF-waarde als jaren, robuust voor diverse input-types.

    MTBF is altijd in jaren. Een kolomnaam of waarde die 'dagen'/'days'
    bevat wijst op een fout in de brondata — de numerieke waarde wordt
    ongewijzigd als jaren teruggegeven.

    Accepteert int, float, en strings (incl. NL-format ``"10,5"`` of
    ``"€ 1.234,56"``-achtige notatie). Geeft ``None`` bij leeg/onparseerbaar.
    """
    if col_name is None:
        return None
    # 1) None / NaN check (pd.isna faalt op str-arrays bij list-achtige input,
    #    daarom in try/except)
    try:
        if raw is None or pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass

    # 2) Numeriek (int/float/bool/Decimal) — direct casten
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        val = float(raw)
        unit_van_value = None
    else:
        # 3) String-pad: strip whitespace, currency-prefix, parse NL-format,
        #    detecteer unit in de string zelf.
        s = str(raw).strip().lstrip('\x80€ ').strip()
        if not s:
            return None
        s_low = s.lower()
        unit_van_value = None
        if "dag" in s_low or "day" in s_low:
            unit_van_value = "dagen"
        elif "jaar" in s_low or "jaren" in s_low or "year" in s_low:
            unit_van_value = "jaren"
        # Houd alleen cijfers + decimaaltekens over
        cleaned = "".join(ch for ch in s if ch.isdigit() or ch in ",.-")
        if not cleaned or cleaned in ("-", ".", ","):
            return None
        if "," in cleaned:
            # NL: punt = duizendteken, komma = decimaal
            cleaned = cleaned.replace(".", "").replace(",", ".")
        try:
            val = float(cleaned)
        except ValueError:
            return None

    if val <= 0:
        return None

    # MTBF is altijd in jaren; dag-indicaties in waarde of kolomnaam zijn
    # brondata-fouten — de waarde wordt ongewijzigd als jaren teruggegeven.
    return val


# ── Scoring ───────────────────────────────────────────────────────────────────


def _gewogen_score(
    score_prijs: pd.Series,
    score_locaties: pd.Series,
    score_orders: pd.Series,
    p: ClassificatieParams,
) -> pd.Series:
    """Aggregeer de drie criterium-scores tot een gewogen eindscore.

    arithmetisch : gewogen som (lineair compensatoir).
    geometrisch  : gewogen geometrisch gemiddelde (gedeeltelijk compensatoir);
                   scores worden verschoven met epsilon en geschaald naar (0, 1]
                   vóór de machtsverheffing, conform G_i in het methoderapport.
    """
    wp, wc, ws = p.weight_prijs, p.weight_locaties, p.weight_orders
    if p.score_methode == "geometrisch":
        eps   = p.epsilon
        denom = 100.0 + eps
        sp = (score_prijs    + eps) / denom
        sc = (score_locaties + eps) / denom
        ss = (score_orders   + eps) / denom
        return (100.0 * (sp ** wp) * (sc ** wc) * (ss ** ws)).round(1)
    return (score_prijs * wp + score_locaties * wc + score_orders * ws).round(1)


def bereken_scores(df: pd.DataFrame, params: ClassificatieParams) -> pd.DataFrame:
    """
    Voegt de kolommen Score_Prijs, Score_Locaties, Score_Orders,
    Gewogen_Score en Classificatie_Beslissing toe.

    De min-max-normalisatie gebeurt over de rijen die in `df` worden
    aangeboden. Roep daarom eerst `pas_basis_filters_toe` aan, zodat de
    scores genormaliseerd worden over de artikelenset NÁ de harde basis-
    filters (ArticleType + minimum klantlocaties). De top-n-markering volgt
    daarna via `pas_topn_selectie_toe`.
    """
    p = params.normaliseer_weights()
    df = df.copy()

    # Score_Prijs: lineaire min-max. Lagere prijs = lagere score.
    _price_raw = df[COL_PRICE].clip(lower=0).fillna(0)
    _p_min, _p_max = _price_raw.min(), _price_raw.max()
    if _p_max > _p_min:
        df["Score_Prijs"] = ((_price_raw - _p_min) / (_p_max - _p_min)) * 100
    else:
        df["Score_Prijs"] = 100.0

    # Score_Locaties: lineaire min-max, meer = beter
    _loc_raw = df[COL_LOCATIONS].clip(lower=0).fillna(0)
    _l_min, _l_max = _loc_raw.min(), _loc_raw.max()
    if _l_max > _l_min:
        df["Score_Locaties"] = ((_loc_raw - _l_min) / (_l_max - _l_min)) * 100
    else:
        df["Score_Locaties"] = 100.0

    # Score_Orders: log + inverse min-max met vaste floor=1.0, niet-lineair.
    # Log-schaal omdat orderfrequentie een rate is: het verschil tussen
    # 0.1 en 1 order/locatie is even relevant als tussen 1 en 10.
    # Alle items met orders <= floor krijgen score 100 (echte slow movers).
    _orders_floor = 1.0
    _orders_clipped = df[COL_ORDERS].clip(lower=_orders_floor)
    _orders_log = np.log1p(_orders_clipped)
    _of_log = np.log1p(_orders_floor)
    _om_log = _orders_log.max()
    if _om_log > _of_log:
        _scaled_orders = ((_om_log - _orders_log) / (_om_log - _of_log)).fillna(0.0)
    else:
        _scaled_orders = pd.Series(1.0, index=df.index)
    df["Score_Orders"] = (_scaled_orders ** p.orders_power) * 100

    df["Gewogen_Score"] = _gewogen_score(
        df["Score_Prijs"], df["Score_Locaties"], df["Score_Orders"], p
    )

    for col in ["Score_Prijs", "Score_Locaties", "Score_Orders"]:
        df[col] = df[col].round(1)

    if p.selectie_modus in ("top_n", "top_pct_all"):
        # Bij top-X en top-pct-all wordt de beslissing pas ná de harde filters
        # bepaald (in pas_topn_selectie_toe), zodat de selectie over de
        # daadwerkelijk in aanmerking komende set gaat. Hier voorlopig alles
        # op "Niet opnemen".
        df["Classificatie_Beslissing"] = "Niet opnemen"
    else:
        df["Classificatie_Beslissing"] = np.where(
            df["Gewogen_Score"] >= p.threshold,
            "Opnemen in lijst",
            "Niet opnemen",
        )

    # λ_i = 1 / MTBF(jaren) — failure rate per individueel component per jaar.
    # MTBF wordt robuust geconverteerd via _mtbf_naar_jaren (tekst/int/float
    # én eenheid-detectie 'dagen'/'days' worden afgevangen).
    _mtbf_col = _find_col(df, MTBF_COL_CANDIDATES)
    if _mtbf_col is not None:
        _mtbf_jr = df[_mtbf_col].apply(lambda v: _mtbf_naar_jaren(v, _mtbf_col))
        df["MTBF_jaren"] = _mtbf_jr.round(4)
        df["Lambda_jr"]  = (1.0 / _mtbf_jr).round(4)
    else:
        df["MTBF_jaren"] = np.nan
        df["Lambda_jr"]  = np.nan

    return df


def pas_basis_filters_toe(df: pd.DataFrame, params: ClassificatieParams) -> pd.DataFrame:
    """Score-ONAFHANKELIJKE harde filters: ArticleType (case-insensitief) +
    minimum klantlocaties.

    Pas dit toe vóór `bereken_scores`, zodat de min-max-normalisatie over de
    daadwerkelijk in aanmerking komende artikelenset gaat in plaats van over
    de volledige ruwe dataset (incl. items die straks toch weggefilterd
    worden).
    """
    if COL_ARTICLE_TYPE in df.columns:
        df = df[df[COL_ARTICLE_TYPE].astype(str).str.strip().str.lower()
                .isin(set(s.lower() for s in params.article_type_filter))]
    if COL_LOCATIONS in df.columns:
        df = df[df[COL_LOCATIONS].fillna(0) >= params.min_klantlocaties]
    if COL_PRICE in df.columns and params.min_prijs > 0:
        df = df[df[COL_PRICE].fillna(0) >= params.min_prijs]
    if COL_ORDERS in df.columns and params.min_orders > 0:
        df = df[df[COL_ORDERS].fillna(0) >= params.min_orders]
    return df


def pas_topn_selectie_toe(df: pd.DataFrame, params: ClassificatieParams) -> pd.DataFrame:
    """Score-AFHANKELIJKE stap: markeer in top_n- of top_pct_all-modus de
    geselecteerde componenten als "Opnemen in lijst". Vereist een reeds
    gescoorde DataFrame (kolommen Score_Prijs/Locaties/Orders + Gewogen_Score).
    """
    if params.selectie_modus == "top_n" and "Gewogen_Score" in df.columns:
        # Bij gelijke score beslist de oorspronkelijke volgorde (stable sort).
        df = df.copy()
        df["Classificatie_Beslissing"] = "Niet opnemen"
        _top_idx = (
            df["Gewogen_Score"]
            .sort_values(ascending=False, kind="stable")
            .head(max(int(params.top_n), 0))
            .index
        )
        df.loc[_top_idx, "Classificatie_Beslissing"] = "Opnemen in lijst"

    elif params.selectie_modus == "top_pct_all" and all(
        c in df.columns for c in ("Score_Prijs", "Score_Locaties", "Score_Orders")
    ):
        # Een component wordt opgenomen als het voor ALLE drie criteria
        # tegelijk in de bovenste top_pct% van de dataset valt.
        df = df.copy()
        _q = 1.0 - params.top_pct / 100.0
        _cp = df["Score_Prijs"].quantile(_q)
        _cl = df["Score_Locaties"].quantile(_q)
        _co = df["Score_Orders"].quantile(_q)
        _mask = (
            (df["Score_Prijs"]    >= _cp)
            & (df["Score_Locaties"] >= _cl)
            & (df["Score_Orders"]   >= _co)
        )
        df["Classificatie_Beslissing"] = np.where(_mask, "Opnemen in lijst", "Niet opnemen")

    return df


def pas_harde_filters_toe(df: pd.DataFrame, params: ClassificatieParams) -> pd.DataFrame:
    """Backwards-compat wrapper: basis-filters + top-n-selectie in één stap.

    Let op: deze functie normaliseert NIET opnieuw. Voor de correcte volgorde
    (basis-filteren → scoren → top-n) gebruik je
    `pas_basis_filters_toe` → `bereken_scores` → `pas_topn_selectie_toe`.
    """
    df = pas_basis_filters_toe(df, params)
    return pas_topn_selectie_toe(df, params)


# ── Selectie-payload ──────────────────────────────────────────────────────────

def bouw_selectie_payload(
    df_scored_filtered: pd.DataFrame,
    params: ClassificatieParams,
    bron_excel: str | None = None,
) -> dict:
    """Bouw het JSON-payload dat naar bpa_selectie.json wordt geschreven."""
    code_col  = _find_col(df_scored_filtered, CODE_COL_CANDIDATES)
    lt_col    = _find_col(df_scored_filtered, LT_COL_CANDIDATES)
    descr_col = _find_col(df_scored_filtered, DESCR_COL_CANDIDATES)
    ip_col    = _find_col(df_scored_filtered, IP_COL_CANDIDATES)
    vp_col    = COL_PRICE if COL_PRICE in df_scored_filtered.columns else None
    mtbf_col  = _find_col(df_scored_filtered, MTBF_COL_CANDIDATES)
    orders_col = _find_col(df_scored_filtered, TOTAAL_ORDERS_COL_CANDIDATES)
    ncust_col = _find_col(df_scored_filtered, N_CUST_COL_CANDIDATES)
    if code_col is None:
        raise ValueError(
            f"Geen artikelcode-kolom gevonden ({CODE_COL_CANDIDATES})."
        )

    sel = df_scored_filtered[
        df_scored_filtered["Classificatie_Beslissing"] == "Opnemen in lijst"
    ].copy()
    sel = sel[sel[code_col].notna()].copy()
    sel[code_col] = sel[code_col].astype(str).str.strip()
    sel = sel[~sel[code_col].str.lower().isin(("", "nan", "none", "null"))]
    sel = (
        sel.sort_values("Gewogen_Score", ascending=False, kind="stable")
        .drop_duplicates(subset=code_col, keep="first")
    )

    def _num(v):
        """Parse numeric value, ook NL-format ('€ 1.234,56') of '1.234,56'."""
        try:
            if v is None or pd.isna(v):
                return None
        except (TypeError, ValueError):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lstrip('\x80€ ').strip()
        if not s:
            return None
        if ',' in s:
            # Dutch: punt = duizendteken, komma = decimaalteken
            s = s.replace('.', '').replace(',', '.')
        try:
            return float(s)
        except ValueError:
            return None

    items = []
    n_g = n_d = n_o = 0
    for _, row in sel.iterrows():
        lt_raw = row.get(lt_col) if lt_col else None
        bron   = _lt_bron(lt_raw, params.lt_default_waarden) if lt_col else "onbekend"
        if   bron == "geupdate":  n_g += 1
        elif bron == "default":   n_d += 1
        elif bron == "ontbreekt": n_o += 1
        items.append({
            "code":     str(row[code_col]),
            "score":    float(row["Gewogen_Score"]),
            "lt_dagen": _parse_lt_dagen(lt_raw) if lt_col else None,
            "lt_bron":  bron,
            "abc":      str(row.get(COL_ABC, "")),
            # Metadata waarmee bpa_beheer een rij kan bouwen voor codes
            # die NIET in de BPA-Excel (`Filtered `-sheet) staan.
            "descr":             (str(row[descr_col])[:80] if descr_col and pd.notna(row.get(descr_col)) else ""),
            "ip":                _num(row.get(ip_col)) if ip_col else None,
            "vp":                _num(row.get(vp_col)) if vp_col else None,
            # MTBF altijd in JAREN opgeslagen. Bij bron-kolom in dagen
            # ("MTBF (dagen)") wordt automatisch door 365 gedeeld.
            "mtbf":              _mtbf_naar_jaren(row.get(mtbf_col), mtbf_col),
            "totaal_orders_5jr": _num(row.get(orders_col)) if orders_col else None,
            "n_cust":            _num(row.get(ncust_col)) if ncust_col else None,
            # λ_jr = N_locaties / MTBF(jaren); float of None
            "lambda_jr":         (float(row["Lambda_jr"])
                                   if "Lambda_jr" in row.index
                                      and pd.notna(row["Lambda_jr"])
                                   else None),
        })

    return {
        "gegenereerd":  str(pd.Timestamp.today()),
        "bron_excel":   bron_excel,
        "threshold":    params.threshold,
        "parameters":   asdict(params),
        "n_items":      len(items),
        "lt_overzicht": {"geupdate": n_g, "default": n_d, "ontbreekt": n_o},
        "items":        items,
    }


def schrijf_selectie_json(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── Convenience: end-to-end run ───────────────────────────────────────────────

def voer_classificatie_uit(
    bron, params: ClassificatieParams, sheet_name="Filtered "
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: laad → score → filter → bouw payload. Schrijft NIETS."""
    df_raw = laad_ruwe_dataset(bron, sheet_name=sheet_name)
    miss = controleer_kolommen(df_raw)
    if miss:
        raise ValueError(f"Ontbrekende kolommen: {miss}")
    # Eerst basis-filteren, daarna scoren: de normalisatie gaat zo over de
    # artikelenset NÁ de harde filters. Top-n volgt op de gescoorde set.
    df_basis    = pas_basis_filters_toe(df_raw, params)
    df_scored   = bereken_scores(df_basis, params)
    df_filtered = pas_topn_selectie_toe(df_scored, params)
    payload     = bouw_selectie_payload(df_filtered, params,
                                        bron_excel=str(bron) if isinstance(bron, str) else None)
    return df_filtered, payload


# ── Gewicht-sensitivity ─────────────────────────────────────────────────

def genereer_gewicht_grid(step: float = 0.1, min_weight: float = 0.0) -> list[tuple]:
    """Genereer alle (w_prijs, w_locaties, w_orders)-combinaties op een simplex.

    De drie gewichten lopen in stappen van `step` van 0 t/m 1 en sommeren per
    combinatie exact naar 1. Met `min_weight > 0` worden degeneratie-combinaties
    (waar een gewicht onder de drempel ligt, bv. exact 0) overgeslagen.

    Voorbeeld: step=0.5 → combinaties als (1,0,0), (0.5,0.5,0), (0,0,1), …
    """
    if not 0 < step <= 1:
        raise ValueError("step moet in (0, 1] liggen.")
    n = round(1.0 / step)
    grid: list[tuple] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            wp, wl, wo = i / n, j / n, k / n
            if min(wp, wl, wo) < min_weight:
                continue
            grid.append((round(wp, 6), round(wl, 6), round(wo, 6)))
    return grid


def _selectie_codes_voor_gewichten(
    df_scored_full: pd.DataFrame, params: ClassificatieParams, code_col: str
) -> set[str]:
    """Bepaal de set artikelcodes die met de gegeven gewichten 'Opnemen in lijst'
    krijgt. Hergebruikt de bestaande selectie-logica (threshold of top_n) en de
    harde filters; alleen de gewogen score wordt opnieuw berekend.
    """
    p = params.normaliseer_weights()
    gewogen = _gewogen_score(
        df_scored_full["Score_Prijs"],
        df_scored_full["Score_Locaties"],
        df_scored_full["Score_Orders"],
        p,
    )
    work = df_scored_full.copy()
    work["Gewogen_Score"] = gewogen
    if p.selectie_modus in ("top_n", "top_pct_all"):
        # Beslissing wordt in pas_topn_selectie_toe (na de filters) bepaald.
        work["Classificatie_Beslissing"] = "Niet opnemen"
    else:
        work["Classificatie_Beslissing"] = np.where(
            gewogen >= p.threshold, "Opnemen in lijst", "Niet opnemen"
        )
    work = pas_harde_filters_toe(work, p)
    sel = work[work["Classificatie_Beslissing"] == "Opnemen in lijst"]
    return set(sel[code_col].astype(str))


def _rbo(lijst_a: list, lijst_b: list, p_decay: float = 0.9) -> float:
    """Rank-Biased Overlap (Webber e.a., 2010).

    Meet hoe sterk twee ranglijsten overeenkomen, met meer gewicht voor de
    *top* van de lijst. p_decay < 1: hoe lager, hoe sterker de top meetelt
    (p=0.9 → ±86% van het gewicht in de top-10). 1.0 = identieke volgorde.
    """
    if not lijst_a or not lijst_b:
        return 0.0
    k = min(len(lijst_a), len(lijst_b))
    sa: set = set()
    sb: set = set()
    som = 0.0
    for d in range(k):
        sa.add(lijst_a[d])
        sb.add(lijst_b[d])
        som += (len(sa & sb) / (d + 1)) * (p_decay ** d)
    return (1 - p_decay) * som


def weight_sensitivity(
    df: pd.DataFrame,
    params: ClassificatieParams,
    step: float = 0.1,
    min_weight: float = 0.0,
    return_combos: bool = False,
):
    """Meet hoe gevoelig de classificatie-set is voor de criteria-gewichten.

    Alleen de drie criteria-gewichten (`weight_prijs`, `weight_locaties`,
    `weight_orders`) worden gevarieerd over een simplex-grid; alle overige
    parameters (drempel/top_n, penalty's, orders_power, harde filters) blijven
    gelijk aan `params`. Omdat de component-scores gewicht-onafhankelijk zijn,
    worden ze één keer berekend en is de sweep daarna goedkoop.

    Parameters
    ----------
    df : DataFrame
        Ruwe dataset óf een reeds gescoorde DataFrame (met Score_Prijs/
        Score_Locaties/Score_Orders). In het eerste geval wordt eenmalig
        `bereken_scores` aangeroepen.
    params : ClassificatieParams
        Basisinstellingen; de gewichten hierin bepalen de baseline-set.
    step : float
        Rasterresolutie van de gewichten (0.1 = stappen van 0,1 → 66 combinaties).
    min_weight : float
        Sla combinaties over waarin een gewicht onder deze drempel ligt
        (0.0 = ook extremen zoals 100% één criterium meenemen).
    return_combos : bool
        Geef ook een DataFrame per gewicht-combinatie terug.

    Returns
    -------
    per_artikel : DataFrame
        Per kandidaat-artikel (na harde filters) gesorteerd op selectie-
        frequentie:
          - Artikelcode
          - Selectie_count       : aantal combinaties waarin geselecteerd
          - Selectie_frequentie  : fractie 0..1 van alle combinaties
          - In_baseline          : zat het in de selectie bij `params`
          - Stabiliteit          : 'altijd' / 'soms' / 'nooit'
    combos : DataFrame   (alleen als return_combos=True)
        Per gewicht-combinatie: gewichten, #opnemen, overlap/Jaccard t.o.v.
        de baseline-set, én rangorde-robuustheid t.o.v. de baseline-rangorde:
          - spearman   : rangcorrelatie over de hele kandidaat-set (−1..1)
          - kendall    : Kendall's tau over de hele kandidaat-set (−1..1)
          - rbo        : rank-biased overlap (top-zwaar, 0..1)
          - max_rank_shift / mean_rank_shift : grootste/gemiddelde
                         positieverschuiving in de rangorde
    """
    if "Score_Prijs" in df.columns and "Score_Orders" in df.columns \
            and "Score_Locaties" in df.columns:
        df_scored = df.copy()
    else:
        # Eerst basis-filteren, dan scoren: normalisatie over de set ná de
        # harde filters (consistent met de hoofd-flow).
        df_scored = bereken_scores(pas_basis_filters_toe(df, params), params)

    code_col = _find_col(df_scored, CODE_COL_CANDIDATES)
    if code_col is None:
        raise ValueError(
            f"Geen artikelcode-kolom gevonden ({CODE_COL_CANDIDATES})."
        )

    grid = genereer_gewicht_grid(step, min_weight)
    if not grid:
        raise ValueError("Lege gewicht-grid; verlaag min_weight of step.")
    n_combos = len(grid)

    # Kandidaat-universum = artikelen die door de harde filters komen
    # (gewicht-onafhankelijk). Threshold-modus voorkomt top_n-markering hier.
    kandidaten = pas_harde_filters_toe(
        df_scored, replace(params, selectie_modus="threshold")
    )
    kandidaat_codes = set(kandidaten[code_col].astype(str))

    # Baseline-set + baseline-rangorde (voor rang-robuustheid). De rangorde
    # gaat over de volledige kandidaat-set, gesorteerd op gewogen score.
    pb = params.normaliseer_weights()
    _kand_codes_ser = kandidaten[code_col].astype(str)

    def _gewogen_op_kandidaten(wp: float, wl: float, wo: float) -> pd.Series:
        from dataclasses import replace as _dc_replace
        _p_tmp = _dc_replace(pb, weight_prijs=wp, weight_locaties=wl, weight_orders=wo)
        return _gewogen_score(
            kandidaten["Score_Prijs"],
            kandidaten["Score_Locaties"],
            kandidaten["Score_Orders"],
            _p_tmp,
        )

    baseline_score = _gewogen_op_kandidaten(
        pb.weight_prijs, pb.weight_locaties, pb.weight_orders
    )
    baseline_rank = baseline_score.rank(ascending=False, method="min")
    baseline_ranked = (
        _kand_codes_ser[baseline_score.sort_values(ascending=False).index].tolist()
    )

    baseline_codes = _selectie_codes_voor_gewichten(df_scored, params, code_col)

    teller: Counter = Counter()
    combo_rows = []
    for wp, wl, wo in grid:
        p = replace(
            params, weight_prijs=wp, weight_locaties=wl, weight_orders=wo
        )
        codes = _selectie_codes_voor_gewichten(df_scored, p, code_col)
        teller.update(codes)
        if return_combos:
            overlap = len(codes & baseline_codes)
            union   = len(codes | baseline_codes) or 1
            # Rangorde-robuustheid over de volledige kandidaat-set
            scen_score = _gewogen_op_kandidaten(wp, wl, wo)
            spear = baseline_score.corr(scen_score, method="spearman")
            kend  = baseline_score.corr(scen_score, method="kendall")
            scen_rank = scen_score.rank(ascending=False, method="min")
            _shift = (baseline_rank - scen_rank).abs()
            scen_ranked = (
                _kand_codes_ser[scen_score.sort_values(ascending=False).index]
                .tolist()
            )
            combo_rows.append({
                "weight_prijs":    wp,
                "weight_locaties": wl,
                "weight_orders":   wo,
                "n_opnemen":       len(codes),
                "overlap_baseline": overlap,
                "alleen_scenario": len(codes - baseline_codes),
                "alleen_baseline": len(baseline_codes - codes),
                "jaccard":         round(overlap / union, 3),
                "spearman":        round(float(spear), 3) if pd.notna(spear) else None,
                "kendall":         round(float(kend), 3) if pd.notna(kend) else None,
                "rbo":             round(_rbo(baseline_ranked, scen_ranked), 3),
                "max_rank_shift":  int(_shift.max()) if len(_shift) else 0,
                "mean_rank_shift": round(float(_shift.mean()), 1) if len(_shift) else 0.0,
            })

    alle_codes = kandidaat_codes | set(teller)
    rows = []
    for code in alle_codes:
        cnt = teller.get(code, 0)
        freq = cnt / n_combos
        rows.append({
            "Artikelcode":        code,
            "Selectie_count":     cnt,
            "Selectie_frequentie": round(freq, 4),
            "In_baseline":        code in baseline_codes,
            "Stabiliteit":        ("altijd" if cnt == n_combos
                                    else "nooit" if cnt == 0
                                    else "soms"),
        })
    per_artikel = (
        pd.DataFrame(rows)
        .sort_values(["Selectie_frequentie", "Artikelcode"],
                     ascending=[False, True])
        .reset_index(drop=True)
    )

    if return_combos:
        combos = pd.DataFrame(combo_rows)
        return per_artikel, combos
    return per_artikel
