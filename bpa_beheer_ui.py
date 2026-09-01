"""
BPA Jaarlijks Beheer Tool – Streamlit UI
=========================================
Start met:
    streamlit run src/bpa_beheer_ui.py

Vereist:
    pip install streamlit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import json
from io import BytesIO

_ORIGINAL_ST_PYPLOT = getattr(st, "_bpa_original_pyplot", st.pyplot)
st._bpa_original_pyplot = _ORIGINAL_ST_PYPLOT


def _pyplot_zonder_grafiektitel(fig=None, *args, **kwargs):
    """Render een Matplotlib-figuur zonder figuur- of subplot-titels."""
    if fig is not None:
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            suptitle.set_text("")
        for axes in fig.axes:
            axes.set_title("")
    return _ORIGINAL_ST_PYPLOT(fig, *args, **kwargs)


st.pyplot = _pyplot_zonder_grafiektitel

# Hergebruik alle logica uit bpa_beheer.py
from bpa_beheer import (
    laad_config,
    sla_config_op,
    bereken_overzicht,
    bouw_model_kosten,
    laad_excel_onderdelen,
    laad_classificatie_selectie,
    extra_klanten_tot_voorraadstap,
    SERVICE_LEVELS,
    CONFIG_PATH,
    HISTORY_PATH,
    SCRIPT_DIR,
    SELECTIE_PATH,
)
from classificatie import (
    ClassificatieParams,
    voer_classificatie_uit,
    schrijf_selectie_json,
    controleer_kolommen,
    laad_ruwe_dataset,
    laad_erp_documenten,
    df_naar_filtered_excel_bytes,
    bereken_scores,
    pas_basis_filters_toe,
    bouw_selectie_payload,
)
from model import BPAOptimizationModel
OVERZICHT_PATH = os.path.join(SCRIPT_DIR, "bpa_overzicht.json")
OPGESLAGEN_EXCEL_PATH = os.path.join(SCRIPT_DIR, "bpa_laatste_bron.xlsx")

# ══════════════════════════════════════════════════════════════════════════════
#  CACHE-WRAPPERS  (sterk versnellen Streamlit-reruns)
# ══════════════════════════════════════════════════════════════════════════════
#
# Streamlit voert dit script opnieuw uit bij élke widget-interactie. Zonder
# caching wordt de (grote) Excel telkens opnieuw geparsed en doorloopt
# `bereken_overzicht` weer alle componenten. De wrappers hieronder zorgen dat
# we alleen herrekenen als (a) een bron-bestand op disk gewijzigd is óf
# (b) de gebruiker de config heeft aangepast. Cache wordt automatisch
# ongeldig zodra een van die inputs verandert.

def _file_mtime(path: str) -> float:
    """Return mtime in seconds; 0.0 als bestand ontbreekt."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_laad_classificatie_selectie(_mtime: float) -> dict:
    """Cached versie van laad_classificatie_selectie — keyed op bestand-mtime."""
    return laad_classificatie_selectie()


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_laad_ruwe_dataset(excel_bytes: bytes, sheet_name) -> pd.DataFrame:
    """Cache de (trage) Excel-parse voor de classificatie.

    Keyed op bestand-mtime + sheet voor de repo-Excel, of op de geüploade
    file-inhoud (Streamlit hasht een UploadedFile op inhoud, dus de parameter
    krijgt GEEN underscore-prefix — anders zou een tweede upload met dezelfde
    sheet-naam onterecht de vorige cache-hit teruggeven). Hierdoor wordt de
    Excel maar één keer geparsed per uniek bestand; daarna gaan parameter-tweaks
    razendsnel omdat alleen de gevectoriseerde scoring opnieuw draait.
    """
    return laad_ruwe_dataset(BytesIO(excel_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_bereken_overzicht(cfg_json: str, excel_bytes: bytes,
                              _selectie_mtime: float) -> pd.DataFrame:
    """Cached overzicht op basis van config en de geüploade workbook."""
    return bereken_overzicht(json.loads(cfg_json), BytesIO(excel_bytes))


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_maximale_klanten(excel_bytes: bytes) -> pd.Series:
    """M_i uit Final_data, of uit Filtered als Final_data ontbreekt."""
    workbook = pd.ExcelFile(BytesIO(excel_bytes))
    sheet_name = "Final_data" if "Final_data" in workbook.sheet_names else "Filtered "
    df = workbook.parse(sheet_name)
    df = df.rename(columns={
        "Verkooporderregel artikel.Artikel.Artikelcode": "Code",
        "Aantal_klantlocaties_5jr": "M_i",
    })
    if "Code" not in df.columns or "M_i" not in df.columns:
        return pd.Series(dtype=float)
    df["Code"] = df["Code"].astype(str).str.strip()
    df["M_i"] = pd.to_numeric(df["M_i"], errors="coerce")
    return df.dropna(subset=["Code", "M_i"]).groupby("Code")["M_i"].max()


def get_classificatie_info() -> dict:
    """Lees bpa_selectie.json (cached). Auto-invalideert bij file-update."""
    return _cached_laad_classificatie_selectie(_file_mtime(SELECTIE_PATH))


def get_overzicht_df(cfg: dict) -> pd.DataFrame:
    """Bereken het overzicht uit de geüploade workbook (cached)."""
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    return _cached_bereken_overzicht(
        cfg_json,
        st.session_state["bron_excel_bytes"],
        _file_mtime(SELECTIE_PATH),
    )


def invalidate_caches() -> None:
    """Forceer een verse Excel/JSON-read bij volgende aanroep."""
    _cached_bereken_overzicht.clear()
    _cached_maximale_klanten.clear()
    _cached_laad_classificatie_selectie.clear()
    _cached_laad_ruwe_dataset.clear()
def _sla_overzicht_op(df: pd.DataFrame, bron_excel_naam: str) -> None:
    """Bewaar het zichtbare overzicht als duurzame staat voor de volgende sessie."""
    payload = {
        "opgeslagen": str(pd.Timestamp.today()),
        "bron_excel_naam": bron_excel_naam,
        "columns": list(df.reset_index().columns),
        "records": json.loads(df.reset_index().to_json(orient="records")),
    }
    tijdelijk_pad = OVERZICHT_PATH + ".tmp"
    with open(tijdelijk_pad, "w", encoding="utf-8") as bestand:
        json.dump(payload, bestand, ensure_ascii=False, indent=2)
    os.replace(tijdelijk_pad, OVERZICHT_PATH)

def _laad_opgeslagen_overzicht() -> tuple[pd.DataFrame | None, str | None]:
    """Laad de laatst opgeslagen overzichtsstaat, indien beschikbaar."""
    if not os.path.exists(OVERZICHT_PATH):
        return None, None
    try:
        with open(OVERZICHT_PATH, encoding="utf-8") as bestand:
            payload = json.load(bestand)
        df = pd.DataFrame(payload.get("records", []))
        columns = payload.get("columns", [])
        if columns:
            df = df.reindex(columns=columns)
        if "Code" not in df.columns:
            return None, None
        return df.set_index("Code"), payload.get("bron_excel_naam")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, None



def representatieve_z(default: int = 1) -> int:
    """Representatief aantal subscripties (Z) uit het huidige overzicht.

    Vervangt de oude globale standaardwaarde: neemt de mediaan van het
    werkelijke aantal klantlocaties (n_klanten) over alle componenten. Wordt
    gebruikt als referentie-/startwaarde in de gevoeligheidsgrafieken.
    """
    _df = st.session_state.get("overzicht_df")
    if _df is not None and not _df.empty and "n_klanten" in _df.columns:
        _med = pd.to_numeric(_df["n_klanten"], errors="coerce").median()
        if pd.notna(_med) and _med >= 1:
            return int(round(_med))
    return default


def _merge_selectie_payloads(bestaande_selectie: dict, nieuwe_selectie: dict) -> dict:
    """Behoud bestaande items en ververs aanwezige codes met nieuwe Excel-data."""
    bestaande_items = bestaande_selectie.get("items", {})
    if isinstance(bestaande_items, list):
        bestaande_items = {
            str(item.get("code", "")).strip(): item for item in bestaande_items
        }
    gecombineerd = {
        str(code).strip(): item for code, item in bestaande_items.items()
    }
    for item in nieuwe_selectie.get("items", []):
        code = str(item.get("code", "")).strip()
        if code:
            gecombineerd[code] = item

    resultaat = dict(nieuwe_selectie)
    resultaat["items"] = list(gecombineerd.values())
    resultaat["n_items"] = len(gecombineerd)
    resultaat["lt_overzicht"] = {
        status: sum(
            item.get("lt_bron") == status for item in gecombineerd.values()
        )
        for status in ("geupdate", "default", "ontbreekt")
    }
    return resultaat


def _verwijder_uit_selectie(code: str) -> bool:
    """Verwijder een artikelcode uit de actieve selectiewhitelist."""
    if not os.path.exists(SELECTIE_PATH):
        return False
    try:
        with open(SELECTIE_PATH, encoding="utf-8") as selectie_bestand:
            payload = json.load(selectie_bestand)
    except (OSError, json.JSONDecodeError):
        return False

    code = str(code).strip()
    items = payload.get("items", [])
    nieuwe_items = [
        item for item in items
        if str(item.get("code", "")).strip() != code
    ]
    if len(nieuwe_items) == len(items):
        return False

    payload["items"] = nieuwe_items
    payload["n_items"] = len(nieuwe_items)
    payload["selectie_actief"] = True
    payload["lt_overzicht"] = {
        status: sum(item.get("lt_bron") == status for item in nieuwe_items)
        for status in ("geupdate", "default", "ontbreekt")
    }
    schrijf_selectie_json(payload, SELECTIE_PATH)
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PAGINA-INSTELLINGEN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BPA Beheer Tool",
    page_icon="⚙️",
    layout="wide",
)

_logo_path = os.path.join(os.path.dirname(__file__), "BPA.png")
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists(_logo_path):
        st.image(_logo_path, width=120)
with col_title:
    st.title("BPA Jaarlijks Beheer Tool")
    st.caption("Componentselectie en voorraadadvies voor het spare parts team")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG IN SESSION STATE LADEN
# ══════════════════════════════════════════════════════════════════════════════

if "cfg" not in st.session_state:
    st.session_state.cfg = laad_config()

cfg = st.session_state.cfg

# ── Centrale databron ─────────────────────────────────────────────────────
_had_session_source = "bron_excel_bytes" in st.session_state
_opgeslagen_df, _opgeslagen_bron_naam = _laad_opgeslagen_overzicht()
if not os.path.exists(OPGESLAGEN_EXCEL_PATH):
    st.error(
        f"Geen bronbestand gevonden op `{OPGESLAGEN_EXCEL_PATH}`. "
        "Plaats hier het Excelbestand met de 'Filtered '-sheet."
    )
    st.stop()
with open(OPGESLAGEN_EXCEL_PATH, "rb") as _opgeslagen_excel:
    _excel_bytes = _opgeslagen_excel.read()
_bron_excel_naam = _opgeslagen_bron_naam or os.path.basename(OPGESLAGEN_EXCEL_PATH)

try:
    _sheet_names = pd.ExcelFile(BytesIO(_excel_bytes)).sheet_names
except Exception as exc:
    st.error(f"Excelbestand kon niet worden gelezen: {exc}")
    st.stop()
if "Filtered " not in _sheet_names:
    st.error("Het verplichte tabblad 'Filtered ' ontbreekt in het Excelbestand.")
    st.stop()

try:
    _upload_df = laad_ruwe_dataset(BytesIO(_excel_bytes), sheet_name="Filtered ")
except Exception as exc:
    st.error(f"Tabblad 'Filtered ' kon niet worden gelezen: {exc}")
    st.stop()

_overzicht_columns = {
    "Verkooporderregel artikel.Artikel.Artikelcode",
    "Omschrijving_standaard_artikelen",
    "Standaard verkoopprijs",
    "Inkoopprijs (standaard)",
    "Totaal_orders_5jr",
    "Aantal_klantlocaties_5jr",
    "Hoofdleverancier.Levertijd",
    "MTBF(years)",
}
_missing_columns = sorted(
    _overzicht_columns.difference(_upload_df.columns)
    | set(controleer_kolommen(_upload_df))
)
if _missing_columns:
    st.error(
        "De volgende verplichte kolommen ontbreken in 'Filtered ': "
        + ", ".join(_missing_columns)
    )
    st.stop()

if st.session_state.get("bron_excel_bytes") != _excel_bytes:
    st.session_state["bron_excel_bytes"] = _excel_bytes
    st.session_state["bron_excel_naam"] = _bron_excel_naam
    st.session_state.pop("overzicht_df", None)
    st.session_state.pop("cls_result", None)
    invalidate_caches()

_excel_file = BytesIO(_excel_bytes)

# Herstel bij een nieuwe sessie exact de laatst opgeslagen staat. Binnen een
# actieve sessie wordt na een wijziging opnieuw berekend en opgeslagen.
if "overzicht_df" not in st.session_state:
    if not _had_session_source and _opgeslagen_df is not None:
        st.session_state.overzicht_df = _opgeslagen_df
    else:
        with st.spinner("Excel laden en basisvoorraden berekenen…"):
            _df = get_overzicht_df(cfg)
        st.session_state.overzicht_df = _df
        _sla_overzicht_op(_df, _bron_excel_naam)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overzicht, tab_classificatie, tab_subscripties, tab_toevoegen, tab_verwijderen, tab_klanten, tab_historie, tab_kosten, tab_budget, tab_drempel = st.tabs([
    "Overzicht 99%",
    "Componenten selecteren",
    "Gegevens aanpassen",
    "Component toevoegen",
    "Component verwijderen",
    "Klanten & contracten",
    "Historiek",
    "Kostenanalyse",
    "Budget-scenario",
    "Subscriptiedrempel",
])

st.markdown(
    """
    <style>
    div[data-baseweb="tab-list"] button:nth-child(n+7) { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – OVERZICHT
# ─────────────────────────────────────────────────────────────────────────────

with tab_overzicht:
    st.subheader("Voorraadadvies voor 99% beschikbaarheid")

    _col_herb, _col_n1, _col_n_reset, _col_leeg = st.columns(4)
    with _col_herb:
        if st.button("🔄 Herbereken (laadt Excel opnieuw)"):
            invalidate_caches()
            with st.spinner("Berekenen…"):
                df = get_overzicht_df(cfg)
            if df.empty:
                st.warning("Geen onderdelen gevonden.")
            else:
                st.session_state.overzicht_df = df
                _sla_overzicht_op(df, _bron_excel_naam)
                st.rerun()
    with _col_n1:
        if st.button("Stel klantlocaties op 1 voor alle componenten"):
            _codes = (
                st.session_state.overzicht_df.index.astype(str).tolist()
                if "overzicht_df" in st.session_state and not st.session_state.overzicht_df.empty
                else []
            )
            cfg.setdefault("n_klanten_overrides", {})
            for _c in _codes:
                cfg["n_klanten_overrides"][_c] = 1
            sla_config_op(cfg, BytesIO(_excel_bytes))
            invalidate_caches()
            st.session_state.pop("overzicht_df", None)
            st.toast(f"Aantal klantlocaties voor {len(_codes)} componenten op 1 gezet.")
            st.rerun()
    with _col_n_reset:
        if st.button("Herstel klantlocaties uit Excel"):
            cfg["n_klanten_overrides"] = {}
            sla_config_op(cfg, BytesIO(_excel_bytes))
            invalidate_caches()
            st.session_state.pop("overzicht_df", None)
            st.toast("De aantallen klantlocaties uit Excel worden weer gebruikt.")
            st.rerun()
    with _col_leeg:
        if st.button("Overzicht helemaal legen"):
            try:
                schrijf_selectie_json({
                    "gegenereerd": str(pd.Timestamp.today()),
                    "selectie_actief": True,
                    "n_items": 0,
                    "items": [],
                    "lt_overzicht": {},
                }, SELECTIE_PATH)
                invalidate_caches()
                st.session_state.pop("overzicht_df", None)
                st.session_state.pop("cls_payload", None)
                st.toast("Het overzicht is helemaal geleegd.", icon="🗑️")
                st.rerun()
            except Exception as e:
                st.error(f"Kon het overzicht niet legen: {e}")

    if "overzicht_df" in st.session_state:
        df = st.session_state.overzicht_df
        sl_cols = ["s@99.0%"] if "s@99.0%" in df.columns else []

        # Samenvattingsregel
        totals = {c: int(df[c].sum()) for c in sl_cols}
        st.write(f"**Totale adviesvoorraad bij 99% beschikbaarheid: {totals.get('s@99.0%', 0)} stuks**")

        # Aandeel S* > 1 — extra voorraadkosten bovenop S*=1
        if sl_cols and 'IP' in df.columns:
            _parts = []
            for _sc in sl_cols:
                _ip_vals     = df['IP'].fillna(0)
                _extra_units = (df[_sc] - 1).clip(lower=0)          # max(S*-1, 0) per component
                _base_cost   = _ip_vals.sum()                        # Σ 1 × IP (S*=1 scenario)
                _extra_cost  = (_extra_units * _ip_vals).sum()       # Σ (S*-1) × IP
                _total_cost  = (df[_sc] * _ip_vals).sum()
                _pct_extra   = _extra_cost / _total_cost * 100 if _total_cost > 0 else 0.0
                _n_gt1       = int((df[_sc] > 1).sum())
                _parts.append(
                    f"**{_n_gt1}** componenten vragen meer dan één stuk voorraad. "
                    f"De extra voorraadwaarde boven één stuk per component is "
                    f"**€ {_extra_cost:,.0f}** (**{_pct_extra:.1f}%** van de totale voorraadwaarde)."
                )
            if _parts:
                st.caption("  \n".join(_parts))

        # Laad vorige snapshot voor Δ-kolommen
        _prev_comp = {}
        _prev_datum = None
        # 1) Voorkeur: vorige overzicht_df uit session_state (vastgelegd bij opslaan)
        _prev_df = st.session_state.get("overzicht_df_prev")
        if _prev_df is not None and not _prev_df.empty:
            _prev_datum = "vorige opgeslagen staat"
            for _code in _prev_df.index:
                _prev_comp[str(_code)] = {
                    _sc: int(_prev_df.at[_code, _sc])
                    for _sc in _prev_df.columns if _sc.startswith("s@")
                }
        # 2) Fallback: oudere snapshot uit history-bestand (legacy)
        elif os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, encoding='utf-8') as _fh:
                    _hist_ov = json.load(_fh)
                for _snap_ov in reversed(_hist_ov):
                    if 'componenten' in _snap_ov:
                        _prev_comp  = _snap_ov['componenten']
                        _prev_datum = _snap_ov['datum']
                        break
            except Exception:
                pass

        # Bouw weergave-df met Δ-kolommen
        _df_disp = df.reset_index().copy()

        # Klantlocaties komt uitsluitend uit de koppelingen in 'Klanten & contracten'
        _klant_telling_ov = {}
        for _artikelen_per_klant_ov in cfg.get("klanten", {}).values():
            for _item_ov in _artikelen_per_klant_ov:
                _code_ov = str(_item_ov.get("code"))
                _klant_telling_ov[_code_ov] = _klant_telling_ov.get(_code_ov, 0) + 1
        _df_disp["n_klanten"] = (
            _df_disp["Code"].astype(str).map(_klant_telling_ov).fillna(0).astype(int)
        )

        # Huidige (fysieke) voorraad — handmatig bijgewerkt door BPA
        cfg.setdefault("voorraad_actueel", {})
        _df_disp["Voorraad_actueel"] = (
            _df_disp["Code"].astype(str).map(cfg["voorraad_actueel"]).astype("Int64")
        )

        _df_disp["+Klanten"] = pd.array([
            extra_klanten_tot_voorraadstap(
                int(row["n_klanten"]),
                float(row["lambda_jr"]),
                int(row["LT_dagen"]),
            )
            for _, row in _df_disp.iterrows()
        ], dtype="Int64")
        _delta_cols = []
        # Vectoriseer: bouw één lookup-DataFrame van vorige S*-waarden per Code,
        # zodat we per SL-kolom alleen een Series-aftrekking nodig hebben
        # (i.p.v. .apply(axis=1) — orde van grootte sneller bij veel rijen).
        if _prev_comp and sl_cols:
            _prev_df_lookup = (
                pd.DataFrame.from_dict(_prev_comp, orient="index")
                  .reindex(columns=sl_cols)
                  .apply(pd.to_numeric, errors="coerce")
            )
            _codes_str = _df_disp["Code"].astype(str)
            for _sc in sl_cols:
                _dc = f"\u0394{_sc}"
                _delta_cols.append(_dc)
                _prev_series = _codes_str.map(_prev_df_lookup[_sc])
                _df_disp[_dc] = (
                    pd.to_numeric(_df_disp[_sc], errors="coerce") - _prev_series
                )
        else:
            # Geen vorige snapshot beschikbaar — vul Δ-kolommen met NaN
            for _sc in sl_cols:
                _dc = f"\u0394{_sc}"
                _delta_cols.append(_dc)
                _df_disp[_dc] = float("nan")

        def _fmt_delta(v):
            if pd.isna(v): return '\u2014'
            iv = int(v)
            return f"+{iv}" if iv > 0 else str(iv)

        def _style_delta(v):
            if pd.isna(v): return ''
            if v > 0: return 'background-color: #f8d7da'   # rood: omhoog
            if v < 0: return 'background-color: #cce5ff'   # blauw: omlaag
            return 'background-color: #d4edda'              # groen: gelijk

        # ── Investering per component ──────────────────────────────────────
        _kp_ov = st.session_state.get('kosten_params', {})
        _sl_ov = 0.990
        _sl_ov_col = "s@99.0%"
        if _sl_ov_col in df.columns and 'IP' in df.columns:
            _df_disp['Inv. (€)'] = (_df_disp[_sl_ov_col] * df['IP'].values).round(2)
            _inv_totaal = _df_disp['Inv. (€)'].sum()
            _df_disp['Inv. %'] = (
                (_df_disp['Inv. (€)'] / _inv_totaal * 100).round(1)
                if _inv_totaal > 0 else 0.0
            )
            _inv_cols = ['Inv. (€)', 'Inv. %']
            st.caption(
                f"Totale voorraadwaarde bij 99% beschikbaarheid: **€ {_inv_totaal:,.0f}**"
            )
        else:
            _inv_cols = []

        # Tabel
        _fmt_inv  = {c: "{:.0f}" for c in _inv_cols if 'Inv. (€)' in c}
        _fmt_inv |= {c: "{:.1f}%" for c in _inv_cols if 'Inv. %' in c}

        def _style_inv_share(v):
            if pd.isna(v) or _inv_totaal == 0:
                return ''
            intensity = min(int(v / 100 * 255), 255)
            return f'background-color: rgba(25, 118, 210, {v/100:.2f}); color: {"white" if v > 50 else "black"}'

        def _style_klanten_tot_extra_stuk(v):
            if pd.notna(v) and int(v) in (1, 2):
                return 'background-color: #f8d7da; color: #842029; font-weight: 600'
            return ''

        # ── LT-status kolom (vanuit classificatie-koppeling) ──────────────
        _LT_ICOON = {
            'geupdate':  '✅ geupdate',
            'override':  '✏️ override',
            'default':   '⚠️ ERP-default',
            'ontbreekt': '❌ ontbreekt',
            'handmatig': '🛠 handmatig',
            'onbekend':  '❔ onbekend',
            'nul→30':   '🔵 0→30 dagen',
        }
        if 'LT_bron' in _df_disp.columns:
            _df_disp['LT-status'] = (
                _df_disp['LT_bron'].astype(str).map(_LT_ICOON).fillna(_LT_ICOON['onbekend'])
            )

            def _kleur_lt(v):
                s = str(v)
                if '🔵' in s:             return 'background-color: #bbdefb'  # blauw: LT was 0 → 30
                if '✅' in s or '✏️' in s: return 'background-color: #e8f5e9'
                if '⚠️' in s:              return 'background-color: #fff8e1'
                if '❌' in s:              return 'background-color: #ffebee'
                if '🛠' in s:              return 'background-color: #e3f2fd'
                return ''

            _n_bevest = _df_disp['LT_bron'].isin(['geupdate', 'override', 'handmatig', 'nul→30']).sum()
            _n_warn   = len(_df_disp) - _n_bevest
            if _n_warn > 0:
                st.warning(
                    f"⚠️ {_n_warn}/{len(_df_disp)} componenten hebben een niet-bevestigde "
                    f"levertijd (standaardwaarde uit ERP of ontbrekend). Corrigeer dit via "
                    f"**Gegevens aanpassen**."
                )

        if "Cls_score" in _df_disp.columns:
            _df_disp = _df_disp.sort_values(
                "Cls_score", ascending=False, na_position="last", kind="stable"
            )
            _df_disp.insert(0, "Rang", range(1, len(_df_disp) + 1))

        _friendly_columns = {
            "Code": "Artikelnummer",
            "Descr": "Omschrijving",
            "Cls_score": "Prioriteitsscore",
            "n_klanten": "Klantlocaties",
            "+Klanten": "Extra klanten tot extra stuk voorraad",
            "lambda_jr": "Verwachte aanvragen per jaar",
            "LT_dagen": "Levertijd (dagen)",
            "IP": "Inkoopprijs",
            "VP": "Verkoopprijs",
            "s@99.0%": "Adviesvoorraad (99%)",
            "Voorraad_actueel": "Huidige voorraad",
            "Inv. (€)": "Voorraadwaarde",
            "Inv. %": "Aandeel voorraadwaarde",
        }
        _visible_columns = [
            c for c in [
                "Rang", "Code", "Descr", "Cls_score", "n_klanten", "+Klanten", "lambda_jr",
                "LT_dagen", "IP", "VP", "s@99.0%", "Voorraad_actueel", "Inv. (€)", "Inv. %",
            ] if c in _df_disp.columns
        ]
        _df_disp = _df_disp[_visible_columns].rename(columns=_friendly_columns)

        styled = (
            _df_disp.style
                .format({
                    "Prioriteitsscore": "{:.1f}",
                    "Extra klanten tot extra stuk voorraad": "{:.0f}",
                    "Verwachte aanvragen per jaar": "{:.2f}",
                    "Inkoopprijs": "€ {:,.2f}",
                    "Verkoopprijs": "€ {:,.2f}",
                    "Adviesvoorraad (99%)": "{:.0f}",
                    "Huidige voorraad": lambda v: "–" if pd.isna(v) else f"{int(v):.0f}",
                    "Voorraadwaarde": "€ {:,.0f}",
                    "Aandeel voorraadwaarde": "{:.1f}%",
                })
        )
        if 'Aandeel voorraadwaarde' in _df_disp.columns:
            styled = styled.map(_style_inv_share, subset=['Aandeel voorraadwaarde'])
        if 'Extra klanten tot extra stuk voorraad' in _df_disp.columns:
            styled = styled.map(
                _style_klanten_tot_extra_stuk,
                subset=['Extra klanten tot extra stuk voorraad'],
            )

        def _style_voorraad_tekort(row):
            styles = [''] * len(row)
            if 'Huidige voorraad' in row.index and 'Adviesvoorraad (99%)' in row.index:
                idx = row.index.get_loc('Huidige voorraad')
                huidig, advies = row['Huidige voorraad'], row['Adviesvoorraad (99%)']
                if pd.notna(huidig) and pd.notna(advies):
                    if huidig < advies:
                        styles[idx] = 'background-color: #f8d7da; color: #842029; font-weight: 600'
                    elif huidig > advies:
                        styles[idx] = 'background-color: #d4edda'
            return styles

        if 'Huidige voorraad' in _df_disp.columns:
            styled = styled.apply(_style_voorraad_tekort, axis=1)

        st.dataframe(styled, use_container_width=True, height=500)

        # Download
        csv = _df_disp.to_csv(sep=";", decimal=",", index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download als CSV",
            data=csv,
            file_name=f"bpa_base_stock_{date.today()}.csv",
            mime="text/csv",
        )

        # ── Huidige voorraad bijwerken ───────────────────────────────────────
        st.divider()
        st.subheader("📦 Huidige voorraad bijwerken")
        st.caption(
            "Vul hier de werkelijke fysieke voorraad per artikel in. BPA werkt dit zelf "
            "bij; de kolom 'Huidige voorraad' hierboven gebruikt deze waarden."
        )

        _voorraad_rows = [
            {"Artikelcode": code, "Huidige voorraad": cfg["voorraad_actueel"].get(str(code))}
            for code in _df_disp["Artikelnummer"]
        ] if "Artikelnummer" in _df_disp.columns else []

        _voorraad_edited = st.data_editor(
            pd.DataFrame(_voorraad_rows) if _voorraad_rows else pd.DataFrame(
                columns=["Artikelcode", "Huidige voorraad"]
            ),
            use_container_width=True,
            hide_index=True,
            disabled=["Artikelcode"],
            column_config={
                "Artikelcode": st.column_config.TextColumn("Artikelcode"),
                "Huidige voorraad": st.column_config.NumberColumn(
                    "Huidige voorraad", min_value=0, step=1,
                ),
            },
            key="voorraad_actueel_editor",
        )
        if st.button("💾 Huidige voorraad opslaan"):
            _nieuwe_voorraad = {}
            for _, _row_va in _voorraad_edited.iterrows():
                _code_va = _row_va.get("Artikelcode")
                if not _code_va or pd.isna(_code_va):
                    continue
                if pd.notna(_row_va["Huidige voorraad"]):
                    _nieuwe_voorraad[str(_code_va)] = int(_row_va["Huidige voorraad"])
            cfg["voorraad_actueel"] = _nieuwe_voorraad
            sla_config_op(cfg, BytesIO(_excel_bytes))
            st.toast(f"Huidige voorraad opgeslagen voor {len(_nieuwe_voorraad)} artikelen.", icon="✅")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – SUBSCRIPTIES / IP / LEVERTIJD AANPASSEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_subscripties:
    st.subheader("Gegevens aanpassen")
    st.info(
        "Het aantal subscripties (Z_i) per component komt uit het werkelijke "
        "aantal klantlocaties en wordt hier niet aangepast.",
        icon="ℹ️",
    )

    st.divider()
    st.subheader("Overrides per artikelcode")
    st.caption("VP = verkoopprijs (€), IP = inkoopprijs (€), LT = levertijd (dagen). "
               "Laat een cel leeg om de brondata-waarde te gebruiken.")

    cfg.setdefault("ip_overrides", {})
    cfg.setdefault("vp_overrides", {})
    cfg.setdefault("lt_overrides", {})

    # Bouw gecombineerde tabel van alle codes met minstens één override
    alle_codes = sorted(
        set(cfg["ip_overrides"]) |
        set(cfg["vp_overrides"]) |
        set(cfg["lt_overrides"])
    )
    override_rows = [
        {
            "Artikelcode": c,
            "VP (€)":      cfg["vp_overrides"].get(c),
            "IP (€)":      cfg["ip_overrides"].get(c),
            "LT (dagen)":  cfg["lt_overrides"].get(c),
        }
        for c in alle_codes
    ]

    edited = st.data_editor(
        pd.DataFrame(override_rows) if override_rows else pd.DataFrame(
            columns=["Artikelcode", "VP (€)", "IP (€)", "LT (dagen)"]
        ),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Artikelcode": st.column_config.TextColumn("Artikelcode", required=True),
            "VP (€)":      st.column_config.NumberColumn("VP (€)", min_value=0.0, format="%.2f"),
            "IP (€)":      st.column_config.NumberColumn("IP (€)", min_value=0.0, format="%.2f"),
            "LT (dagen)":  st.column_config.NumberColumn("LT (dagen)", min_value=1, step=1),
        },
        key="overrides_editor",
    )

    if st.button("💾 Opslaan overrides"):
        ip_ov, vp_ov, lt_ov = {}, {}, {}
        for _, row in edited.iterrows():
            code = row.get("Artikelcode")
            if not code or pd.isna(code):
                continue
            code = str(code)
            if pd.notna(row["VP (€)"]):
                vp_ov[code] = float(row["VP (€)"])
            if pd.notna(row["IP (€)"]):
                ip_ov[code] = float(row["IP (€)"])
            if pd.notna(row["LT (dagen)"]):
                lt_ov[code] = int(row["LT (dagen)"])
        cfg["ip_overrides"]        = ip_ov
        cfg["vp_overrides"]        = vp_ov
        cfg["lt_overrides"]        = lt_ov
        # Bewaar huidige overzicht_df als vorige snapshot vóór recompute
        if "overzicht_df" in st.session_state:
            st.session_state.overzicht_df_prev = st.session_state.overzicht_df.copy()
        sla_config_op(cfg, BytesIO(_excel_bytes))
        st.toast(f"Overrides opgeslagen — {len(vp_ov)} VP, {len(ip_ov)} IP, {len(lt_ov)} LT.", icon="✅")
        st.session_state.pop("overzicht_df", None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 – COMPONENT TOEVOEGEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_toevoegen:
    st.subheader("Nieuw component toevoegen")
    st.write("Gebruik dit voor componenten die nog niet in de Excel staan.")

    with st.form("form_toevoegen"):
        col1, col2 = st.columns(2)
        with col1:
            f_code  = st.text_input("Artikelcode *")
            f_descr = st.text_input("Omschrijving")
            f_lam   = st.number_input(
                "Lambda – vraag per jaar *",
                min_value=0.0001, value=1.0, step=0.1, format="%.4f",
            )
        with col2:
            f_lt = st.number_input(
                "Levertijd leverancier → BPA (dagen) *",
                min_value=1, value=30, step=1,
            )
            f_n = st.number_input(
                "Aantal subscripties (Z)",
                min_value=1, value=1, step=1,
            )
            f_ip = st.number_input(
                "Inkoopprijs (€)", min_value=0.0, value=0.0, step=10.0, format="%.2f",
            )
        submitted = st.form_submit_button("➕ Component opslaan")

    if submitted:
        if not f_code:
            st.error("Artikelcode is verplicht.")
        elif f_code in cfg["handmatige_componenten"]:
            st.warning(f"'{f_code}' bestaat al. Verwijder het eerst via het tabblad 'Component verwijderen'.")
        else:
            cfg["handmatige_componenten"][f_code] = {
                "descr":           f_descr,
                "lambda_per_jaar": float(f_lam),
                "lt_dagen":        int(f_lt),
                "n_klanten":       int(f_n),
                "ip":              float(f_ip),
            }
            sla_config_op(cfg, BytesIO(_excel_bytes))
            st.success(f"Component '{f_code}' toegevoegd.")

            # Preview berekende basisvoorraden
            lt_jr = int(f_lt) / 365
            preview = {
                f"s@{sl:.1%}": BPAOptimizationModel.inverse_service_level(sl, float(f_lam), lt_jr)
                for sl in SERVICE_LEVELS
            }
            st.write("**Berekende basisvoorraden voor dit component:**")
            st.dataframe(pd.DataFrame([preview]), use_container_width=False)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 – COMPONENT VERWIJDEREN
# ─────────────────────────────────────────────────────────────────────────────

with tab_verwijderen:
    st.subheader("Component verwijderen uit overzicht")
    st.write("Het component verdwijnt uit het overzicht en uit 'Al in overzicht'. "
             "Een Excel-component kan daarna opnieuw worden geselecteerd.")

    handmatig   = cfg["handmatige_componenten"]
    uitgesloten = cfg.setdefault("uitgesloten_componenten", [])

    # Gebruik dezelfde codes als in het Overzicht-tab (= classificatie-whitelist
    # toegepast, inclusief synthetische classificatie-rijen).
    _ov_df = st.session_state.get("overzicht_df")
    if _ov_df is None or _ov_df.empty:
        try:
            _ov_df = get_overzicht_df(cfg)
            st.session_state["overzicht_df"] = _ov_df
        except Exception:
            _ov_df = pd.DataFrame()

    if _ov_df is not None and not _ov_df.empty and "bron" in _ov_df.columns:
        excel_codes = [str(c) for c, b in zip(_ov_df.index, _ov_df["bron"])
                       if b in ("excel", "classificatie")]
    else:
        excel_codes = []

    # Alle actieve codes met bron
    opties = (
        [(c, "handmatig", handmatig[c].get("descr", "")) for c in handmatig if c not in uitgesloten] +
        [(c, "excel",     "") for c in excel_codes if c not in handmatig and c not in uitgesloten]
    )

    if not opties:
        st.info("Geen actieve componenten om te verwijderen.")
    else:
        keuze = st.selectbox(
            "Selecteer component",
            options=[c for c, _, _ in opties],
            format_func=lambda c: next(
                f"{c}  [{bron}]  {descr}" for code, bron, descr in opties if code == c
            ),
        )
        bron_keuze = next(bron for c, bron, _ in opties if c == keuze)
        if bron_keuze == "handmatig":
            v = handmatig[keuze]
            st.write(f"**{keuze}** (handmatig) &nbsp;|&nbsp; λ = {v['lambda_per_jaar']:.4f}/jr "
                     f"&nbsp;|&nbsp; LT = {v['lt_dagen']} d")
            st.warning("Dit component wordt permanent verwijderd.")
            if st.button("🗑️ Verwijder permanent", type="primary"):
                del cfg["handmatige_componenten"][keuze]
                _verwijder_uit_selectie(keuze)
                if keuze in uitgesloten:
                    uitgesloten.remove(keuze)
                sla_config_op(cfg, BytesIO(_excel_bytes))
                invalidate_caches()
                st.session_state.pop("overzicht_df", None)
                st.session_state.pop("cls_result", None)
                st.success(f"'{keuze}' verwijderd.")
                st.rerun()
        else:
            st.write(f"**{keuze}** (uit Excel) wordt uit de huidige selectie verwijderd.")
            st.info("Het artikel blijft in Excel staan en is na verwijdering opnieuw selecteerbaar.")
            if st.button("🗑️ Uit selectie verwijderen", type="primary"):
                _verwijder_uit_selectie(keuze)
                if keuze in uitgesloten:
                    uitgesloten.remove(keuze)
                sla_config_op(cfg, BytesIO(_excel_bytes))
                invalidate_caches()
                st.session_state.pop("overzicht_df", None)
                st.session_state.pop("cls_result", None)
                st.success(f"'{keuze}' uit de selectie verwijderd en weer selecteerbaar gemaakt.")
                st.rerun()

    # Uitgesloten Excel-componenten terugzetten
    if uitgesloten:
        st.divider()
        st.subheader("Uitgesloten componenten terugzetten")
        terugzetten = st.selectbox(
            "Selecteer component om terug te zetten",
            options=uitgesloten,
            key="terugzetten_selectbox",
        )
        if st.button("↩️ Zet terug in model"):
            uitgesloten.remove(terugzetten)
            sla_config_op(cfg, BytesIO(_excel_bytes))
            st.success(f"'{terugzetten}' is weer actief.")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4A – KLANTEN & CONTRACTEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_klanten:
    st.subheader("Klanten en gecontracteerde artikelen")
    st.write("Voeg klanten toe en koppel per klant de artikelen uit het overzicht, "
             "met de ingangsdatum van het contract.")

    klanten = cfg.setdefault("klanten", {})

    with st.form("form_klant_toevoegen", clear_on_submit=True):
        f_klant_naam = st.text_input("Naam nieuwe klant *")
        klant_submitted = st.form_submit_button("➕ Klant toevoegen")
    if klant_submitted:
        naam = f_klant_naam.strip()
        if not naam:
            st.error("Naam is verplicht.")
        elif naam in klanten:
            st.warning(f"'{naam}' bestaat al.")
        else:
            klanten[naam] = []
            sla_config_op(cfg, BytesIO(_excel_bytes))
            st.success(f"Klant '{naam}' toegevoegd.")
            st.rerun()

    if not klanten:
        st.info("Nog geen klanten toegevoegd.")
    else:
        st.divider()
        st.subheader("Artikelen toevoegen aan klant")

        _handmatig_kl   = cfg["handmatige_componenten"]
        _uitgesloten_kl = cfg.get("uitgesloten_componenten", [])
        _ov_df_kl = st.session_state.get("overzicht_df")
        if _ov_df_kl is None or _ov_df_kl.empty:
            try:
                _ov_df_kl = get_overzicht_df(cfg)
                st.session_state["overzicht_df"] = _ov_df_kl
            except Exception:
                _ov_df_kl = pd.DataFrame()

        _descr_map_kl = {}
        if _ov_df_kl is not None and not _ov_df_kl.empty and "bron" in _ov_df_kl.columns:
            _excel_codes_kl = [str(c) for c, b in zip(_ov_df_kl.index, _ov_df_kl["bron"])
                               if b in ("excel", "classificatie")]
            if "Descr" in _ov_df_kl.columns:
                _descr_map_kl = {str(c): str(d) for c, d in zip(_ov_df_kl.index, _ov_df_kl["Descr"])}
        else:
            _excel_codes_kl = []

        _alle_codes_kl = sorted(set(
            [c for c in _handmatig_kl if c not in _uitgesloten_kl]
            + [c for c in _excel_codes_kl if c not in _uitgesloten_kl]
        ))

        if not _alle_codes_kl:
            st.info("Geen actieve componenten beschikbaar om te koppelen.")
        else:
            _klant_keuze = st.selectbox("Klant", options=sorted(klanten.keys()), key="klant_keuze_toevoegen")
            f_codes_kl = st.multiselect(
                "Artikelen",
                options=_alle_codes_kl,
                format_func=lambda c: f"{c}  {_descr_map_kl.get(c) or _handmatig_kl.get(c, {}).get('descr', '')}".strip(),
                key="klant_artikelen_multiselect",
            )
            _col_ingang_kl, _col_factuur_kl = st.columns(2)
            with _col_ingang_kl:
                f_ingangsdatum_kl = st.date_input("Ingangsdatum contract", value=date.today())
            with _col_factuur_kl:
                f_facturatiedatum_kl = st.date_input(
                    "Facturatiedatum (jaarlijks)", value=f_ingangsdatum_kl,
                    help="Vanaf deze datum wordt het component elk jaar opnieuw gefactureerd.",
                )
            if st.button("➕ Toevoegen aan klant", disabled=not f_codes_kl):
                _bestaand_kl = {item["code"]: item for item in klanten[_klant_keuze]}
                for code in f_codes_kl:
                    _bestaand_kl[code] = {
                        "code": code,
                        "descr": _descr_map_kl.get(code) or _handmatig_kl.get(code, {}).get("descr", ""),
                        "ingangsdatum": str(f_ingangsdatum_kl),
                        "facturatiedatum": str(f_facturatiedatum_kl),
                    }
                klanten[_klant_keuze] = list(_bestaand_kl.values())
                sla_config_op(cfg, BytesIO(_excel_bytes))
                st.success(f"{len(f_codes_kl)} artikel(en) gekoppeld aan '{_klant_keuze}' vanaf {f_ingangsdatum_kl}.")
                st.rerun()

        st.divider()
        st.subheader("Overzicht per klant")
        for _klant_naam in sorted(klanten.keys()):
            _artikelen_kl = klanten[_klant_naam]
            with st.expander(f"{_klant_naam}  ({len(_artikelen_kl)} artikel(en))"):
                if _artikelen_kl:
                    _df_klant = pd.DataFrame(_artikelen_kl).rename(columns={
                        "code": "Artikelnummer",
                        "descr": "Omschrijving",
                        "ingangsdatum": "Ingangsdatum contract",
                        "facturatiedatum": "Facturatiedatum (jaarlijks)",
                    })
                    st.dataframe(_df_klant, use_container_width=True)

                    f_verwijder_code_kl = st.selectbox(
                        "Artikel verwijderen bij deze klant",
                        options=[a["code"] for a in _artikelen_kl],
                        key=f"verwijder_artikel_{_klant_naam}",
                    )
                    if st.button("🗑️ Artikel verwijderen", key=f"btn_verwijder_artikel_{_klant_naam}"):
                        klanten[_klant_naam] = [
                            a for a in _artikelen_kl if a["code"] != f_verwijder_code_kl
                        ]
                        sla_config_op(cfg, BytesIO(_excel_bytes))
                        st.success(f"'{f_verwijder_code_kl}' verwijderd bij '{_klant_naam}'.")
                        st.rerun()
                else:
                    st.write("Nog geen artikelen gekoppeld.")

                if st.button("🗑️ Klant verwijderen", key=f"btn_verwijder_klant_{_klant_naam}"):
                    del klanten[_klant_naam]
                    sla_config_op(cfg, BytesIO(_excel_bytes))
                    st.success(f"Klant '{_klant_naam}' verwijderd.")
                    st.rerun()

        # ── Aankomende facturatie ────────────────────────────────────────────
        st.divider()
        st.subheader("📅 Aankomende facturatie (komende 30 dagen)")
        st.caption("Jaarprijs per gecontracteerd onderdeel = α × verkoopprijs, met α = 11%.")

        _ALPHA_FACTURATIE = 0.11
        _prijs_map_kl = {}
        if _ov_df_kl is not None and not _ov_df_kl.empty and "VP" in _ov_df_kl.columns:
            _prijs_map_kl = {
                str(c): float(v) * _ALPHA_FACTURATIE
                for c, v in zip(_ov_df_kl.index, _ov_df_kl["VP"])
            }

        def _volgende_factuurdatum(datum_str, vandaag):
            """Eerstvolgende jaarlijkse herhaling van datum_str, op of na vandaag."""
            try:
                _basis = date.fromisoformat(str(datum_str))
            except (TypeError, ValueError):
                return None
            for _jaar in (vandaag.year, vandaag.year + 1):
                try:
                    _kandidaat = _basis.replace(year=_jaar)
                except ValueError:
                    _kandidaat = _basis.replace(year=_jaar, day=28)  # 29 feb in niet-schrikkeljaar
                if _kandidaat >= vandaag:
                    return _kandidaat
            return None

        _vandaag_kl = date.today()
        _horizon_kl = _vandaag_kl + timedelta(days=30)
        _facturatie_rows = []
        for _klant_naam_f, _artikelen_f in klanten.items():
            for _item_f in _artikelen_f:
                _factuurbron = _item_f.get("facturatiedatum") or _item_f.get("ingangsdatum")
                _volgende = _volgende_factuurdatum(_factuurbron, _vandaag_kl)
                if _volgende is not None and _volgende <= _horizon_kl:
                    _facturatie_rows.append({
                        "Klant": _klant_naam_f,
                        "Artikelnummer": _item_f.get("code"),
                        "Omschrijving": _item_f.get("descr", ""),
                        "Volgende factuurdatum": _volgende,
                        "Bedrag": _prijs_map_kl.get(str(_item_f.get("code")), 0.0),
                    })

        if not _facturatie_rows:
            st.info("Geen klanten met een factuur in de komende 30 dagen.")
        else:
            _df_facturatie = pd.DataFrame(_facturatie_rows).sort_values("Volgende factuurdatum")
            st.dataframe(
                _df_facturatie.style.format({"Bedrag": "€ {:,.2f}"}),
                use_container_width=True,
            )
            _totaal_per_klant = (
                _df_facturatie.groupby("Klant", as_index=False)["Bedrag"].sum()
                .rename(columns={"Bedrag": "Totaalbedrag"})
            )
            st.write("**Totaalbedrag per klant:**")
            st.dataframe(
                _totaal_per_klant.style.format({"Totaalbedrag": "€ {:,.2f}"}),
                use_container_width=True,
            )
            st.caption(
                f"Totaal te factureren in de komende 30 dagen: € {_df_facturatie['Bedrag'].sum():,.2f}"
            )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 – HISTORIEK
# ─────────────────────────────────────────────────────────────────────────────

with tab_historie:
    st.subheader("Historiek basisvoorraden")
    st.caption("Elke keer dat je een wijziging opslaat wordt automatisch een snapshot bewaard.")

    if not os.path.exists(HISTORY_PATH):
        st.info("Nog geen historiek beschikbaar. Sla een wijziging op om de eerste snapshot te maken.")
    else:
        with open(HISTORY_PATH, encoding="utf-8") as _f:
            history = json.load(_f)

        if not history:
            st.info("Nog geen snapshots.")
        else:
            # Bouw DataFrame op
            rows = []
            for h in history:
                row = {"Datum": h["datum"], "Z": h["n_klanten"], "# componenten": h["n_actief"]}
                row.update(h.get("totalen", {}))
                rows.append(row)
            hist_df = pd.DataFrame(rows).set_index("Datum")

            sl_cols = [c for c in hist_df.columns if c.startswith("s@")]

            # Grafiek
            if sl_cols:
                import matplotlib.pyplot as plt
                import matplotlib.ticker as ticker

                fig, ax = plt.subplots(figsize=(10, 4))
                for col in sl_cols:
                    ax.plot(hist_df.index, hist_df[col], marker="o", linewidth=2, label=col)
                    for x, y in zip(hist_df.index, hist_df[col]):
                        if pd.notna(y):
                            ax.annotate(str(int(y)), (x, y), textcoords="offset points",
                                        xytext=(0, 6), ha="center", fontsize=8)

                ax.set_xlabel("Update date", fontsize=11)
                ax.set_ylabel("Total base stock (units)", fontsize=11)
                ax.set_title("Total BPA base stock per update moment", fontsize=12)
                ax.legend(fontsize=9)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Tabel
            st.write("**Snapshots:**")
            st.dataframe(hist_df.reset_index(), use_container_width=True)

            # Snapshot handmatig toevoegen (huidige staat)
            st.divider()
            if st.button("📸 Voeg snapshot toe van huidige staat"):
                from bpa_beheer import _sla_history_snapshot
                _sla_history_snapshot(cfg)
                st.success("Snapshot toegevoegd.")
                st.rerun()

    # ── Sensitivity grafieken ──────────────────────────────────────────────
    st.divider()
    st.subheader("Sensitivity grafieken")

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.info("Laad het overzicht (tabblad 📊) om de sensitivity grafieken te berekenen.")
    else:
        # Haal draaiknoppen op uit Kostenanalyse; gebruik defaults als nog niet berekend
        _kp = st.session_state.get('kosten_params', {})
        _ALPHA_DEF      = _kp.get('alpha',     0.15)
        _KAPPA_BPA_DEF  = _kp.get('kappa_bpa', 0.20)
        _KAPPA_C_DEF    = _kp.get('kappa_c',   0.25)

        st.caption(
            f"Vaste waarden buiten de gesweepte parameter: "
            f"α = **{_ALPHA_DEF:.0%}**, κ\\_BPA = **{_KAPPA_BPA_DEF:.0%}**, "
            f"κ\\_c = **{_KAPPA_C_DEF:.0%}**, N = standaard uit overzicht. "
            f"_(pas aan via tabblad 💰 Kostenanalyse)_"
        )

        _SL_SWEEP_S     = SERVICE_LEVELS
        _N_VALS         = [1, 2, 5, 10, 50]
        _ALPHA_SWEEP_S  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        _SL_ALPHA       = [sl for sl in SERVICE_LEVELS if sl >= 0.98]

        if st.button("📊 Bereken sensitivity grafieken"):
            _ov = st.session_state.overzicht_df
            _g1 = {n: [] for n in _N_VALS}
            _g2 = []
            _g3 = {sl: [] for sl in _SL_ALPHA}

            with st.spinner("Berekenen (kan even duren)…"):
                # Grafieken 1 & 2: sweep over service levels
                for _sl in _SL_SWEEP_S:
                    try:
                        _m2, _ = bouw_model_kosten(_ov, _ALPHA_DEF, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl)
                        _g2.append({'sl': _sl, 'base': sum(_m2.calculate_base_stock_levels().values())})
                    except Exception:
                        _g2.append({'sl': _sl, 'base': None})
                    for _n in _N_VALS:
                        try:
                            _, _r1 = bouw_model_kosten(
                                _ov, _ALPHA_DEF, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl,
                                n_klanten_override=_n,
                            )
                            _g1[_n].append({'sl': _sl, 'marge': _r1['bpa_margin']})
                        except Exception:
                            _g1[_n].append({'sl': _sl, 'marge': None})
                # Grafiek 3: sweep over alpha per SL ≥ 98%
                for _sl in _SL_ALPHA:
                    for _a in _ALPHA_SWEEP_S:
                        try:
                            _, _r3 = bouw_model_kosten(_ov, _a, _KAPPA_BPA_DEF, _KAPPA_C_DEF, _sl)
                            _g3[_sl].append({'alpha': _a, 'marge': _r3['bpa_margin']})
                        except Exception:
                            _g3[_sl].append({'alpha': _a, 'marge': None})

            st.session_state.sens_g1 = _g1
            st.session_state.sens_g2 = _g2
            st.session_state.sens_g3 = _g3

        if 'sens_g1' in st.session_state:
            import matplotlib.pyplot as _plt
            import matplotlib.ticker as _mt

            _COLORS5 = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F']
            _COLORS4 = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
            _fmt_eur = _mt.FuncFormatter(lambda v, _: f'€{v:,.0f}')
            _fmt_sl  = _mt.FuncFormatter(lambda v, _: f'{v:.2f}%')

            # ── Grafiek 1: service level vs marge per N ────────────────────
            _fig1, _ax1 = _plt.subplots(figsize=(10, 5))
            for _n, _col in zip(_N_VALS, _COLORS5):
                _pts = [(r['sl']*100, r['marge'])
                        for r in st.session_state.sens_g1[_n] if r['marge'] is not None]
                if _pts:
                    _ax1.plot([p[0] for p in _pts], [p[1] for p in _pts],
                              marker='o', linewidth=2, color=_col, label=f'N = {_n}')
            _ax1.axhline(0, color='grey', linewidth=0.8)
            _ax1.set_xlabel('Service level (%)', fontsize=11)
            _ax1.set_ylabel('Annual margin (€)', fontsize=11)
            _ax1.set_title(
                f'Margin vs. service level  '
                f'(α = {_ALPHA_DEF:.0%}, κ_BPA = {_KAPPA_BPA_DEF:.0%})',
                fontsize=12,
            )
            _ax1.yaxis.set_major_formatter(_fmt_eur)
            _ax1.xaxis.set_major_formatter(_fmt_sl)
            _ax1.set_xticks([sl*100 for sl in _SL_SWEEP_S])
            _ax1.legend(fontsize=9)
            _ax1.grid(True, alpha=0.3)
            _plt.setp(_ax1.get_xticklabels(), rotation=25, ha='right')
            _fig1.tight_layout()
            st.pyplot(_fig1)
            _plt.close(_fig1)

            # ── Grafiek 2: service level vs basisvoorraad ──────────────────
            _fig2, _ax2 = _plt.subplots(figsize=(10, 4))
            _pts2 = [(r['sl']*100, r['base'])
                     for r in st.session_state.sens_g2 if r['base'] is not None]
            if _pts2:
                _ax2.plot([p[0] for p in _pts2], [p[1] for p in _pts2],
                          marker='s', linewidth=2, color='#FF9800', label='Total S*')
                for _xv, _yv in _pts2:
                    _ax2.annotate(str(int(_yv)), (_xv, _yv),
                                  textcoords='offset points', xytext=(0, 7),
                                  ha='center', fontsize=9)
            _ax2.set_xlabel('Service level (%)', fontsize=11)
            _ax2.set_ylabel('Total base stock (units)', fontsize=11)
            _ax2.set_title(
                f'Base stock vs. service level  '
                f'(α = {_ALPHA_DEF:.0%}, N = standard)',
                fontsize=12,
            )
            _ax2.xaxis.set_major_formatter(_fmt_sl)
            _ax2.set_xticks([sl*100 for sl in _SL_SWEEP_S])
            _ax2.yaxis.set_major_locator(_mt.MaxNLocator(integer=True))
            _ax2.legend(fontsize=9)
            _ax2.grid(True, alpha=0.3)
            _plt.setp(_ax2.get_xticklabels(), rotation=25, ha='right')
            _fig2.tight_layout()
            st.pyplot(_fig2)
            _plt.close(_fig2)

            # ── Grafiek 3: alpha vs marge per service level ────────────────
            _fig3, _ax3 = _plt.subplots(figsize=(10, 5))
            for _sl3, _col3 in zip(_SL_ALPHA, _COLORS4):
                _pts3 = [(r['alpha']*100, r['marge'])
                         for r in st.session_state.sens_g3[_sl3] if r['marge'] is not None]
                if _pts3:
                    _ax3.plot([p[0] for p in _pts3], [p[1] for p in _pts3],
                              marker='o', linewidth=2, color=_col3, label=f'SL = {_sl3:.1%}')
            _ax3.axhline(0, color='grey', linewidth=0.8)
            _ax3.set_xlabel('Subscription rate α (%)', fontsize=11)
            _ax3.set_ylabel('Annual margin (€)', fontsize=11)
            _ax3.set_title(
                f'Margin vs. subscription rate  '
                f'(κ_BPA = {_KAPPA_BPA_DEF:.0%}, N = standard)',
                fontsize=12,
            )
            _ax3.yaxis.set_major_formatter(_fmt_eur)
            _ax3.xaxis.set_major_formatter(_mt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
            _ax3.set_xticks([a*100 for a in _ALPHA_SWEEP_S])
            _ax3.legend(fontsize=9)
            _ax3.grid(True, alpha=0.3)
            _plt.setp(_ax3.get_xticklabels(), rotation=25, ha='right')
            _fig3.tight_layout()
            st.pyplot(_fig3)
            _plt.close(_fig3)

        # ── Haalbaarheid BPA per (N, SL) – heatmap ────────────────────────────────
        st.divider()
        st.subheader("Haalbaarheid BPA per (Z, serviceniveau)")
        st.caption(
            "Groen = BPA is haalbaar (marge ≥ 0), rood = niet haalbaar. "
            "α wordt overgenomen uit tabblad 💰 Kostenanalyse; κ_BPA en κ_c idem."
        )

        _N_NSL_VALS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

        if st.button("📊 Bereken haalbaarheid (N × SL)"):
            _kp_nsl = st.session_state.get('kosten_params', {})
            _a_nsl  = _kp_nsl.get('alpha',     0.15)
            _kb_nsl = _kp_nsl.get('kappa_bpa', 0.20)
            _kc_nsl = _kp_nsl.get('kappa_c',   0.25)

            _nsl_grid = {}
            with st.spinner("Berekenen haalbaarheid (N × SL)…"):
                for _n_nsl in _N_NSL_VALS:
                    _nsl_grid[_n_nsl] = {}
                    for _sl_nsl in SERVICE_LEVELS:
                        try:
                            _, _r_nsl = bouw_model_kosten(
                                st.session_state.overzicht_df,
                                _a_nsl, _kb_nsl, _kc_nsl, _sl_nsl,
                                n_klanten_override=_n_nsl,
                            )
                            _nsl_grid[_n_nsl][_sl_nsl] = {
                                'feasible': _r_nsl['feasible'],
                                'margin':   _r_nsl['bpa_margin'],
                            }
                        except Exception:
                            _nsl_grid[_n_nsl][_sl_nsl] = {'feasible': False, 'margin': None}

            st.session_state.sens_nsl_grid  = _nsl_grid
            st.session_state.sens_nsl_alpha = _a_nsl
            st.session_state.sens_nsl_kb    = _kb_nsl

        if 'sens_nsl_grid' in st.session_state:
            import matplotlib.pyplot as _plt_nsl
            import matplotlib.colors as _mcolors_nsl
            import numpy as _np_nsl

            _grid   = st.session_state.sens_nsl_grid
            _a_lbl  = st.session_state.sens_nsl_alpha
            _kb_lbl = st.session_state.sens_nsl_kb
            _n_std_nsl = representatieve_z()

            _rows_nsl = SERVICE_LEVELS       # y-as
            _cols_nsl = _N_NSL_VALS          # x-as

            # Bouw matrices: haalbaarheid (0/1) en genormaliseerde marge
            _feas_mat = _np_nsl.zeros((len(_rows_nsl), len(_cols_nsl)))
            _marg_mat = _np_nsl.full((len(_rows_nsl), len(_cols_nsl)), float('nan'))

            for _ci, _n_v in enumerate(_cols_nsl):
                for _ri, _sl_v in enumerate(_rows_nsl):
                    _cell = _grid.get(_n_v, {}).get(_sl_v, {})
                    _feas_mat[_ri, _ci] = 1.0 if _cell.get('feasible') else 0.0
                    if _cell.get('margin') is not None:
                        _marg_mat[_ri, _ci] = _cell['margin']

            # Kleurschaal: rood → geel → groen via marge-waarden
            _valid = _marg_mat[~_np_nsl.isnan(_marg_mat)]
            if len(_valid) > 0:
                _abs_max = max(abs(_valid.min()), abs(_valid.max()), 1)
            else:
                _abs_max = 1
            _norm_nsl = _mcolors_nsl.TwoSlopeNorm(
                vmin=-_abs_max, vcenter=0, vmax=_abs_max
            )

            _fig_nsl, _ax_nsl = _plt_nsl.subplots(figsize=(13, 5))
            _im_nsl = _ax_nsl.imshow(
                _marg_mat, aspect='auto',
                cmap='RdYlGn', norm=_norm_nsl,
                interpolation='nearest',
            )
            _plt_nsl.colorbar(_im_nsl, ax=_ax_nsl, label='BPA margin (€)', fraction=0.03, pad=0.02)

            # Annotaties per cel
            for _ri, _sl_v in enumerate(_rows_nsl):
                for _ci, _n_v in enumerate(_cols_nsl):
                    _cell = _grid.get(_n_v, {}).get(_sl_v, {})
                    _feas = _cell.get('feasible', False)
                    _mg   = _cell.get('margin')
                    _sym  = '✓' if _feas else '✗'
                    _tc   = '#1a5c1a' if _feas else '#7a0000'
                    _ax_nsl.text(_ci, _ri, _sym,
                                 ha='center', va='center' if _mg is None else 'bottom',
                                 fontsize=13, color=_tc, fontweight='bold')
                    if _mg is not None:
                        _ax_nsl.text(_ci, _ri + 0.28, f'€{_mg:,.0f}',
                                     ha='center', va='center', fontsize=6.5, color=_tc)

            # Assen
            _ax_nsl.set_xticks(range(len(_cols_nsl)))
            _ax_nsl.set_xticklabels([str(n) for n in _cols_nsl], fontsize=9)
            _ax_nsl.set_yticks(range(len(_rows_nsl)))
            _ax_nsl.set_yticklabels([f'{sl:.1%}' for sl in _rows_nsl], fontsize=9)
            _ax_nsl.set_xlabel('Number of subscriptions (Z)', fontsize=11)
            _ax_nsl.set_ylabel('Service level', fontsize=11)
            _ax_nsl.set_title(
                f'BPA feasibility per (Z, service level)  '
                f'(α = {_a_lbl:.0%}, κ_BPA = {_kb_lbl:.0%})',
                fontsize=12,
            )

            # Markeer huidige N
            try:
                _ni_std = min(range(len(_cols_nsl)),
                              key=lambda k: abs(_cols_nsl[k] - _n_std_nsl))
                _ax_nsl.axvline(_ni_std, color='black', linewidth=2.0, linestyle=':')
                _ax_nsl.text(_ni_std + 0.15, -0.7, f'Z={_n_std_nsl}',
                             fontsize=8, color='black')
            except Exception:
                pass

            _fig_nsl.tight_layout()
            st.pyplot(_fig_nsl)
            _plt_nsl.close(_fig_nsl)

        # ── Investering vs. N ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Investering vs. aantal subscripties")
        st.caption(
            "Totale voorraadwaarde (Σ S\u002a × inkoopprijs) als functie van het aantal "
            "subscripties per service level. Toont hoeveel kapitaal BPA in voorraad "
            "moet investeren naarmate het klantenbestand groeit. De x-as toont het "
            "TOTALE aantal subscripties over alle componenten en start bij de som van "
            "de huidige geconfigureerde Z_i-waarden; alle componenten schalen van daaruit "
            "proportioneel mee omhoog."
        )

        # Baseline per component = het geconfigureerde n_klanten (Z_i).
        # De x-as toont het TOTAAL aantal subscripties over alle componenten
        # (= som van de baselines bij factor 1.0); alle componenten schalen
        # proportioneel mee.
        _sim_base_inv = {}
        # Groeifactoren: 20 punten van 1.0x (huidig totaal) tot 2.9x in stappen van 0.1.
        _INV_FACTORS = [round(1.0 + 0.1 * _k, 1) for _k in range(20)]
        _COLORS_INV = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

        if st.button("📊 Bereken investering vs. totaal subs"):
            _ov_inv = st.session_state.overzicht_df.reset_index()
            # Verzamel per component: lambda per subscriptie, baseline-subs, LT, IP, VP, code
            _comp_inv = []
            for _, _ri in _ov_inv.iterrows():
                _ni = float(_ri.get('n_klanten', 0) or 0)
                _li = float(_ri.get('lambda_jr', 0) or 0)
                _lt = float(_ri.get('LT_dagen', 0) or 0)
                _ip = float(_ri.get('IP', 0) or 0)
                _vp = float(_ri.get('VP', 0) or 0)
                if _ni > 0 and _li > 0 and _lt > 0:
                    _code = str(_ri.get('Code', ''))
                    _comp_inv.append({
                        'code':        _code,
                        'descr':       str(_ri.get('Descr', '')),
                        'lam_per_sub': _li / _ni,
                        'n_base':      float(_sim_base_inv.get(_code, _ni)),
                        'lt_jr':       _lt / 365,
                        'ip':          _ip,
                        'vp':          _vp,
                    })

            # Totaal subs bij factor 1.0 = som van de (verwachte) baselines.
            _T0_inv = sum(_c['n_base'] for _c in _comp_inv)

            _inv_results = {sl: [] for sl in SERVICE_LEVELS}
            # Per-component resultaten voor top-5 grafiek (alle SL's)
            _sl_top = st.session_state.get('kosten_params', {}).get('service_level', 0.990)
            _inv_per_comp = {sl: {c['code']: [] for c in _comp_inv} for sl in SERVICE_LEVELS}

            with st.spinner("Berekenen investering vs. totaal subs…"):
                for _f_inv in _INV_FACTORS:
                    _tot_subs = int(round(_T0_inv * _f_inv))   # x-waarde: totaal subscripties
                    for _sl_inv in SERVICE_LEVELS:
                        _totaal = sum(
                            BPAOptimizationModel.inverse_service_level(
                                _sl_inv, _c['lam_per_sub'] * _c['n_base'] * _f_inv, _c['lt_jr']
                            ) * _c['ip']
                            for _c in _comp_inv
                        )
                        _inv_results[_sl_inv].append({'n': _tot_subs, 'inv': _totaal})
                    # Per-component per SL (voor top-5/top-10 grafiek)
                    for _c in _comp_inv:
                        for _sl_c in SERVICE_LEVELS:
                            _s = BPAOptimizationModel.inverse_service_level(
                                _sl_c, _c['lam_per_sub'] * _c['n_base'] * _f_inv, _c['lt_jr']
                            )
                            _inv_per_comp[_sl_c][_c['code']].append({'n': _tot_subs, 'inv': _s * _c['ip']})

            # Top 5 / Top 10 duurste componenten op VP
            _top5_codes  = sorted(_comp_inv, key=lambda c: c['vp'], reverse=True)[:5]
            _top10_codes = sorted(_comp_inv, key=lambda c: c['vp'], reverse=True)[:10]

            st.session_state.sens_inv        = _inv_results
            st.session_state.sens_inv_comp   = _inv_per_comp
            st.session_state.sens_inv_top5   = _top5_codes
            st.session_state.sens_inv_top10  = _top10_codes
            st.session_state.sens_inv_sl_top = _sl_top
            st.session_state.sens_inv_t0     = int(round(_T0_inv))

        if 'sens_inv' in st.session_state:
            import matplotlib.pyplot as _plt_inv
            import matplotlib.ticker as _mt_inv

            _inv_d    = st.session_state.sens_inv
            _tot0_inv = int(st.session_state.get('sens_inv_t0', 0))
            _x_ticks_inv = [r['n'] for r in _inv_d[SERVICE_LEVELS[0]]]
            _fmt_inv  = _mt_inv.FuncFormatter(lambda v, _: f'€{v:,.0f}')

            _fig_inv, _ax_inv = _plt_inv.subplots(figsize=(11, 5))
            for _sl_inv, _col_inv in zip(SERVICE_LEVELS, _COLORS_INV):
                _pts_inv = [(r['n'], r['inv']) for r in _inv_d[_sl_inv] if r['inv'] is not None]
                if _pts_inv:
                    _xi, _yi = zip(*_pts_inv)
                    _ax_inv.plot(_xi, _yi, marker='o', linewidth=2,
                                 color=_col_inv, label=f'SL = {_sl_inv:.1%}')

            _ax_inv.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                            label=f'Total subs (sim) = {_tot0_inv}')
            _ax_inv.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
            _ax_inv.set_ylabel('Total inventory value (€)', fontsize=11)
            _ax_inv.set_title(
                'Required investment in base stock vs. total number of subscriptions',
                fontsize=12,
            )
            _ax_inv.yaxis.set_major_formatter(_fmt_inv)
            _ax_inv.set_xticks(_x_ticks_inv)
            _plt_inv.setp(_ax_inv.get_xticklabels(), rotation=30, ha='right')
            _ax_inv.legend(fontsize=9)
            _ax_inv.grid(True, alpha=0.3)
            _fig_inv.tight_layout()
            st.pyplot(_fig_inv)
            _plt_inv.close(_fig_inv)

            # Tabel: investering per totaal aantal subs en SL
            _inv_tbl_rows = []
            for _n_v in _x_ticks_inv:
                _row_t = {'Totaal subs': _n_v}
                for _sl_v in SERVICE_LEVELS:
                    _pts = [r for r in _inv_d[_sl_v] if r['n'] == _n_v]
                    _row_t[f'SL {_sl_v:.1%}'] = f"€{_pts[0]['inv']:,.0f}" if _pts else '—'
                _inv_tbl_rows.append(_row_t)
            st.dataframe(pd.DataFrame(_inv_tbl_rows).set_index('Totaal subs'), use_container_width=False)

            # ── Top-5 duurste componenten per VP ──────────────────────────
            if 'sens_inv_top5' in st.session_state:
                _top5    = st.session_state.sens_inv_top5
                _comp_d  = st.session_state.sens_inv_comp
                _sl_lbl  = st.session_state.sens_inv_sl_top

                _COLORS_TOP5 = ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C', '#1976D2']
                st.subheader("Top 5 duurste componenten (VP) — investering vs. totaal subs (per service level)")
                st.caption(
                    "Gesommeerde investeringswaarde (S\u002a × IP) van de top 5 duurste componenten "
                    "(op verkoopprijs) als functie van het totaal aantal subscripties, per service level."
                )

                # Haal x-waarden op uit de data
                _comp_d_sl0 = _comp_d.get(SERVICE_LEVELS[0], {})
                _x5_vals = [p['n'] for p in _comp_d_sl0.get(_top5[0]['code'], [])] if _top5 else []

                if _x5_vals:
                    _fig_t5, _ax_t5 = _plt_inv.subplots(figsize=(11, 5))

                    for _sl_t5, _col_t5, _ls_t5 in zip(
                            SERVICE_LEVELS, _COLORS_INV, ['-', '--', '-.', ':']):
                        _cd_sl = _comp_d.get(_sl_t5, {})
                        _tot_sl = [
                            sum(
                                next((p['inv'] for p in _cd_sl.get(_c5['code'], [])
                                      if p['n'] == _nv), 0)
                                for _c5 in _top5
                            )
                            for _nv in _x5_vals
                        ]
                        _ax_t5.plot(_x5_vals, _tot_sl, color=_col_t5, marker='o',
                                    linewidth=2.0, linestyle=_ls_t5,
                                    label=f'SL {_sl_t5:.1%}')

                    _ax_t5.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                   label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t5.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t5.set_ylabel('Summed investment value top 5 (€)', fontsize=11)
                    _ax_t5.set_title(
                        'Top 5 most expensive components (VP): summed investment vs. total subs per SL',
                        fontsize=12,
                    )
                    _ax_t5.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t5.set_xticks(_x5_vals)
                    _plt_inv.setp(_ax_t5.get_xticklabels(), rotation=30, ha='right')
                    _ax_t5.legend(fontsize=9, loc='upper left')
                    _ax_t5.grid(True, alpha=0.3)
                    _fig_t5.tight_layout()
                    st.pyplot(_fig_t5)
                    _plt_inv.close(_fig_t5)

            # ── Top-10 duurste componenten — gesommeerde lijnen per SL ──────
            if 'sens_inv_top10' in st.session_state:
                import matplotlib as _mpl_inv
                _top10   = st.session_state.sens_inv_top10
                _comp_d  = st.session_state.sens_inv_comp

                _comp_d_sl0_t10 = _comp_d.get(SERVICE_LEVELS[0], {})
                _x10_sum_vals = [p['n'] for p in _comp_d_sl0_t10.get(_top10[0]['code'], [])] if _top10 else []

                st.subheader("Top 10 duurste componenten (VP) — gesommeerde investering vs. totaal subs (per service level)")
                st.caption(
                    "Gesommeerde investeringswaarde (S\u002a × IP) van de top 10 duurste componenten "
                    "(op verkoopprijs) als functie van het totaal aantal subscripties, per service level."
                )

                if _x10_sum_vals:
                    _fig_t10s, _ax_t10s = _plt_inv.subplots(figsize=(11, 5))
                    for _sl_t10s, _col_t10s, _ls_t10s in zip(
                            SERVICE_LEVELS, _COLORS_INV, ['-', '--', '-.', ':']):
                        _cd10s = _comp_d.get(_sl_t10s, {})
                        _tot10s = [
                            sum(
                                next((p['inv'] for p in _cd10s.get(_c10['code'], [])
                                      if p['n'] == _nv), 0)
                                for _c10 in _top10
                            )
                            for _nv in _x10_sum_vals
                        ]
                        _ax_t10s.plot(_x10_sum_vals, _tot10s, color=_col_t10s, marker='o',
                                      linewidth=2.0, linestyle=_ls_t10s,
                                      label=f'SL {_sl_t10s:.1%}')
                    _ax_t10s.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                     label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t10s.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t10s.set_ylabel('Summed investment value top 10 (€)', fontsize=11)
                    _ax_t10s.set_title(
                        'Top 10 most expensive components (VP): summed investment vs. total subs per SL',
                        fontsize=12,
                    )
                    _ax_t10s.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t10s.set_xticks(_x10_sum_vals)
                    _plt_inv.setp(_ax_t10s.get_xticklabels(), rotation=30, ha='right')
                    _ax_t10s.legend(fontsize=9, loc='upper left')
                    _ax_t10s.grid(True, alpha=0.3)
                    _fig_t10s.tight_layout()
                    st.pyplot(_fig_t10s)
                    _plt_inv.close(_fig_t10s)

                # ── Top-10 duurste componenten — individuele lijnen ──────────
                _sl_opts_t10 = [f'SL {s:.1%}' for s in SERVICE_LEVELS]
                _sl_sel_t10  = st.selectbox(
                    'Service level voor top-10 grafiek',
                    _sl_opts_t10,
                    index=1,
                    key='top10_sl_select',
                )
                _sl_val_t10 = SERVICE_LEVELS[_sl_opts_t10.index(_sl_sel_t10)]

                st.subheader("Top 10 duurste componenten (VP) — investering per component vs. totaal subs")
                st.caption(
                    "Investeringswaarde (S\u002a \u00d7 IP) per component als functie van het totaal aantal subscripties "
                    "voor het geselecteerde service level."
                )

                _cd10_sl = _comp_d.get(_sl_val_t10, {})
                _x10_vals = [p['n'] for p in _cd10_sl.get(_top10[0]['code'], [])] if _top10 else []

                if _x10_vals:
                    _cmap10  = _mpl_inv.colormaps['tab10']
                    _fig_t10, _ax_t10 = _plt_inv.subplots(figsize=(12, 5))

                    for _ci, _c10 in enumerate(_top10):
                        _pts10 = [p['inv'] for p in _cd10_sl.get(_c10['code'], [])]
                        if _pts10:
                            _lbl10 = f"{_c10['code']} – {_c10.get('descr', '')[:25]}"
                            _ax_t10.plot(
                                _x10_vals, _pts10,
                                color=_cmap10(_ci / 10),
                                marker='o', linewidth=1.8, markersize=5,
                                label=_lbl10,
                            )

                    _ax_t10.axvline(_tot0_inv, color='black', linewidth=1.0, linestyle=':',
                                    label=f'Total subs (sim) = {_tot0_inv}')
                    _ax_t10.set_xlabel('Total number of subscriptions (all components)', fontsize=11)
                    _ax_t10.set_ylabel('Investment value per component (€)', fontsize=11)
                    _ax_t10.set_title(
                        f'Top 10 most expensive components — investment vs. total subs  ({_sl_sel_t10})',
                        fontsize=12,
                    )
                    _ax_t10.yaxis.set_major_formatter(_fmt_inv)
                    _ax_t10.set_xticks(_x10_vals)
                    _plt_inv.setp(_ax_t10.get_xticklabels(), rotation=30, ha='right')
                    _ax_t10.legend(fontsize=8, loc='upper left', ncol=2)
                    _ax_t10.grid(True, alpha=0.3)
                    _fig_t10.tight_layout()
                    st.pyplot(_fig_t10)
                    _plt_inv.close(_fig_t10)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 – KOSTENANALYSE
# ─────────────────────────────────────────────────────────────────────────────

with tab_kosten:
    st.subheader("Kostenanalyse BPA")
    st.caption(
        "Berekent BPA-kosten, omzet, marge en α-interval per component "
        "op basis van het huidige overzicht en de gekozen draaiknoppen."
    )

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.warning("Laad eerst het overzicht via het tabblad 📊 Overzicht.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            k_alpha = st.number_input(
                "α (abonnementstarief, %)",
                min_value=1.0, max_value=50.0, value=15.0, step=1.0, format="%.0f",
                help="Abonnementsprijs als percentage van verkoopprijs",
            ) / 100
        with col_b:
            k_kappa_bpa = st.number_input(
                "κ_BPA (%)",
                min_value=1.0, max_value=100.0, value=20.0, step=1.0, format="%.0f",
                help="κ_BPA = financiering + opslag + obsolescence (BPA)",
            ) / 100
        with col_c:
            k_kappa_c = st.number_input(
                "κ_c (%)",
                min_value=1.0, max_value=100.0, value=25.0, step=1.0, format="%.0f",
                help="κ_c = financiering + opslag + obsolescence (klant)",
            ) / 100
        with col_d:
            k_sl = st.selectbox(
                "Service level",
                options=SERVICE_LEVELS,
                index=SERVICE_LEVELS.index(0.990) if 0.990 in SERVICE_LEVELS else 0,
                format_func=lambda v: f"{v:.1%}",
            )

        if st.button("💰 Bereken kosten"):
            with st.spinner("Kostenmodel berekenen…"):
                try:
                    _m, _r = bouw_model_kosten(
                        st.session_state.overzicht_df,
                        alpha=k_alpha,
                        kappa_bpa=k_kappa_bpa,
                        kappa_c=k_kappa_c,
                        service_level=k_sl,
                    )
                    st.session_state.kosten_result = (_m, _r)
                    st.session_state.kosten_params = {
                        'alpha': k_alpha, 'kappa_bpa': k_kappa_bpa,
                        'kappa_c': k_kappa_c, 'service_level': k_sl,
                    }
                except Exception as _e:
                    st.error(f"Fout bij berekening: {_e}")

        if "kosten_result" in st.session_state:
            _m, _r = st.session_state.kosten_result
            _iv = _r['alpha_intervals']
            _p  = st.session_state.kosten_params

            _det = _m.calculate_detailed_bpa_costs()
            _per = _iv['per_component']
            _profitable_codes = [
                _code for _code in _m.sets['spare_parts']
                if _r['revenue_by_part'].get(_code, 0) - _det[_code]['total'] > 0
            ]
            _kpi_revenue = sum(_r['revenue_by_part'].get(_code, 0)
                               for _code in _profitable_codes)
            _kpi_costs = sum(_det[_code]['total'] for _code in _profitable_codes)
            _kpi_margin = _kpi_revenue - _kpi_costs
            _kpi_feasible = bool(_profitable_codes) and all(
                _per.get(_code, {}).get('alpha_U') is not None
                and _p['alpha'] <= _per[_code]['alpha_U']
                for _code in _profitable_codes
            )

            # ── Samenvatting ───────────────────────────────────────────────
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Haalbaar", "✓ JA" if _kpi_feasible else "✗ NEE",
                       help="Berekend over uitsluitend winstgevende onderdelen.")
            _c2.metric("Totale omzet", f"€ {_kpi_revenue:,.0f}")
            _c3.metric("BPA kosten", f"€ {_kpi_costs:,.0f}")
            _c4.metric("Marge", f"€ {_kpi_margin:+,.0f}")
            st.caption(
                f"KPI's bevatten alleen de **{len(_profitable_codes)} winstgevende** "
                f"onderdelen van {len(_m.sets['spare_parts'])} totaal."
            )

            _al = _iv['universal_alpha_L']
            _au = _iv['universal_alpha_U']
            if _al is not None:
                st.info(
                    f"Universeel α-interval: **[{_al:.4%} – {_au:.4%}]**  "
                    f"{'✓ Haalbaar' if _iv['universal_feasible'] else '✗ Niet haalbaar'}"
                )

            # ── Per-component kosten tabel ─────────────────────────────────
            st.subheader("Kosten per component")
            _bsl = _m.calculate_base_stock_levels()
            _lt  = _m.parameters['lead_time']

            _rows = []
            for _code in _m.sets['spare_parts']:
                _d = _det[_code]
                _pc = _per.get(_code, {})
                _al = _pc.get('alpha_L')
                _au = _pc.get('alpha_U')
                _ok = (
                    _al is not None and _au is not None
                    and _al <= _p['alpha'] <= _au
                )
                _rows.append({
                    'Code':       _code,
                    'S*':         _bsl.get(_code, 0),
                    'Λ_BPA':      round(_d['demand'], 4),
                    'μ=Λ·L':      round(_d['demand'] * _lt.get(_code, 0), 4),
                    'C_BPA (€)':  round(_d['total'], 2),
                    'Omzet (€)':  round(_r['revenue_by_part'].get(_code, 0), 2),
                    'Marge (€)':  round(_r['revenue_by_part'].get(_code, 0) - _d['total'], 2),
                    'α_L,i':      f"{_al:.3%}" if _al is not None else '—',
                    'α_U,i':      f"{_au:.3%}" if _au is not None else '—',
                    'OK':         '✓' if _ok else '✗',
                })
            _tbl = pd.DataFrame(_rows).set_index('Code')

            st.dataframe(
                _tbl.style.format({
                    'S*':        '{:.0f}',
                    'Λ_BPA':    '{:.4f}',
                    'μ=Λ·L':    '{:.4f}',
                    'C_BPA (€)': '€ {:,.2f}',
                    'Omzet (€)': '€ {:,.2f}',
                    'Marge (€)': '€ {:+,.2f}',
                }),
                use_container_width=True,
                height=420,
            )
            st.write(
                f"**Totaal winstgevende onderdelen:** "
                f"S\\* = {int(_tbl.loc[_profitable_codes, 'S*'].sum())}  |  "
                f"C\\_BPA = € {_kpi_costs:,.2f}  |  "
                f"Omzet = € {_kpi_revenue:,.2f}  |  "
                f"Marge = € {_kpi_margin:+,.2f}"
            )

            # ── Klantbesparingen ───────────────────────────────────────────
            with st.expander("Klantbesparingen"):
                _klant_rows = [
                    {
                        'Klant':              _cust,
                        'Eigen kosten (€)':   b['self_stocking_cost'],
                        'BPA abonnement (€)': b['bpa_service_cost'],
                        'Besparing (€)':      b['savings'],
                        'Voordeel':           '✓' if b['benefits'] else '✗',
                    }
                    for _cust, b in _r['customer_benefits'].items()
                ]
                st.dataframe(
                    pd.DataFrame(_klant_rows).set_index('Klant').style.format({
                        'Eigen kosten (€)':   '€ {:,.2f}',
                        'BPA abonnement (€)': '€ {:,.2f}',
                        'Besparing (€)':      '€ {:+,.2f}',
                    }),
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 – BUDGET-SCENARIO
# ─────────────────────────────────────────────────────────────────────────────

with tab_budget:
    st.subheader("Budget-scenario — greedy selectie")
    st.caption(
        "Selecteer binnen een maximaal investeringsbudget de winstgevende onderdelen "
        "met de hoogste waarde per geïnvesteerde euro. Investering = S* × IP en "
        "jaarwinst = Z × α × VP − κ_BPA × IP × S*."
    )

    _ov_budget = st.session_state.get("overzicht_df")
    if _ov_budget is None or _ov_budget.empty:
        st.warning("Laad eerst het overzicht via het tabblad 📊 Overzicht.")
    else:
        _ov_budget = _ov_budget.copy()
        _sl_cols_budget = [c for c in _ov_budget.columns if c.startswith("s@")]

        if not _sl_cols_budget:
            st.error("Geen service-levelkolommen gevonden in het overzicht.")
        else:
            _kp_budget = st.session_state.get("kosten_params", {})
            _bc1, _bc2, _bc3 = st.columns(3)
            with _bc1:
                _sl_budget = st.selectbox(
                    "Service level voor S*",
                    options=_sl_cols_budget,
                    index=len(_sl_cols_budget) // 2,
                    key="budget_sl",
                )

            _full_investment = float(
                (_ov_budget[_sl_budget] * _ov_budget["IP"]).sum()
            )
            with _bc2:
                _max_budget = st.number_input(
                    "Maximaal budget (€)",
                    min_value=0.0,
                    max_value=max(_full_investment * 1.2, 1_000_000.0),
                    value=float(round(_full_investment * 0.5, 0)),
                    step=1000.0,
                    key="budget_max",
                )
            with _bc3:
                _criterion_budget = st.selectbox(
                    "Waardecriterium",
                    options=[
                        "Winst / investering (ROI)",
                        "Classificatie-score",
                        "λ × LT × VP (uitval-impact)",
                        "VP (verkoopprijs)",
                    ],
                    key="budget_criterion",
                )

            _bp1, _bp2 = st.columns(2)
            with _bp1:
                _alpha_budget = st.number_input(
                    "α (abonnementstarief, %)",
                    min_value=0.1,
                    max_value=50.0,
                    value=float(_kp_budget.get("alpha", 0.15)) * 100,
                    step=0.5,
                    format="%.1f",
                    key="budget_alpha",
                ) / 100
            with _bp2:
                _kappa_budget = st.number_input(
                    "κ_BPA (carrying rate, %)",
                    min_value=0.1,
                    max_value=100.0,
                    value=float(_kp_budget.get("kappa_bpa", 0.20)) * 100,
                    step=0.5,
                    format="%.1f",
                    key="budget_kappa",
                ) / 100

            _df_budget = _ov_budget.copy()
            _df_budget["S_star"] = pd.to_numeric(
                _df_budget[_sl_budget], errors="coerce"
            ).fillna(0.0)
            _df_budget["Investering"] = _df_budget["S_star"] * _df_budget["IP"]
            _df_budget["Omzet_jr"] = (
                _df_budget["n_klanten"] * _alpha_budget * _df_budget["VP"]
            )
            _df_budget["C_BPA"] = (
                _kappa_budget * _df_budget["IP"] * _df_budget["S_star"]
            )
            _df_budget["Winst_jr"] = (
                _df_budget["Omzet_jr"] - _df_budget["C_BPA"]
            )

            if _criterion_budget == "Winst / investering (ROI)":
                _df_budget["Waarde"] = _df_budget["Winst_jr"]
            elif _criterion_budget == "Classificatie-score":
                _score_budget = pd.to_numeric(
                    _df_budget.get("Cls_score"), errors="coerce"
                )
                _fallback_budget = (
                    _df_budget["lambda_jr"]
                    * (_df_budget["LT_dagen"] / 365)
                    * _df_budget["VP"]
                )
                _df_budget["Waarde"] = _score_budget.fillna(_fallback_budget)
            elif _criterion_budget == "λ × LT × VP (uitval-impact)":
                _df_budget["Waarde"] = (
                    _df_budget["lambda_jr"]
                    * (_df_budget["LT_dagen"] / 365)
                    * _df_budget["VP"]
                )
            else:
                _df_budget["Waarde"] = _df_budget["VP"]

            _df_budget["Waarde_per_euro"] = np.where(
                _df_budget["Investering"] > 0,
                _df_budget["Waarde"] / _df_budget["Investering"],
                np.inf,
            )
            _df_budget = (
                _df_budget
                .assign(_ratio_sort=lambda d: d["Waarde_per_euro"].round(3))
                .sort_values(
                    ["_ratio_sort", "Investering"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .drop(columns="_ratio_sort")
            )

            _eligible_budget = (
                np.isfinite(_df_budget["Winst_jr"])
                & (_df_budget["Winst_jr"] > 0)
            )
            _selected_budget = np.zeros(len(_df_budget), dtype=bool)
            _remaining_budget = float(_max_budget)
            for _position in np.where(_eligible_budget.to_numpy())[0]:
                _item_investment = float(
                    _df_budget["Investering"].iloc[_position]
                )
                if _item_investment <= _remaining_budget:
                    _selected_budget[_position] = True
                    _remaining_budget -= _item_investment
            _df_budget["In_selectie"] = _selected_budget
            _df_budget["Marge_pct"] = np.where(
                _df_budget["Omzet_jr"] > 0,
                _df_budget["Winst_jr"] / _df_budget["Omzet_jr"] * 100,
                np.nan,
            )
            _df_budget["ROI_jr"] = np.where(
                _df_budget["Investering"] > 0,
                _df_budget["Winst_jr"] / _df_budget["Investering"] * 100,
                np.nan,
            )

            _selected_rows = _df_budget[_df_budget["In_selectie"]]
            _selected_investment = float(_selected_rows["Investering"].sum())
            _selected_revenue = float(_selected_rows["Omzet_jr"].sum())
            _selected_costs = float(_selected_rows["C_BPA"].sum())
            _selected_profit = float(_selected_rows["Winst_jr"].sum())
            _portfolio_roi = (
                _selected_profit / _selected_investment * 100
                if _selected_investment > 0 else 0.0
            )
            _loss_count = int((~_eligible_budget).sum())

            st.caption(
                f"Volledige voorraadwaarde bij {_sl_budget}: "
                f"**€ {_full_investment:,.0f}** · budget: **€ {_max_budget:,.0f}** · "
                f"α = **{_alpha_budget:.1%}** · κ_BPA = **{_kappa_budget:.1%}**"
            )
            if _loss_count:
                st.warning(
                    f"{_loss_count} onderdelen hebben geen positieve jaarwinst en "
                    "worden daarom niet geselecteerd."
                )

            _bm1, _bm2, _bm3, _bm4 = st.columns(4)
            _bm1.metric(
                "Geselecteerd",
                f"{len(_selected_rows)} / {len(_df_budget)}",
            )
            _bm2.metric("Investering", f"€ {_selected_investment:,.0f}")
            _bm3.metric(
                "Budget-benutting",
                f"{_selected_investment / _max_budget:.1%}" if _max_budget > 0 else "—",
            )
            _bm4.metric("ROI / jaar", f"{_portfolio_roi:+.1f}%")

            _bw1, _bw2, _bw3 = st.columns(3)
            _bw1.metric("Omzet / jaar", f"€ {_selected_revenue:,.0f}")
            _bw2.metric("BPA-kosten / jaar", f"€ {_selected_costs:,.0f}")
            _bw3.metric("Winst / jaar", f"€ {_selected_profit:+,.0f}")

            _table_budget = _df_budget.reset_index().rename(columns={
                _df_budget.index.name or "index": "Code",
                "n_klanten": "Z",
                "S_star": "S*",
                "Investering": "Inv. (€)",
                "Omzet_jr": "Omzet/jr (€)",
                "C_BPA": "C_BPA/jr (€)",
                "Winst_jr": "Winst/jr (€)",
                "Marge_pct": "Marge %",
                "ROI_jr": "ROI/jr %",
                "In_selectie": "In selectie",
            })
            _table_budget["In selectie"] = _table_budget["In selectie"].map(
                {True: "✓", False: "✗"}
            )
            _budget_columns = [
                "Code", "Descr", "Z", "S*", "IP", "VP", "Omzet/jr (€)",
                "C_BPA/jr (€)", "Winst/jr (€)", "Marge %", "Inv. (€)",
                "ROI/jr %", "In selectie",
            ]

            def _budget_selection_color(value):
                return (
                    "background-color: #c8e6c9" if value == "✓"
                    else "background-color: #ffcdd2"
                )

            st.subheader("Budgetselectie per onderdeel")
            st.dataframe(
                _table_budget[_budget_columns].style.format({
                    "IP": "€ {:,.2f}",
                    "VP": "€ {:,.2f}",
                    "Omzet/jr (€)": "€ {:,.0f}",
                    "C_BPA/jr (€)": "€ {:,.0f}",
                    "Winst/jr (€)": "€ {:+,.0f}",
                    "Marge %": "{:+.1f}%",
                    "Inv. (€)": "€ {:,.0f}",
                    "ROI/jr %": "{:+.1f}%",
                }, na_rep="—").map(
                    _budget_selection_color, subset=["In selectie"]
                ),
                use_container_width=True,
                height=460,
                hide_index=True,
            )

            with st.expander("Cumulatieve waarde versus investering"):
                import matplotlib.pyplot as _plt_budget

                _eligible_plot = _df_budget[_eligible_budget].copy()
                _cum_investment = _eligible_plot["Investering"].cumsum()
                _cum_value = _eligible_plot["Waarde"].cumsum()
                _fig_budget, _ax_budget = _plt_budget.subplots(figsize=(9, 4.5))
                _ax_budget.plot(_cum_investment, _cum_value, color="#1976D2", linewidth=2)
                _ax_budget.axvline(
                    _max_budget, color="#C62828", linestyle="--",
                    label=f"Budget € {_max_budget:,.0f}",
                )
                _ax_budget.set_xlabel("Cumulative investment (€)")
                _ax_budget.set_ylabel("Cumulative value")
                _ax_budget.set_title("Greedy budget selection")
                _ax_budget.grid(True, alpha=0.3)
                _ax_budget.legend()
                _fig_budget.tight_layout()
                st.pyplot(_fig_budget)
                _plt_budget.close(_fig_budget)

            if st.button("🚫 Pas budgetselectie toe via uitsluitingen"):
                _selected_codes = {
                    str(code) for code in _df_budget.index[_df_budget["In_selectie"]]
                }
                _manual_codes = set(cfg.get("handmatige_componenten", {}))
                _excluded_codes = [
                    str(code) for code in _df_budget.index
                    if str(code) not in _selected_codes and str(code) not in _manual_codes
                ]
                cfg.setdefault("uitgesloten_componenten", [])
                for _code in _excluded_codes:
                    if _code not in cfg["uitgesloten_componenten"]:
                        cfg["uitgesloten_componenten"].append(_code)
                sla_config_op(cfg, BytesIO(_excel_bytes))
                invalidate_caches()
                st.session_state.pop("overzicht_df", None)
                st.toast(
                    f"Budgetselectie toegepast; {len(_excluded_codes)} onderdelen uitgesloten.",
                    icon="✅",
                )
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────────
#  TAB 8 – SUBSCRIPTIEDREMPEL
# ─────────────────────────────────────────────────────────────────────────────────

with tab_drempel:
    st.subheader("Subscriptiedrempel per component")
    st.caption(
        "Per component: hoeveel extra subscripties zijn er nodig voordat S\u002a met 1 stijgt? "
        "Aanname: λ schaalt lineair met Z (λ = Z × λ_huidig / Z_huidig). "
        "Van toepassing op MTBF-gebaseerde componenten."
    )

    if "overzicht_df" not in st.session_state or st.session_state.overzicht_df.empty:
        st.warning("Laad eerst het overzicht via het tabblad 📊 Overzicht.")
    else:
        _df_ov = st.session_state.overzicht_df.copy().reset_index()

        _sl_d = st.selectbox(
            "Service level",
            options=SERVICE_LEVELS,
            index=SERVICE_LEVELS.index(0.985) if 0.985 in SERVICE_LEVELS else 0,
            format_func=lambda v: f"{v:.1%}",
            key="drempel_sl",
        )
        _sl_col = f"s@{_sl_d:.1%}"

        _MAX_N_SEARCH = 100_000
        _drempel_rows = []

        for _, _row in _df_ov.iterrows():
            _code     = _row["Code"]
            _n_orig   = int(_row["n_klanten"])
            _lam_orig = float(_row["lambda_jr"])
            _lt_jr    = float(_row["LT_dagen"]) / 365

            _n = _n_orig
            _lam_pn = _lam_orig / _n_orig if _n_orig > 0 else _lam_orig
            _lam = _lam_orig

            # S* op het huidige geconfigureerde Z_i (Poisson-inverse)
            _s_now = (BPAOptimizationModel.inverse_service_level(_sl_d, _lam, _lt_jr)
                      if _n > 0 and _lam > 0 and _lt_jr > 0 else 0)

            if _n > 0 and _lam_pn > 0 and _lt_jr > 0:
                # Binary search: kleinste N_drempel waarbij S* > _s_now
                _lo, _hi = _n + 1, _n + _MAX_N_SEARCH
                _s_hi = BPAOptimizationModel.inverse_service_level(
                    _sl_d, _lam_pn * _hi, _lt_jr
                )
                if _s_hi <= _s_now:
                    _n_drempel = None
                else:
                    while _lo < _hi:
                        _mid = (_lo + _hi) // 2
                        _s_mid = BPAOptimizationModel.inverse_service_level(
                            _sl_d, _lam_pn * _mid, _lt_jr
                        )
                        if _s_mid > _s_now:
                            _hi = _mid
                        else:
                            _lo = _mid + 1
                    _n_drempel = _lo
            else:
                _n_drempel = None

            _extra = (_n_drempel - _n) if _n_drempel is not None else None
            _drempel_rows.append({
                "Code":          _code,
                "Omschrijving":  str(_row.get("Descr", ""))[:35],
                "Z huidig":      _n,
                "S* huidig":     _s_now,
                "Z voor S*+1":   _n_drempel if _n_drempel is not None else f">{_n + _MAX_N_SEARCH}",
                "Extra Z nodig": _extra,
                "λ/jr":          round(_lam, 4),
                "μ = λ·L":       round(_lam * _lt_jr, 4),
            })

        _tbl_d = pd.DataFrame(_drempel_rows).set_index("Code")
        _tbl_d_sorted = _tbl_d.sort_values("Extra Z nodig", na_position="last")
        # Styler.apply werkt niet met een niet-unieke index (dubbele 'Code').
        # Reset naar een unieke RangeIndex en verberg die in de weergave.
        if not _tbl_d_sorted.index.is_unique:
            _tbl_d_sorted = _tbl_d_sorted.reset_index()

        # Tabel weergeven met kleurcodering op basis van drempel
        def _kleur_drempel(row):
            v = row["Extra Z nodig"]
            if pd.isna(v):
                bg = "#d4edda"   # groen: geen drempel gevonden in zoekbereik
            elif int(v) <= 2:
                bg = "#f8d7da"   # rood: 1-2 extra subscripties
            elif int(v) <= 5:
                bg = "#fff3cd"   # oranje: 3-5 extra subscripties
            else:
                bg = "#d4edda"   # groen: 6+ extra subscripties
            return [f"background-color: {bg}"] * len(row)

        st.dataframe(
            _tbl_d_sorted.style
                .apply(_kleur_drempel, axis=1)
                .format({
                    "Z huidig":      "{:.0f}",
                    "S* huidig":     "{:.0f}",
                    "λ/jr":          "{:.4f}",
                    "μ = λ·L":       "{:.4f}",
                    "Extra Z nodig": lambda v: f"{int(v)}" if pd.notna(v) else "—",
                }),
            use_container_width=True,
            height=500,
        )

        # ── Pooling scenarios: Z_i = 1, 2, 3, M_i ──────────────────────
        _pooling_scenarios = (
            ("Zᵢ = 1", 1),
            ("Zᵢ = 2", 2),
            ("Zᵢ = 3", 3),
            ("Zᵢ = Mᵢ", None),
        )
        _pooling_mi = _cached_maximale_klanten(_excel_bytes)
        _pooling_results = []
        for _scenario_label, _scenario_z in _pooling_scenarios:
            _scenario_subscribers = 0
            _scenario_stock = 0
            for _, _row in _df_ov.iterrows():
                _code = str(_row["Code"])
                _n_orig = int(_row["n_klanten"])
                _lam_orig = float(_row["lambda_jr"])
                _lt_jr = float(_row["LT_dagen"]) / 365
                _mi = max(1, int(round(float(_pooling_mi.get(_code, _n_orig)))))
                _zi = _mi if _scenario_z is None else min(_scenario_z, _mi)
                _lam_per_subscriber = _lam_orig / _n_orig if _n_orig > 0 else 0.0
                _stock = BPAOptimizationModel.inverse_service_level(
                    _sl_d,
                    _lam_per_subscriber * _zi,
                    _lt_jr,
                )
                _scenario_subscribers += _zi
                _scenario_stock += _stock

            _pooling_results.append({
                "Scenario": _scenario_label,
                "Subscriptions": _scenario_subscribers,
                "Base stock": _scenario_stock,
                "Stock per subscriber": (
                    _scenario_stock / _scenario_subscribers
                    if _scenario_subscribers else 0.0
                ),
                "Pooling gain": _scenario_subscribers - _scenario_stock,
            })

        _pooling_df = pd.DataFrame(_pooling_results)
        st.markdown("**Pooling scenarios**")
        st.caption(
            "For the intermediate scenarios, each component uses "
            "Zᵢ = min(k, Mᵢ). Stock per subscriber is calculated as "
            "ΣSᵢ*/ΣZᵢ."
        )

        import matplotlib.pyplot as _plt_pool

        _pool_colors = ["#2667A8", "#2A9D6F", "#E9A23B", "#C94C4C"]
        _fig_pool, _axes_pool = _plt_pool.subplots(
            1, 3, figsize=(14, 4.4), dpi=100, layout="constrained"
        )
        _pool_metrics = (
            ("Base stock", "Total base stock ΣSᵢ*", "{:.0f}"),
            ("Stock per subscriber", "Stock per subscriber ΣSᵢ*/ΣZᵢ", "{:.3f}"),
            ("Pooling gain", "Pooling gain Σ(Zᵢ − Sᵢ*)", "{:.0f}"),
        )
        for _ax_pool, (_metric, _title, _value_format) in zip(
                _axes_pool, _pool_metrics):
            _bars_pool = _ax_pool.bar(
                _pooling_df["Scenario"],
                _pooling_df[_metric],
                color=_pool_colors,
                width=0.68,
            )
            _ax_pool.bar_label(
                _bars_pool,
                labels=[_value_format.format(_v) for _v in _pooling_df[_metric]],
                padding=3,
                fontsize=9,
            )
            _ax_pool.set_title(_title, fontsize=11)
            _ax_pool.grid(True, axis="y", alpha=0.25)
            _ax_pool.set_axisbelow(True)
            _ax_pool.tick_params(axis="x", labelsize=9)

        _fig_pool.suptitle(
            f"Pooling by adoption scenario (SL = {_sl_d:.1%})",
            fontsize=13,
        )
        st.pyplot(_fig_pool, use_container_width=True)
        _plt_pool.close(_fig_pool)

        # ── Bar chart: Extra N nodig per component ─────────────────────────
        _plot_d = _tbl_d_sorted[_tbl_d_sorted["Extra Z nodig"].notna()].copy()
        if not _plot_d.empty:
            import matplotlib.pyplot as _plt_d

            # Cap het aantal balken: bij honderden componenten wordt de grafiek
            # onleesbaar én PIL gooit een DecompressionBombError zodra het
            # gerenderde PNG > ~179 megapixels wordt. Toon de top-N met
            # de hoogste drempel (relevante "rode" gevallen eerst).
            _MAX_BARS = 60
            _n_total  = len(_plot_d)
            if _n_total > _MAX_BARS:
                _plot_d = _plot_d.nsmallest(_MAX_BARS, "Extra Z nodig")
                st.caption(
                    f"📊 Grafiek toont de **{_MAX_BARS}** componenten met de "
                    f"laagste drempel (van {_n_total} totaal). Volledige lijst "
                    f"staat in de tabel hierboven."
                )

            # Begrens figuur-breedte (max 32 inch) en zet expliciet dpi=100
            # om gegarandeerd onder de PIL-pixellimiet te blijven.
            _fig_w = min(max(8, len(_plot_d) * 0.55), 32)
            _fig_d, _ax_d = _plt_d.subplots(figsize=(_fig_w, 5), dpi=100)
            _ax_d.bar(
                range(len(_plot_d)),
                _plot_d["Extra Z nodig"].astype(int),
                color="#1976D2",
            )
            _ax_d.set_xticks(range(len(_plot_d)))
            _ax_d.set_xticklabels(
                _plot_d.index, rotation=45, ha="right", fontsize=9
            )
            _ax_d.set_ylabel("Extra subscriptions for S*+1", fontsize=11)
            _ax_d.set_title(
                f"Subscription threshold per component  (SL = {_sl_d:.1%})",
                fontsize=12,
            )
            _ax_d.grid(True, axis="y", alpha=0.3)
            _fig_d.tight_layout()
            st.pyplot(_fig_d)
            _plt_d.close(_fig_d)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 9 – CLASSIFICATIE
# ─────────────────────────────────────────────────────────────────────────────

with tab_classificatie:
    st.subheader("Componenten selecteren")
    st.caption(
        "Bepaal de rangorde op basis van prijs, aantal klantlocaties en bestelfrequentie. "
        "Componenten die al eerder zijn geselecteerd blijven in het overzicht staan. "
        "Kies hier alleen welke nieuwe componenten je deze periode wilt toevoegen."
    )

    # ── Bron voor de ranglijst: altijd twee CSV-exports uit het ERP ──────
    _cls_bron_modus = "erp_documenten"
    _cls_sheet = "Filtered "  # vast — zelfde sheet als BPA-overzicht (MTBF(years) correct)
    st.caption(
        "Upload hieronder **twee CSV-bestanden** uit het ERP-systeem. Reeds "
        "geselecteerde componenten blijven in het overzicht staan."
    )
    _cls_col_doc, _cls_col_doc1 = st.columns(2)
    with _cls_col_doc:
        _cls_doc_upload = st.file_uploader(
            "📄 Artikeloverzicht — upload CSV",
            type=["csv"], key="cls_doc_upload",
            help="ERP-export met artikelstamdata: prijs, levertijd, ArticleType.",
        )
    with _cls_col_doc1:
        _cls_doc1_upload = st.file_uploader(
            "🧾 Orderregels — upload CSV",
            type=["csv"], key="cls_doc1_upload",
            help="ERP-export met orderregels: gebruikt voor klantlocaties en aantal orders.",
        )

    st.divider()

    # ── Parameters ──
    st.markdown("**Wat bepaalt de prioriteit?**")
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        _w_prijs = st.slider("Belang van verkoopprijs", 0.0, 1.0, 1/3, 0.05, key="cls_w_prijs")
    with _c2:
        _w_loc   = st.slider("Belang van aantal klantlocaties", 0.0, 1.0, 1/3, 0.05, key="cls_w_loc")
    with _c3:
        _w_ord   = st.slider("Belang van weinig bestellingen", 0.0, 1.0, 1/3, 0.05, key="cls_w_ord")

    # ── Selectiemethode: vast op "top X componenten" ────────────────────
    _sel_modus = "top_n"
    _thr = 0.0
    _top_pct = 20.0
    st.markdown("**Voorselectie op basis van de ranglijst**")
    _top_n = st.number_input(
        "Aantal nieuwe componenten om vooraf aan te vinken", 1, 100_000, 100, 1,
        key="cls_top_n",
        help=("De hoogst gerangschikte componenten die nog niet in het overzicht "
              "staan worden alvast aangevinkt."),
    )

    _ord_pow = 2.0

    st.markdown("**Ondergrenzen**")
    st.caption(
        "Artikelen onder deze drempels worden uitgesloten vóór de weging wordt toegepast."
    )
    _mf1, _mf2 = st.columns(2)
    with _mf1:
        _min_prijs = st.number_input(
            "Min. verkoopprijs (€)", 0.0, 100_000.0, 0.0, 10.0,
            key="cls_min_prijs",
            help="Artikelen met verkoopprijs < dit bedrag worden uitgesloten (harde filter).",
        )
    with _mf2:
        _min_orders = st.number_input(
            "Minimaal gemiddeld aantal bestellingen per locatie", 0.0, 100.0, 0.0, 0.1,
            key="cls_min_orders",
            format="%.1f",
            help="Artikelen met gem. orders/locatie < deze waarde worden uitgesloten.",
        )

    # ── Aggregatiemethode: vast op geometrisch ──────────────────────────
    _score_methode = "geometrisch"
    _epsilon = 1.0

    _min_loc = st.number_input("Minimaal aantal klantlocaties", 0, 100, 5, 1, key="cls_min_loc")
    # ArticleType-filter: vast op critical + onbekend
    _art_types = ("critical", "onbekend")
    st.caption("Alleen kritieke componenten en componenten zonder bekende categorie worden beoordeeld.")

    _params = ClassificatieParams(
        threshold=float(_thr),
        selectie_modus=_sel_modus,
        top_n=int(_top_n),
        top_pct=float(_top_pct),
        weight_prijs=float(_w_prijs),
        weight_locaties=float(_w_loc),
        weight_orders=float(_w_ord),
        orders_power=float(_ord_pow),
        min_prijs=float(_min_prijs),
        min_orders=float(_min_orders),
        min_klantlocaties=int(_min_loc),
        article_type_filter=_art_types,
        score_methode=_score_methode,
        epsilon=float(_epsilon),
    )

    st.divider()

    # ── Run-knop ──
    _col_run, _col_apply = st.columns([1, 1])
    with _col_run:
        _run_cls = st.button("Bereken ranglijst", type="primary", key="cls_run")
    with _col_apply:
        _apply_cls = st.button("Nieuwe selectie toevoegen", key="cls_apply",
                               disabled=("cls_result" not in st.session_state))

    if _run_cls:
        if _cls_bron_modus == "erp_documenten" and (_cls_doc_upload is None or _cls_doc1_upload is None):
            st.error("Upload zowel Document als Document 1 om de ranglijst te berekenen.")
            st.stop()
        try:
            with st.spinner("Classificatie berekenen…"):
                if _cls_bron_modus == "erp_documenten":
                    _df_raw = laad_erp_documenten(_cls_doc_upload, _cls_doc1_upload)
                    # Ververs ook de centrale bron (Overzicht/Kosten/Budget/...)
                    # met dezelfde twee documenten, zodat de hele tool met de
                    # nieuwste artikel-/orderdata rekent.
                    _nieuwe_excel_bytes = df_naar_filtered_excel_bytes(_df_raw)
                    with open(OPGESLAGEN_EXCEL_PATH + ".tmp", "wb") as _opgeslagen_excel:
                        _opgeslagen_excel.write(_nieuwe_excel_bytes)
                    os.replace(OPGESLAGEN_EXCEL_PATH + ".tmp", OPGESLAGEN_EXCEL_PATH)
                    st.session_state.pop("bron_excel_bytes", None)
                    st.session_state.pop("overzicht_df", None)
                    invalidate_caches()
                else:
                    # De (trage) Excel-parse wordt gecachet, zodat alleen de
                    # gevectoriseerde scoring opnieuw draait bij parameter-tweaks.
                    _df_raw = _cached_laad_ruwe_dataset(_excel_bytes, _cls_sheet)
                _bron_excel = None
                _miss = controleer_kolommen(_df_raw)
                if _miss:
                    raise ValueError(f"Ontbrekende kolommen: {_miss}")
                # Eerst basis-filteren, daarna scoren: de min-max-normalisatie
                # gaat zo over de artikelenset NÁ de harde filters. Top-n volgt
                # op de gescoorde set.
                _df_basis    = pas_basis_filters_toe(_df_raw, _params)
                _df_scored   = bereken_scores(_df_basis, _params)
                _df_filtered = _df_scored.copy()
                _run_code_col = next((c for c in [
                    "Verkooporderregel artikel.Artikel.Artikelcode",
                    "Artikelcode", "Code",
                ] if c in _df_filtered.columns), None)
                if _run_code_col is None:
                    raise ValueError("Geen artikelcode-kolom gevonden.")

                _bestaande_selectie = get_classificatie_info()
                _bestaande_codes = set(_bestaande_selectie.get("items", {}))
                _huidig_overzicht = st.session_state.get("overzicht_df")
                if _huidig_overzicht is not None and not _huidig_overzicht.empty:
                    _bestaande_codes.update(
                        str(code).strip() for code in _huidig_overzicht.index
                    )
                _run_codes = _df_filtered[_run_code_col].astype(str).str.strip()

                # Reeds-geselecteerde componenten die niet (meer) door de
                # huidige harde filters komen (bv. lagere prijs/orders in
                # een nieuw geüploade dataset) blijven toch in de tabel staan,
                # met verse parameterwaarden uit die nieuwe dataset.
                _te_herstellen = _bestaande_codes - set(_run_codes)
                if _te_herstellen:
                    _df_scored_alle = bereken_scores(_df_raw, _params)
                    _alle_codes = (
                        _df_scored_alle[_run_code_col].astype(str).str.strip()
                    )
                    _herstel = _df_scored_alle.loc[_alle_codes.isin(_te_herstellen)]
                    if not _herstel.empty:
                        _df_filtered = pd.concat(
                            [_df_filtered, _herstel], ignore_index=False
                        )
                        _run_codes = (
                            _df_filtered[_run_code_col].astype(str).str.strip()
                        )

                _nieuwe_kandidaten = _df_filtered.loc[
                    ~_run_codes.isin(_bestaande_codes)
                ]
                _nieuwe_top_indices = (
                    _nieuwe_kandidaten["Gewogen_Score"]
                    .sort_values(ascending=False, kind="stable")
                    .head(max(int(_params.top_n), 0))
                    .index
                )
                _df_filtered["Classificatie_Beslissing"] = "Niet opnemen"
                _df_filtered.loc[
                    _nieuwe_top_indices, "Classificatie_Beslissing"
                ] = "Opnemen in lijst"
                # Reeds-geselecteerde componenten blijven altijd opgenomen,
                # ook als hun (verse) rij buiten de harde filters valt.
                _df_filtered.loc[
                    _run_codes.isin(_bestaande_codes), "Classificatie_Beslissing"
                ] = "Opnemen in lijst"
                _payload     = bouw_selectie_payload(
                    _df_filtered, _params, bron_excel=_bron_excel
                )
            st.session_state.cls_result   = _df_filtered
            st.session_state.cls_payload  = _payload
            st.session_state.cls_params   = _params
            st.session_state.cls_raw      = _df_raw
            _sel_info = (
                f"top {_params.top_n}" if _params.selectie_modus == "top_n"
                else f"top {_params.top_pct:.0f}% per criterium" if _params.selectie_modus == "top_pct_all"
                else f"drempel ≥ {_params.threshold}"
            )
            st.toast(f"{_payload['n_items']} componenten geselecteerd "
                     f"({_sel_info})", icon="✅")
            if _cls_bron_modus == "erp_documenten":
                st.toast("Bronbestand voor de rest van de tool ververst.", icon="🔄")
                st.rerun()
        except Exception as e:
            st.error(f"Fout tijdens classificatie: {e}")

    # ── Resultaten ──
    if "cls_result" in st.session_state:
        _res = st.session_state.cls_result
        _pl  = st.session_state.cls_payload

        _n_tot     = len(_res)
        _n_opnemen = (_res["Classificatie_Beslissing"] == "Opnemen in lijst").sum()
        _bestaande_selectie = get_classificatie_info()
        _bestaande_codes = set(_bestaande_selectie.get("items", {}))
        _huidig_overzicht = st.session_state.get("overzicht_df")
        if _huidig_overzicht is not None and not _huidig_overzicht.empty:
            _bestaande_codes.update(
                str(code).strip() for code in _huidig_overzicht.index
            )

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Componenten na ondergrenzen", _n_tot)
        _m2.metric("Al in overzicht", len(_bestaande_codes))
        _m3.metric("Nieuwe voorselectie", int(_n_opnemen),
                   delta=f"{_n_opnemen/_n_tot*100:.0f}%" if _n_tot else "—")
        _m4.metric("Levertijd controleren",
                   _pl["lt_overzicht"]["default"] + _pl["lt_overzicht"]["ontbreekt"])

        # Ranglijst met een handmatige eindselectie voor BPA.
        _code_col = next((c for c in [
            "Verkooporderregel artikel.Artikel.Artikelcode", "Artikelcode", "Code"
        ] if c in _res.columns), None)
        _show_cols = [c for c in [
            _code_col,
            "Priority",
            "Standaard verkoopprijs",
            "Aantal_klantlocaties_met_orders_5jr",
            "Gem_orders_per_klantlocatie_5jr",
            "Gewogen_Score", "Classificatie_Beslissing",
            "Hoofdleverancier.Levertijd",
        ] if c in _res.columns]
        _df_show = _res[_show_cols].sort_values(
            "Gewogen_Score", ascending=False, kind="stable"
        )
        _df_show.insert(0, "Rang", range(1, len(_df_show) + 1))
        _codes_show = _df_show[_code_col].astype(str).str.strip()
        _df_show.insert(
            1,
            "Selecteren",
            _df_show["Classificatie_Beslissing"].eq("Opnemen in lijst")
            & ~_codes_show.isin(_bestaande_codes),
        )
        _df_show.insert(
            2,
            "Status",
            np.where(
                _codes_show.isin(_bestaande_codes), "Al in overzicht", "Nieuw"
            ),
        )
        _df_show = _df_show.drop(columns=["Classificatie_Beslissing"]).rename(columns={
            "Verkooporderregel artikel.Artikel.Artikelcode": "Artikelnummer",
            "Artikelcode": "Artikelnummer",
            "Code": "Artikelnummer",
            "Standaard verkoopprijs": "Verkoopprijs",
            "Aantal_klantlocaties_met_orders_5jr": "Klantlocaties",
            "Gem_orders_per_klantlocatie_5jr": "Gem. bestellingen per locatie",
            "Gewogen_Score": "Prioriteitsscore",
            "Hoofdleverancier.Levertijd": "Levertijd (dagen)",
        })

        _df_bestaand = _df_show[_df_show["Status"] == "Al in overzicht"].copy()
        _df_nieuw = _df_show[_df_show["Status"] == "Nieuw"].copy()

        if not _df_bestaand.empty:
            with st.expander(
                f"Al in overzicht ({len(_df_bestaand)}) — niet opnieuw selecteerbaar",
                expanded=False,
            ):
                st.dataframe(
                    _df_bestaand.drop(columns=["Selecteren"]),
                    use_container_width=True,
                    hide_index=True,
                )

        _edited_selection = st.data_editor(
            _df_nieuw,
            use_container_width=True,
            height=500,
            hide_index=True,
            disabled=[c for c in _df_show.columns if c != "Selecteren"],
            column_config={
                "Selecteren": st.column_config.CheckboxColumn("Naar overzicht"),
                "Verkoopprijs": st.column_config.NumberColumn(format="€ %.2f"),
                "Prioriteitsscore": st.column_config.NumberColumn(format="%.1f"),
            },
            key="cls_selection_editor",
        )

        _selected_indices = _df_nieuw.index[
            _edited_selection["Selecteren"].to_numpy()
        ]
        _res_selected = _res.copy()
        _res_selected["Classificatie_Beslissing"] = "Niet opnemen"
        _res_selected.loc[_selected_indices, "Classificatie_Beslissing"] = "Opnemen in lijst"
        _res_codes = _res_selected[_code_col].astype(str).str.strip()
        _res_selected.loc[
            _res_codes.isin(_bestaande_codes), "Classificatie_Beslissing"
        ] = "Opnemen in lijst"
        _nieuwe_payload = bouw_selectie_payload(
            _res_selected, st.session_state.cls_params, bron_excel=None
        )
        st.session_state.cls_payload = _merge_selectie_payloads(
            _bestaande_selectie, _nieuwe_payload
        )
        st.caption(
            f"{len(_selected_indices)} nieuwe componenten geselecteerd; "
            f"{len(_bestaande_selectie.get('items', {}))} bestaande componenten "
            "blijven behouden. Klik op 'Nieuwe selectie toevoegen' om bij te werken."
        )

        # Download
        _csv = _df_show.to_csv(sep=";", decimal=",", index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download gescoorde tabel (CSV)",
            data=_csv, file_name=f"classificatie_{date.today()}.csv",
            mime="text/csv",
        )

    # ── Apply: schrijf bpa_selectie.json + invalideer overzicht ──
    if _apply_cls and "cls_payload" in st.session_state:
        try:
            schrijf_selectie_json(st.session_state.cls_payload, SELECTIE_PATH)
            invalidate_caches()
            st.session_state.pop("overzicht_df", None)
            st.toast(
                f"Selectie opgeslagen ({st.session_state.cls_payload['n_items']} componenten als whitelist).",
                icon="✅",
            )
            st.rerun()
        except Exception as e:
            st.error(f"Kon bpa_selectie.json niet schrijven: {e}")

