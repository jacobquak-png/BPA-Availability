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

