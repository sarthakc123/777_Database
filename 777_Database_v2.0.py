# chem_reaction_tracker_supabase.py
# Fixed version with better error handling

import io
import csv
import tempfile
import requests
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path

import streamlit as st
import pandas as pd
import itertools

# ---------- Optional RDKit (skip if not needed) ----------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# =========================
# Config & Constants
# =========================
ID_PAD_WIDTH = 6  # JPM000001
DEFAULT_CONDITIONS = ["Air", "Glovebox"]
DEFAULT_BASES = ["LiOH", "BTMG", "TMSOK", "P2Et", "TMSONa", "MTBD", "DBU"]
DEFAULT_SOLVENTS = ["DMSO", "THF", "1,4-dioxane", "Hexane","DMF", "2-MeTHF", "Toluene", "Acetone"]
DEFAULT_CHEMISTS = [("JPM", "John Marin"), ("SD", "Swapna Debnath"), ("CKC","Chieh-Kai Chan"), ("MDB","Dr. Martin Burke")]
DEFAULT_COUPLINGS = ["SMC", "BHC", "Amide","AMC","Deprotection"]
MAX_SETS_UI = 4


# =========================
# Supabase REST API Setup
# =========================
@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Initialize Supabase REST API client"""
    try:
        SUPABASE_URL = st.secrets["supabase"]["url"]
        SUPABASE_KEY = st.secrets["supabase"]["anon_key"]

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        # Test connection
        response = requests.get(f"{SUPABASE_URL}/rest/v1/chemists?limit=1", headers=headers)
        if response.status_code == 200:
            return SUPABASE_URL, headers
        else:
            st.error(f"Supabase connection failed: {response.status_code} - {response.text}")
            st.stop()

    except KeyError as e:
        st.error(f"Missing secret key: {e}. Please check your .streamlit/secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        st.stop()


# =========================
# Database Operations via REST API
# =========================
def api_request(
    method: str,
    endpoint: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Optional[dict]:
    """Generic Supabase PostgREST wrapper with query params + JSON bodies."""
    try:
        url, headers = get_supabase_client()
        full_url = f"{url}/rest/v1/{endpoint}"

        req_headers = headers.copy()
        if method.upper() in ("POST", "PUT", "PATCH"):
            # Ask PostgREST to return inserted/updated rows
            req_headers["Prefer"] = "return=representation"

        if method.upper() == "GET":
            response = requests.get(full_url, headers=req_headers, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(full_url, headers=req_headers, json=json_body, params=params, timeout=30)
        elif method.upper() == "PUT":
            response = requests.put(full_url, headers=req_headers, json=json_body, params=params, timeout=30)
        elif method.upper() == "PATCH":
            response = requests.patch(full_url, headers=req_headers, json=json_body, params=params, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(full_url, headers=req_headers, params=params, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        if response.status_code not in (200, 201, 204):
            st.error(f"API Error {response.status_code} for {method} {endpoint}: {response.text}")
            return None

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except Exception:
            return response.text

    except Exception as e:
        st.error(f"Request failed for {method} {endpoint}: {e}")
        return None

def get_chemists() -> List[dict]:
    """Fetch all chemists"""
    result = api_request("GET", "chemists?select=*&order=initials")
    return result if result is not None else []


def add_chemist(initials: str, full_name: str) -> bool:
    """Add new chemist"""
    chemist_data = {"initials": initials, "full_name": full_name}
    result = api_request("POST", "chemists", chemist_data)
    return result is not None


def get_reactions(limit: int = 100) -> List[dict]:
    """Fetch recent reactions"""
    reactions = api_request("GET", f"reactions?select=*&order=created_at.desc&limit={limit}")
    if not reactions:
        return []

    # For each reaction, get its conditions
    for reaction in reactions:
        rid = reaction["id"]
        conditions = api_request("GET", f"reaction_conditions?reaction_id=eq.{rid}&select=*&order=set_index")
        if conditions:
            reaction["condition_count"] = len(conditions)
            reaction["first_condition"] = conditions[0]
        else:
            reaction["condition_count"] = 0
            reaction["first_condition"] = {}

    return reactions


def next_reaction_index(initials: str) -> int:
    """Get next reaction index for chemist"""
    reactions = api_request("GET", f"reactions?id=like.{initials}*&select=id&order=id.desc&limit=1")
    if not reactions:
        return 1

    last_id = reactions[0]["id"]
    suffix = "".join(ch for ch in last_id[len(initials):] if ch.isdigit())
    try:
        return int(suffix) + 1
    except:
        return 1


def insert_reaction_with_sets(
        rid: str, initials: str,
        s_smi: Optional[str], e_smi: Optional[str], c_smi: Optional[str],
        expected_smi: Optional[str], rxn_scale_mol: Optional[float],
        oligomer_type: Optional[str], sets: List[dict],comments: Optional[str]=None
) -> bool:
    """Insert reaction and condition sets"""

    try:
        # Insert reaction first
        reaction_data = {
            "id": rid,
            "chemist_initials": initials,
            "s_block_smiles": s_smi,
            "e_block_smiles": e_smi,
            "c_block_smiles": c_smi,
            "oligomer_type": oligomer_type,
            "expected_smiles": expected_smi,
            "rxn_scale_mol": rxn_scale_mol,
            "created_at": datetime.utcnow().isoformat(),
            "comments": comments
        }

        #st.write("Debug: Inserting reaction ", reaction_data)  # Debug line
        reaction_result = api_request("POST", "reactions", reaction_data)

        if not reaction_result:
            st.error("Failed to insert reaction")
            return False

        # Insert condition sets
        for cs in sets:
            condition_data = {
                "reaction_id": rid,
                "set_index": cs["set_index"],
                "coupling_type": cs["coupling"],
                "temperature_c": cs["temp"],
                "condition": cs["cond"],
                "base": cs["base"],
                "time_hours": cs["time_h"],
                "solvent": cs["solv"]
            }

            #st.write(f"Debug: Inserting condition set {cs['set_index']}:", condition_data)  # Debug line
            condition_result = api_request("POST", "reaction_conditions", condition_data)

            if not condition_result:
                st.error(f"Failed to insert condition set {cs['set_index']}")
                return False

        return True

    except Exception as e:
        st.error(f"Error in insert_reaction_with_sets: {str(e)}")
        return False


# =========================
# Helper Functions
# =========================
def format_reaction_id(initials: str, n: int, width: int = ID_PAD_WIDTH) -> str:
    return f"{initials}{n:0{width}d}"


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return smiles
    if RDKit_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol)
        except:
            pass
    return smiles


def molfile_to_smiles(path: str) -> Optional[str]:
    if not RDKit_AVAILABLE:
        return None
    try:
        smi = None
        if path.lower().endswith(".mol"):
            mol = Chem.MolFromMolFile(path)
            if mol: smi = Chem.MolToSmiles(mol)
        elif path.lower().endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path)
            mol = next((m for m in suppl if m is not None), None)
            if mol: smi = Chem.MolToSmiles(mol)
        return canonicalize_smiles(smi) if smi else None
    except:
        return None


def smiles_image_bytes(smiles: str, size: Tuple[int, int] = (280, 280)) -> Optional[bytes]:
    if not (RDKit_AVAILABLE and smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except:
        return None


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Molecule Maker Lab", page_icon="24-MML-3Col (2).jpg", layout="wide")
# Header row with title on left, logo on right
left, right = st.columns([4, 1.5], vertical_alignment="center")
with left:
    st.title("777 Reaction Tracker")
    st.caption("Log all your reactions here!")
with right:
    st.image("24-MML-3Col (2).jpg",use_container_width=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Initialize Supabase connection
get_supabase_client()

# Sidebar
with st.sidebar:
    st.sidebar.image("mml-team-image-scaled.jpg", use_container_width=True)
    st.header("Chemist")

    # Load chemists
    chemists_data = get_chemists()
    chemists = [(c["initials"], c.get("full_name", "")) for c in chemists_data]

    if chemists:
        labels = [f"{i} — {n}" if n else i for i, n in chemists]
        idx = st.selectbox("Select chemist", options=list(range(len(chemists))), format_func=lambda i: labels[i])
        selected_initials = chemists[idx][0]
    else:
        selected_initials = "JPM"
        st.warning("No chemists found. Add one below.")

    with st.expander("➕ Add a new chemist"):
        new_initials = st.text_input("Initials (e.g., JPM)", max_chars=6)
        new_name = st.text_input("Full name (optional)")
        if st.button("Add chemist"):
            if new_initials.strip():
                if add_chemist(new_initials.strip().upper(), new_name.strip()):
                    st.success(f"Added {new_initials.strip().upper()}.")
                    st.rerun()
                else:
                    st.error("Failed to add chemist.")
            else:
                st.error("Please enter initials.")

# Main form
st.subheader("Add a reaction")
with st.form("reaction_form", clear_on_submit=False):
    next_idx = next_reaction_index(selected_initials)
    preview_id = format_reaction_id(selected_initials, next_idx)
    st.info(f"Next ID will be **{preview_id}** for chemist **{selected_initials}**.")

    st.markdown("#### S/E/C Blocks — paste SMILES or upload MOL/SDF")


    def smiles_block(label_key: str):
        colA, colB, colC = st.columns([2, 1, 1])
        smi = colA.text_input(f"{label_key} SMILES", key=f"{label_key}_txt")
        uploaded = colB.file_uploader(f"Upload {label_key} (.mol/.sdf)", type=["mol", "sdf"], key=f"{label_key}_file")
        if uploaded:
            if RDKit_AVAILABLE:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tf:
                    tf.write(uploaded.getvalue())
                    tf.flush()
                    smi_conv = molfile_to_smiles(tf.name)
                if smi_conv:
                    smi = smi_conv
                    st.success(f"{label_key}: Converted file → SMILES")
                else:
                    st.warning(f"{label_key}: Could not parse file to SMILES.")
            else:
                st.warning("RDKit not installed — paste SMILES instead.")
        smi = canonicalize_smiles(smi)
        if smi:
            img = smiles_image_bytes(smi)
            if img: colC.image(img, caption=f"{label_key}")
        return smi


    s_smi = smiles_block("S-Block")
    e_smi = smiles_block("E-Block")
    c_smi = smiles_block("C-Block")
    # Positional multi-SEC (optional). One variant per line; nth S pairs with nth E and nth C.
    with st.expander("Add multiple SEC variants (positional pairing)"):
        st.caption("Enter extra variants, one per line (Sₙ pairs with Eₙ and Cₙ).")
        s_multi = st.text_area("Extra S variants (one per line)", key="s_multi_pos").strip()
        e_multi = st.text_area("Extra E variants (one per line)", key="e_multi_pos").strip()
        c_multi = st.text_area("Extra C variants (one per line)", key="c_multi_pos").strip()


    def _canonical_list_lines(txt: str) -> list:
        if not txt: return []
        vals = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                vals.append(None)  # keep position even if blank
            else:
                vals.append(canonicalize_smiles(line))
        return vals


    st.markdown("---")

    # Number of condition sets (1..4)
    if "num_sets" not in st.session_state:
        st.session_state["num_sets"] = 1
    num_sets = st.number_input("Number of condition sets", min_value=1, max_value=MAX_SETS_UI,
                               value=st.session_state["num_sets"], step=1)
    if st.form_submit_button("Apply set count"):
        st.session_state["num_sets"] = int(num_sets)
        st.rerun()
    repeats = st.session_state["num_sets"]

    # Oligomer flag from blocks
    block_count = sum(1 for x in [s_smi, e_smi, c_smi] if x)
    oligomer_type = "dimer" if block_count == 2 else "trimer" if block_count == 3 else None
    st.caption(f"Blocks detected: {block_count or 0}. Oligomer: **{oligomer_type or '—'}**.")


    def condition_set(i: int):
        st.markdown(f"**Condition set {i}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            coupling = st.selectbox(f"Coupling Type {i} *", options=DEFAULT_COUPLINGS, key=f"cpl_{i}")
            temp = st.number_input(f"Temperature {i} (°C) *", value=25.0, step=0.5, key=f"temp_{i}")
        with c2:
            cond_opt = st.selectbox(f"Condition {i}", options=DEFAULT_CONDITIONS + ["Other…"], key=f"cond_{i}")
            cond = st.text_input(f"Condition {i} (custom)",
                                 key=f"cond_custom_{i}") if cond_opt == "Other…" else cond_opt
            base_opt = st.selectbox(f"Base {i}", options=DEFAULT_BASES + ["Other…"], key=f"base_{i}")
            base = st.text_input(f"Base {i} (custom) *", key=f"base_custom_{i}") if base_opt == "Other…" else base_opt
        with c3:
            time_h = st.number_input(f"Time {i} (hours) *", min_value=0.0, value=0.0, step=0.25, key=f"time_{i}")
            solv_opt = st.selectbox(f"Solvent {i}", options=DEFAULT_SOLVENTS + ["Other…"], key=f"solv_{i}")
            solv = st.text_input(f"Solvent {i} (custom) *",
                                 key=f"solv_custom_{i}") if solv_opt == "Other…" else solv_opt
        return {"set_index": i, "coupling": coupling, "temp": float(temp), "cond": cond,
                "base": base, "time_h": float(time_h), "solv": solv}


    sets: List[dict] = []
    for i in range(1, repeats + 1):
        if i > 1: st.markdown("---")
        sets.append(condition_set(i))

    st.markdown("---")
    # Expected SMILES (one per line, maps to row n)
    expected_list_raw = st.text_area(
        "Expected compound SMILES (one per line; maps to row n) *",
        help="Line n corresponds to Sₙ/Eₙ/Cₙ. Leave a line blank to set None for that row."
    ).strip()
    EXP_list = _canonical_list_lines(expected_list_raw)
    rxn_scale_mol = st.number_input("RXN_Scale (mol) *", min_value=0.0, value = None, step=0.001, format="%f")
    # Comments (free text)
    comments = st.text_area("Comments (optional)", help="Notes, observations, yields, deviations, etc.")

    # Validation
    errors = []
    if not any([s_smi, e_smi, c_smi]) and not any([s_multi, e_multi, c_multi]):
        errors.append("Provide at least one of S/E/C block SMILES (single or multi-SEC).")
    if rxn_scale_mol is None or rxn_scale_mol <= 0:
        errors.append("RXN_Scale (mol) must be > 0.")
    for i, cs in enumerate(sets, start=1):
        if not cs["coupling"] or cs["temp"] is None:
            errors.append(f"Set {i}: Coupling Type and Temperature are required.")
        if cs["time_h"] is None or cs["time_h"] < 0:
            errors.append(f"Set {i}: Time (hours) must be ≥ 0.")
        if not cs["base"] or cs["base"].strip() == "":
            errors.append(f"Set {i}: Base is required (specify custom if 'Other…').")
        if not cs["solv"] or cs["solv"].strip() == "":
            errors.append(f"Set {i}: Solvent is required (specify custom if 'Other…').")

    submitted = st.form_submit_button("Save reaction", type="primary")
    if submitted:
        if errors:
            for e in errors:
                st.error(e)
        else:
            # Build S/E/C positional lists from single fields + (optional) multi-SEC text areas
            S_list = ([canonicalize_smiles(s_smi)] if s_smi else []) + (
                _canonical_list_lines(s_multi) if 's_multi' in locals() else [])
            E_list = ([canonicalize_smiles(e_smi)] if e_smi else []) + (
                _canonical_list_lines(e_multi) if 'e_multi' in locals() else [])
            C_list = ([canonicalize_smiles(c_smi)] if c_smi else []) + (
                _canonical_list_lines(c_multi) if 'c_multi' in locals() else [])

            # If SEC lists are all empty, nothing to save
            if not (S_list or E_list or C_list):
                st.error("No SEC variants to save.")
                st.stop()

            # Positional length = max of SEC lists
            n = max(len(S_list), len(E_list), len(C_list))

            # Expected list must be provided and match 'n' (1-to-1)
            if len(EXP_list) != n:
                st.error(f"Expected list length ({len(EXP_list)}) must match the number of SEC rows ({n}).")
                st.stop()

            # Build rows, skipping positions where all S/E/C are None/empty
            rows = []
            for i in range(n):
                s_val = S_list[i] if i < len(S_list) else None
                e_val = E_list[i] if i < len(E_list) else None
                c_val = C_list[i] if i < len(C_list) else None
                exp_val = EXP_list[i]  # exact 1-to-1 mapping
                if any([s_val, e_val, c_val]):  # keep only meaningful SEC rows
                    rows.append((s_val, e_val, c_val, exp_val))

            if not rows:
                st.error("All SEC rows are empty.")
                st.stop()

            # Reserve contiguous IDs; one reaction per positional row
            base_idx = next_reaction_index(selected_initials)
            saved, failures = 0, 0

            for offset, (s_val, e_val, c_val, exp_val) in enumerate(rows):
                rid = format_reaction_id(selected_initials, base_idx + offset)
                present = sum(1 for x in [s_val, e_val, c_val] if x)
                oligo = "dimer" if present == 2 else ("trimer" if present == 3 else None)

                ok = insert_reaction_with_sets(
                    rid=rid,
                    initials=selected_initials,
                    s_smi=s_val, e_smi=e_val, c_smi=c_val,
                    expected_smi=exp_val,
                    rxn_scale_mol=float(rxn_scale_mol),
                    oligomer_type=oligo,
                    sets=sets,
                    comments=comments if "comments" in locals() else None
                )
                if ok:
                    saved += 1
                else:
                    failures += 1

            if saved:
                st.success(f"Saved **{saved}** reaction(s) (positional SEC + Expected) with the same condition set(s).")
                st.balloons()
            if failures:
                st.warning(f"{failures} reaction(s) failed to save. Check data/RLS and try again.")

# =========================
# Recent submissions — consolidated join (reactions × reaction_conditions)
# =========================
st.markdown("---")
st.subheader("Recent submissions (consolidated)")

# 1) Fetch raw tables
rx_cols = "id,chemist_initials,oligomer_type,s_block_smiles,e_block_smiles,c_block_smiles,expected_smiles,rxn_scale_mol,comments,created_at"
rc_cols = "reaction_id,set_index,coupling_type,temperature_c,condition,base,time_hours,solvent"

rx = api_request("GET", "reactions", params={"select": rx_cols, "order": "created_at.desc", "limit": 200}) or []
rc = api_request("GET", "reaction_conditions", params={"select": rc_cols, "order": "reaction_id,set_index"}) or []

# 2) DataFrames
df_rx = pd.DataFrame(rx)
df_rc = pd.DataFrame(rc)

if df_rx.empty:
    st.info("No reactions yet. Add your first one above.")
else:
    # Ensure expected columns exist even if rc is empty
    for col in ["reaction_id","set_index","coupling_type","temperature_c","condition","base","time_hours","solvent"]:
        if col not in df_rc.columns:
            df_rc[col] = pd.Series(dtype="object")

    # 3) Join: one row per condition set; reactions with no sets still included (NaN on set cols)
    df_join = pd.merge(
        df_rx,
        df_rc,
        left_on="id",
        right_on="reaction_id",
        how="left",
        sort=False,
        suffixes=("", "_rc")
    )

    # 4) Order & rename columns for a tidy view
    cols = [
        "id","chemist_initials","oligomer_type","set_index","coupling_type",
        "temperature_c","base","solvent","time_hours","condition",
        "rxn_scale_mol","comments","created_at","s_block_smiles","e_block_smiles","c_block_smiles","expected_smiles"
    ]
    # Keep only those that exist (in case schema varies)
    cols = [c for c in cols if c in df_join.columns]
    df_join = df_join[cols].rename(columns={
        "id":"ID",
        "chemist_initials":"Chemist",
        "oligomer_type":"Oligomer",
        "set_index":"Set",
        "coupling_type":"Coupling",
        "temperature_c":"Temp_C",
        "time_hours":"Time_h",
        "rxn_scale_mol":"Scale_mol",
        "created_at":"Created_UTC",
        "s_block_smiles":"S_SMILES",
        "e_block_smiles":"E_SMILES",
        "c_block_smiles":"C_SMILES",
        "expected_smiles":"Expected_SMILES",
    })

    st.dataframe(df_join, use_container_width=True, hide_index=True)

    st.download_button(
        "Download consolidated_view.csv",
        df_join.to_csv(index=False).encode("utf-8"),
        "consolidated_view.csv",
        "text/csv"
    )

with st.sidebar:
    st.markdown("---")
    st.header("Export")

    if st.button("Export reactions.csv"):
        st.download_button(
            "Download consolidated_view.csv",
            df_join.to_csv(index=False).encode("utf-8"),
            "consolidated_view.csv",
            "text/csv"
        )