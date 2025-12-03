import io
import csv
import tempfile
import requests
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import streamlit as st
import pandas as pd
import uuid

# ---------- Optional RDKit ----------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# =========================
# App Config & Constants
# =========================
APP_TITLE = "777 Reaction Tracker"
LOGO_FILE = "24-MML-3Col (2).jpg"

ID_PAD_WIDTH = 6  # JPM000001
DEFAULT_CONDITIONS = ["Air", "Glovebox"]
DEFAULT_BASES = ["LiOH", "BTMG", "TMSOK", "P2Et", "TMSONa", "MTBD", "DBU"]
DEFAULT_SOLVENTS = ["DMSO", "THF", "1,4-dioxane", "Hexane", "DMF", "2-MeTHF", "Toluene", "Acetone"]
DEFAULT_CHEMISTS = [
    ("JPM", "John Marin"),
    ("SD", "Swapna Debnath"),
    ("CKC", "Chieh-Kai Chan"),
    ("MDB", "Dr. Martin Burke"),
]
DEFAULT_COUPLINGS = ["SMC", "BHC", "Amide", "AMC", "Deprotection"]
MAX_SETS_UI = 4

# =========================
# Auth helpers
# =========================
def _auth_enabled() -> bool:
    try:
        return bool(st.secrets["auth"].get("enabled", True))
    except Exception:
        return False

def _valid_user(u: str, p: str) -> bool:
    try:
        users = st.secrets["auth"]["users"]  # {username: password}
        return u in users and str(users[u]) == str(p)
    except Exception:
        return False

def maybe_set_page_config():
    if not st.session_state.get("_pc_set"):
        st.set_page_config(page_title=APP_TITLE, page_icon=LOGO_FILE, layout="wide")
        st.session_state["_pc_set"] = True

def require_login_first_page():
    if not _auth_enabled():
        return  # no auth

    if st.session_state.get("auth_user"):
        with st.sidebar:
            st.success(f"Signed in as {st.session_state['auth_user']}")
            if st.button("Log out", key="logout_btn"):
                st.session_state.pop("auth_user", None)
        return

    maybe_set_page_config()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_FILE, use_container_width=True)
        st.title(APP_TITLE)
        st.subheader("Sign in")

        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            ok = st.form_submit_button("Sign in", type="primary")

        if ok:
            if _valid_user(u.strip(), p):
                st.session_state["auth_user"] = u.strip()
            else:
                st.error("Invalid credentials.")
    st.stop()

# Gate the app
require_login_first_page()
maybe_set_page_config()

# =========================
# Supabase REST API Setup
# =========================
@st.cache_resource(show_spinner=False)
def get_supabase() -> Tuple[str, Dict[str, str]]:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        r = requests.get(f"{url}/rest/v1/chemists?select=initials&limit=1", headers=headers, timeout=30)
        if r.status_code in (200, 206):
            return url, headers

        st.error(f"Supabase connection failed: {r.status_code} - {r.text}")
        st.stop()
    except KeyError as e:
        st.error(f"Missing secret key: {e}. Please check your .streamlit/secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        st.stop()

def sb_request(
    method: str,
    endpoint: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Any:
    url, headers = get_supabase()
    full = f"{url}/rest/v1/{endpoint}"
    hdrs = headers.copy()

    if method.upper() in ("POST", "PUT", "PATCH"):
        hdrs["Prefer"] = "return=representation"

    try:
        r = requests.request(
            method=method,
            url=full,
            headers=hdrs,
            params=params,
            json=json_body,
            timeout=30,
        )
        if r.status_code not in (200, 201, 204, 206):
            st.error(f"API Error {r.status_code} for {method} {endpoint}: {r.text}")
            return None

        if r.status_code == 204 or not r.content:
            return None

        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.text
    except Exception as e:
        st.error(f"Request failed for {method} {endpoint}: {e}")
        return None

# Convenience wrappers
def get_chemists() -> List[dict]:
    res = sb_request("GET", "chemists", params={"select": "initials,full_name", "order": "initials"})
    return res if res else []

def add_chemist(initials: str, full_name: str) -> bool:
    chemist_data = {"initials": initials, "full_name": full_name}
    res = sb_request("POST", "chemists", json_body=chemist_data)
    return res is not None

def next_reaction_index(initials: str) -> int:
    res = sb_request(
        "GET",
        "reactions",
        params={"id": f"like.{initials}*", "select": "id", "order": "id.desc", "limit": 1},
    )
    if not res:
        return 1
    last_id = res[0]["id"]
    suffix = "".join(ch for ch in last_id[len(initials):] if ch.isdigit())
    try:
        return int(suffix) + 1
    except Exception:
        return 1

def insert_reaction_with_sets(
    rid: str,
    initials: str,
    s_smi: Optional[str],
    e_smi: Optional[str],
    c_smi: Optional[str],
    expected_smi: Optional[str],
    rxn_scale_mol: Optional[float],
    oligomer_type: Optional[str],
    sets: List[dict],
    comments: Optional[str] = None,
    submission_id: Optional[str] = None,
) -> bool:
    """Insert one reaction + its condition sets."""
    try:
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
            "comments": comments,
            "submission_id": submission_id,
        }
        rx_res = sb_request("POST", "reactions", json_body=reaction_data)
        if not rx_res:
            st.error(f"Failed to insert reaction {rid}")
            return False

        for cs in sets:
            condition_data = {
                "reaction_id": rid,
                "set_index": cs["set_index"],
                "coupling_type": cs["coupling"],
                "temperature_c": cs["temp"],
                "condition": cs["cond"],
                "base": cs["base"],
                "time_hours": cs["time_h"],
                "solvent": cs["solv"],
                "yrts": cs.get("yrts"),
                "assay_yield": cs.get("assay_yield"),
                "purity_pct": cs.get("purity_pct"),
                "entry_mode": cs.get("entry_mode"),
            }

            condition_result = sb_request(
                "POST",
                "reaction_conditions",
                json_body=condition_data,
            )

            if not condition_result:
                st.error(f"Failed to insert condition set {cs['set_index']} for {rid}")
                return False

        return True
    except Exception as e:
        st.error(f"Error in insert_reaction_with_sets for {rid}: {e}")
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
        except Exception:
            pass
    return smiles

def molfile_to_smiles(path: str) -> Optional[str]:
    if not RDKit_AVAILABLE:
        return None
    try:
        smi = None
        if path.lower().endswith(".mol"):
            mol = Chem.MolFromMolFile(path)
            if mol:
                smi = Chem.MolToSmiles(mol)
        elif path.lower().endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path)
            mol = next((m for m in suppl if m is not None), None)
            if mol:
                smi = Chem.MolToSmiles(mol)
        return canonicalize_smiles(smi) if smi else None
    except Exception:
        return None

from urllib.parse import quote  # add at top

def smiles_image_bytes(smiles: str, size=(280, 280)) -> Optional[bytes]:
    """Render SMILES using RDKit if available, otherwise use NCI Cactus API."""
    if not smiles:
        return None

    # Try RDKit first
    if RDKit_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=size)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            pass

    # Fallback: NCI Cactus API
    try:
        safe = quote(smiles, safe="")  # URL-encode the SMILES
        url = f"https://cactus.nci.nih.gov/chemical/structure/{safe}/image"
        r = requests.get(url, timeout=10)

        # Make sure we really got an image, not HTML
        if (
            r.status_code == 200
            and r.headers.get("content-type", "").startswith("image/")
        ):
            return r.content
        return None
    except Exception:
        return None


def _canonical_list_lines(txt: str) -> list:
    if not txt:
        return []
    vals = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            vals.append(None)
        else:
            vals.append(canonicalize_smiles(line))
    return vals

# =========================
# UI Shell (header, sidebar)
# =========================
hdr_l, hdr_r = st.columns([4, 1.5])
with hdr_l:
    st.title(APP_TITLE)
    st.caption("Log all your reactions here!")
with hdr_r:
    st.image(LOGO_FILE, use_container_width=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Ensure Supabase is reachable
get_supabase()

# Sidebar: chemist picker & add-new
with st.sidebar:
    st.image("mml-team-image-scaled.jpg", use_container_width=True)
    st.header("Chemist")

    chemists_data = get_chemists()
    chemists = [(c["initials"], c.get("full_name", "")) for c in chemists_data]

    if chemists:
        labels = [f"{i} — {n}" if n else i for i, n in chemists]
        idx = st.selectbox(
            "Select chemist",
            options=list(range(len(chemists))),
            format_func=lambda i: labels[i],
            key="chemist_select",
        )
        selected_initials = chemists[idx][0]
    else:
        selected_initials = "JPM"
        st.warning("No chemists found. Add one below.")

    with st.expander("➕ Add a new chemist"):
        new_initials = st.text_input("Initials (e.g., JPM)", max_chars=6, key="new_initials")
        new_name = st.text_input("Full name (optional)", key="new_fullname")
        if st.button("Add chemist", key="add_chemist_btn"):
            if new_initials.strip():
                if add_chemist(new_initials.strip().upper(), new_name.strip()):
                    st.success(f"Added {new_initials.strip().upper()}.")
                    st.rerun()
                else:
                    st.error("Failed to add chemist.")
            else:
                st.error("Please enter initials.")

    st.markdown("---")
    st.header("Export (consolidated)")

# =========================
# Tabs
# =========================
tab_new, tab_recent = st.tabs(["Add Reaction", "Recent Submissions"])

# ----- TAB 1: Add Reaction -----
with tab_new:
    st.subheader("Add a reaction")

    with st.form("reaction_form", clear_on_submit=False):
        next_idx = next_reaction_index(selected_initials)
        preview_id = format_reaction_id(selected_initials, next_idx)
        st.info(f"Next ID will be **{preview_id}** for chemist **{selected_initials}**.")

        st.markdown("#### S/E/C Blocks")

        def smiles_block(label_key: str):
            colA, colB, colC = st.columns([2, 1, 1])
            smi = colA.text_input(f"{label_key} SMILES", key=f"{label_key}_txt")
            uploaded = colB.file_uploader(
                f"Upload {label_key} (.mol/.sdf)", type=["mol", "sdf"], key=f"{label_key}_file"
            )
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
                    st.warning("RDKit not installed, paste SMILES instead.")
            smi = canonicalize_smiles(smi)
            if smi:
                img = smiles_image_bytes(smi)
                if img:
                    try:
                        colC.image(img, caption=f"{label_key}")
                    except Exception:
                        st.warning(f"Could not render preview for {label_key} (invalid image).")


        s_smi = smiles_block("S-Block")
        e_smi = smiles_block("E-Block")
        c_smi = smiles_block("C-Block")

        with st.expander("Add multiple SEC variants (positional pairing)"):
            st.caption("Enter extra variants, one per line (Sₙ pairs with Eₙ and Cₙ).")
            s_multi = st.text_area("Extra S variants (one per line)", key="s_multi_pos")
            e_multi = st.text_area("Extra E variants (one per line)", key="e_multi_pos")
            c_multi = st.text_area("Extra C variants (one per line)", key="c_multi_pos")

        st.markdown("---")

        # Number of condition sets (1..4)
        if "num_sets" not in st.session_state:
            st.session_state["num_sets"] = 1
        num_sets = st.number_input(
            "Number of condition sets",
            min_value=1,
            max_value=MAX_SETS_UI,
            value=st.session_state["num_sets"],
            step=1,
        )
        if st.form_submit_button("Apply set count"):
            st.session_state["num_sets"] = int(num_sets)
            st.rerun()
        repeats = st.session_state["num_sets"]

        # Oligomer flag
        block_count = sum(1 for x in [s_smi, e_smi, c_smi] if x)
        oligomer_type = "dimer" if block_count == 2 else "trimer" if block_count == 3 else None
        st.caption(f"Blocks detected: {block_count or 0}. Oligomer: **{oligomer_type or '—'}**.")

        def condition_set(i: int):
            st.markdown(f"**Condition set {i}**")
            c1, c2, c3 = st.columns(3)

            with c1:
                coupling = st.selectbox(
                    f"Coupling Type {i} *",
                    options=DEFAULT_COUPLINGS,
                    key=f"cpl_{i}",
                )
                temp = st.number_input(
                    f"Temperature {i} (°C) *",
                    value=25.0,
                    step=0.5,
                    key=f"temp_{i}",
                )
                yrts = st.number_input(
                    f"YRTS {i}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    key=f"yrts_{i}",
                )

            with c2:
                cond_opt = st.selectbox(
                    f"Condition {i}",
                    options=DEFAULT_CONDITIONS + ["Other…"],
                    key=f"cond_{i}",
                )
                cond = (
                    st.text_input(f"Condition {i} (custom)", key=f"cond_custom_{i}")
                    if cond_opt == "Other…"
                    else cond_opt
                )

                base_opt = st.selectbox(
                    f"Base {i}",
                    options=DEFAULT_BASES + ["Other…"],
                    key=f"base_{i}",
                )
                base = (
                    st.text_input(f"Base {i} (custom) *", key=f"base_custom_{i}")
                    if base_opt == "Other…"
                    else base_opt
                )

                entry_mode = st.selectbox(
                    f"Mode {i}",
                    options=["Spike", "Tom", "Jerry", "Manual"],
                    index=3,
                    key=f"mode_{i}",
                )

            with c3:
                time_h = st.number_input(
                    f"Time {i} (hours) *",
                    min_value=0.0,
                    value=0.0,
                    step=0.25,
                    key=f"time_{i}",
                )
                solv_opt = st.selectbox(
                    f"Solvent {i}",
                    options=DEFAULT_SOLVENTS + ["Other…"],
                    key=f"solv_{i}",
                )
                solv = (
                    st.text_input(f"Solvent {i} (custom) *", key=f"solv_custom_{i}")
                    if solv_opt == "Other…"
                    else solv_opt
                )

                assay_yield = st.number_input(
                    f"Assay Yield {i} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key=f"assay_{i}",
                )
                purity = st.number_input(
                    f"Purity {i} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key=f"purity_{i}",
                )

            return {
                "set_index": i,
                "coupling": coupling,
                "temp": float(temp),
                "cond": cond,
                "base": base,
                "time_h": float(time_h),
                "solv": solv,
                "yrts": None if yrts == 0.0 else float(yrts),
                "assay_yield": None if assay_yield == 0.0 else float(assay_yield),
                "purity_pct": None if purity == 0.0 else float(purity),
                "entry_mode": entry_mode,
            }

        sets: List[dict] = []
        for i in range(1, repeats + 1):
            if i > 1:
                st.markdown("---")
            sets.append(condition_set(i))

        st.markdown("---")

        expected_list_raw = st.text_area(
            "Expected compound SMILES (one per line; maps to row n) *",
            help="Line n corresponds to Sₙ/Eₙ/Cₙ. Leave a line blank to set None for that row.",
            key="expected_smiles_list",
        )
        EXP_list = _canonical_list_lines(expected_list_raw.strip())
        rxn_scale_mol = st.number_input(
            "RXN_Scale (mol) *",
            min_value=0.0,
            value=None,
            step=0.001,
            format="%f",
            key="rxn_scale_mol",
        )
        comments = st.text_area(
            "Comments (optional)",
            help="Notes, observations, yields, deviations, etc.",
            key="comments",
        )

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
                S_list = ([canonicalize_smiles(s_smi)] if s_smi else []) + _canonical_list_lines(s_multi.strip())
                E_list = ([canonicalize_smiles(e_smi)] if e_smi else []) + _canonical_list_lines(e_multi.strip())
                C_list = ([canonicalize_smiles(c_smi)] if c_smi else []) + _canonical_list_lines(c_multi.strip())

                if not (S_list or E_list or C_list):
                    st.error("No SEC variants to save.")
                    st.stop()

                n = max(len(S_list), len(E_list), len(C_list))
                if len(EXP_list) != n:
                    st.error(f"Expected list length ({len(EXP_list)}) must match the number of SEC rows ({n}).")
                    st.stop()

                rows = []
                for i in range(n):
                    s_val = S_list[i] if i < len(S_list) else None
                    e_val = E_list[i] if i < len(E_list) else None
                    c_val = C_list[i] if i < len(C_list) else None
                    exp_val = EXP_list[i]
                    if any([s_val, e_val, c_val]):
                        rows.append((s_val, e_val, c_val, exp_val))

                if not rows:
                    st.error("All SEC rows are empty.")
                    st.stop()

                base_idx = next_reaction_index(selected_initials)
                saved, failures = 0, 0

                batch_id = str(uuid.uuid4())

                for offset, (s_val, e_val, c_val, exp_val) in enumerate(rows):
                    rid = format_reaction_id(selected_initials, base_idx + offset)
                    present = sum(1 for x in [s_val, e_val, c_val] if x)
                    oligo = "dimer" if present == 2 else ("trimer" if present == 3 else None)

                    ok = insert_reaction_with_sets(
                        rid=rid,
                        initials=selected_initials,
                        s_smi=s_val,
                        e_smi=e_val,
                        c_smi=c_val,
                        expected_smi=exp_val,
                        rxn_scale_mol=float(rxn_scale_mol),
                        oligomer_type=oligo,
                        sets=sets,
                        comments=comments or None,
                        submission_id=batch_id,
                    )

                    if ok:
                        saved += 1
                    else:
                        failures += 1

                if saved:
                    st.success(
                        f"Saved **{saved}** reaction(s) (positional SEC + Expected) "
                        f"with the same condition set(s)."
                    )
                    st.balloons()
                if failures:
                    st.warning(f"{failures} reaction(s) failed to save. Check data/RLS and try again.")

# ----- TAB 2: Recent Submissions -----
with tab_recent:
    st.subheader("Recent submissions")

    rx_cols = (
        "id,chemist_initials,oligomer_type,"
        "s_block_smiles,e_block_smiles,c_block_smiles,"
        "expected_smiles,rxn_scale_mol,comments,created_at"
    )
    rc_cols = (
        "reaction_id,set_index,coupling_type,temperature_c,condition,base,"
        "time_hours,solvent,yrts,assay_yield,purity_pct,entry_mode"
    )

    rx = sb_request(
        "GET",
        "reactions",
        params={"select": rx_cols, "order": "created_at.desc", "limit": 200},
    ) or []
    rc = sb_request(
        "GET",
        "reaction_conditions",
        params={"select": rc_cols, "order": "reaction_id,set_index"},
    ) or []

    df_rx = pd.DataFrame(rx)
    df_rc = pd.DataFrame(rc)

    if df_rx.empty:
        st.info("No reactions yet. Add your first one in the other tab.")
    else:
        for col in [
            "reaction_id",
            "set_index",
            "coupling_type",
            "temperature_c",
            "condition",
            "base",
            "time_hours",
            "solvent",
        ]:
            if col not in df_rc.columns:
                df_rc[col] = pd.Series(dtype="object")

        df_join = pd.merge(
            df_rx,
            df_rc,
            left_on="id",
            right_on="reaction_id",
            how="left",
            sort=False,
        )

        cols = [
            "id",
            "chemist_initials",
            "oligomer_type",
            "set_index",
            "coupling_type",
            "temperature_c",
            "base",
            "solvent",
            "time_hours",
            "condition",
            "yrts",
            "assay_yield",
            "purity_pct",
            "entry_mode",
            "rxn_scale_mol",
            "comments",
            "created_at",
            "s_block_smiles",
            "e_block_smiles",
            "c_block_smiles",
            "expected_smiles",
        ]
        cols = [c for c in cols if c in df_join.columns]

        df_join = df_join[cols].rename(
            columns={
                "id": "ID",
                "chemist_initials": "Chemist",
                "oligomer_type": "Oligomer",
                "set_index": "Set",
                "coupling_type": "Coupling",
                "temperature_c": "Temp_C",
                "time_hours": "Time_h",
                "yrts": "YRTS",
                "assay_yield": "Assay_Yield_pct",
                "purity_pct": "Purity_pct",
                "entry_mode": "Mode",
                "rxn_scale_mol": "Scale_mol",
                "created_at": "Created_UTC",
                "s_block_smiles": "S_SMILES",
                "e_block_smiles": "E_SMILES",
                "c_block_smiles": "C_SMILES",
                "expected_smiles": "Expected_SMILES",
            }
        )

        st.dataframe(df_join, use_container_width=True, hide_index=True)

        csv_bytes = df_join.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download consolidated_view.csv",
            csv_bytes,
            "consolidated_view.csv",
            "text/csv",
            key="download_consolidated",
        )

        with st.sidebar:
            if st.button("Show download button", key="sidebar_export_trigger"):
                st.download_button(
                    "Download consolidated_view.csv",
                    csv_bytes,
                    "consolidated_view.csv",
                    "text/csv",
                    key="sidebar_consolidated_download",
                )

# =========================================================
# ALWAYS-VISIBLE CONSOLIDATED TABLE (bottom of page)
# =========================================================
st.markdown("---")
st.subheader("All Reactions")

rx_cols_all = (
    "id,chemist_initials,oligomer_type,"
    "s_block_smiles,e_block_smiles,c_block_smiles,"
    "expected_smiles,rxn_scale_mol,comments,created_at"
)
rc_cols_all = (
    "reaction_id,set_index,coupling_type,temperature_c,condition,base,"
    "time_hours,solvent,yrts,assay_yield,purity_pct,entry_mode"
)

rx_all = sb_request(
    "GET",
    "reactions",
    params={"select": rx_cols_all, "order": "created_at.desc", "limit": 200},
) or []

rc_all = sb_request(
    "GET",
    "reaction_conditions",
    params={"select": rc_cols_all, "order": "reaction_id,set_index"},
) or []

df_rx_all = pd.DataFrame(rx_all)
df_rc_all = pd.DataFrame(rc_all)

if df_rx_all.empty:
    st.info("No reactions recorded yet.")
else:
    for col in [
        "reaction_id",
        "set_index",
        "coupling_type",
        "temperature_c",
        "condition",
        "base",
        "time_hours",
        "solvent",
    ]:
        if col not in df_rc_all.columns:
            df_rc_all[col] = pd.Series(dtype="object")

    df_join_all = pd.merge(
        df_rx_all,
        df_rc_all,
        left_on="id",
        right_on="reaction_id",
        how="left",
        sort=False,
    )

    cols_all = [
        "id",
        "chemist_initials",
        "oligomer_type",
        "set_index",
        "coupling_type",
        "temperature_c",
        "base",
        "solvent",
        "time_hours",
        "condition",
        "yrts",
        "assay_yield",
        "purity_pct",
        "entry_mode",
        "rxn_scale_mol",
        "comments",
        "created_at",
        "s_block_smiles",
        "e_block_smiles",
        "c_block_smiles",
        "expected_smiles",
    ]
    cols_all = [c for c in cols_all if c in df_join_all.columns]

    df_view = df_join_all[cols_all].rename(
        columns={
            "id": "ID",
            "chemist_initials": "Chemist",
            "oligomer_type": "Oligomer",
            "set_index": "Set",
            "coupling_type": "Coupling",
            "temperature_c": "Temp_C",
            "time_hours": "Time_h",
            "yrts": "YRTS",
            "assay_yield": "Assay_Yield_pct",
            "purity_pct": "Purity_pct",
            "entry_mode": "Mode",
            "rxn_scale_mol": "Scale_mol",
            "created_at": "Created_UTC",
            "s_block_smiles": "S_SMILES",
            "e_block_smiles": "E_SMILES",
            "c_block_smiles": "C_SMILES",
            "expected_smiles": "Expected_SMILES",
        }
    )

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.download_button(
        "Download all reactions (CSV)",
        df_view.to_csv(index=False).encode("utf-8"),
        "all_reactions_consolidated.csv",
        "text/csv",
        key="download_always_visible",
    )
