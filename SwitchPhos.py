# switchphos_independent.py
# Independent create/update; live auto-populating lookups; quick-picks; joined tables.
# Updates:
#  - Live preview on lookup (no buttons) for Phosphine & OAC
#  - YRTS is a percentage (0..100)
#  - OAC quick-pick pushes into Coupling tab
#  - SIMPLE LOGIN PAGE (first screen) controlled by secrets.auth

import requests
from typing import Optional, Dict, Tuple, Any
import streamlit as st
import pandas as pd

# ---- Optional RDKit for SMILES canonicalization ----
try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

APP_TITLE = "SwitchPhos DB"
LOGO_FILE = "24-MML-3Col (2).jpg"
COUPLING_TYPES = ["Suzuki", "Buchwald", "Other"]

# =========================
# Auth helpers (first page)
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
    """Call set_page_config at most once in this run."""
    if not st.session_state.get("_pc_set"):
        st.set_page_config(page_title=APP_TITLE, page_icon=LOGO_FILE, layout="wide")
        st.session_state["_pc_set"] = True

def require_login_first_page():
    """Show login screen (first page) and stop the script until authenticated."""
    if not _auth_enabled():
        return  # no auth gate

    # Already authenticated?
    if st.session_state.get("auth_user"):
        # Show a small status + logout
        with st.sidebar:
            st.success(f"Signed in as {st.session_state['auth_user']}")
            if st.button("Log out", key="logout_btn"):
                st.session_state.pop("auth_user", None)
        return

    # Not authenticated yet: render the login page and stop
    maybe_set_page_config()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(LOGO_FILE, use_container_width=True)
        st.title("SwitchPhos DB")
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

# Gate the app before anything else renders
require_login_first_page()

# -------------------------
# Supabase thin client
# -------------------------
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
        r = requests.get(f"{url}/rest/v1/sp_phosphines?select=phos_code&limit=1", headers=headers, timeout=30)
        if r.status_code in (200, 206):
            return url, headers
        st.error(f"Supabase connection failed: {r.status_code} - {r.text}")
        st.stop()
    except KeyError as e:
        st.error(f"Missing secret key: {e}. Add to .streamlit/secrets.toml")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        st.stop()

def sb_request(method: str, endpoint: str, json_body: Optional[dict]=None, params: Optional[dict]=None) -> Any:
    url, headers = get_supabase()
    full = f"{url}/rest/v1/{endpoint}"
    hdrs = headers.copy()
    if method.upper() in ("POST", "PATCH", "PUT"):
        hdrs["Prefer"] = "return=representation"
    try:
        r = requests.request(method=method, url=full, headers=hdrs, params=params, json=json_body, timeout=30)
        if r.status_code not in (200, 201, 204, 206):
            st.error(f"API Error {r.status_code} for {method} {endpoint}: {r.text}")
            return None
        if r.status_code == 204 or not r.content:
            return None
        if r.headers.get("content-type","").startswith("application/json"):
            return r.json()
        return r.text
    except Exception as e:
        st.error(f"Request failed for {method} {endpoint}: {e}")
        return None

def sb_get_verbose(endpoint: str, params: Optional[dict]=None) -> Tuple[Optional[int], str, Any]:
    url, headers = get_supabase()
    full = f"{url}/rest/v1/{endpoint}"
    try:
        r = requests.get(full, headers=headers, params=params, timeout=30)
        parsed = None
        if r.content and r.headers.get("content-type","").startswith("application/json"):
            try:
                parsed = r.json()
            except Exception:
                parsed = r.text
        return r.status_code, r.text, parsed
    except Exception as e:
        return None, str(e), None

# -------------------------
# Helpers
# -------------------------
def canon(smi: Optional[str]) -> Optional[str]:
    if not smi:
        return smi
    if RDKit_AVAILABLE:
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                return Chem.MolToSmiles(m)
        except Exception:
            pass
    return smi

def fetch_phosphine(phos_code: str) -> Dict[str, Any]:
    if not phos_code:
        return {"ok": False, "status": None, "rows": [], "msg": "empty code"}
    status, raw, parsed = sb_get_verbose(
        "sp_phosphines",
        params={"select":"phos_code,phosphine_name,phosphine_smiles,notes",
                "phos_code":f"eq.{phos_code.strip()}"}
    )
    rows = parsed if isinstance(parsed, list) else []
    return {"ok": status in (200,206), "status": status, "rows": rows, "msg": raw[:2000]}

def fetch_oac(oac_code: str) -> Dict[str, Any]:
    if not oac_code:
        return {"ok": False, "status": None, "rows": [], "msg": "empty code"}
    status, raw, parsed = sb_get_verbose(
        "sp_oac_reactions",
        params={"select":"oac_code,phos_code,oac_smiles,bromine_name,bromine_smiles,notes",
                "oac_code":f"eq.{oac_code.strip()}"}
    )
    rows = parsed if isinstance(parsed, list) else []
    return {"ok": status in (200,206), "status": status, "rows": rows, "msg": raw[:2000]}

# ---- Safe "push to another input" helpers ----
def push_for_next_run(target_key: str, value):
    """Store a value to be applied to a widget on the next run, then rerun."""
    st.session_state[f"_pending_{target_key}"] = value
    st.rerun()

def consume_pending_or_default(target_key: str, default_value=""):
    """
    If a pending value exists for target_key, consume and return it; else
    return current session_state value or the provided default.
    """
    pend_key = f"_pending_{target_key}"
    if pend_key in st.session_state:
        v = st.session_state.pop(pend_key)
        st.session_state[target_key] = v  # sets the value BEFORE widget creation
        return v
    return st.session_state.get(target_key, default_value)

def on_change_oac_code_autofill():
    """When OAC code changes, fetch row and prefill OAC tab inputs for next run."""
    oac_code = st.session_state.get("oac_code_in","").strip()
    if not oac_code:
        return
    res = fetch_oac(oac_code)
    if not (res and res.get("ok") and len(res.get("rows",[])) > 0):
        return
    row = res["rows"][0]
    if row.get("phos_code") is not None:
        st.session_state["_pending_oac_phos_code_ref"] = row["phos_code"]
        st.session_state["last_phos_code"] = row["phos_code"]
    st.session_state["_pending_oac_smiles_in"]         = row.get("oac_smiles") or ""
    st.session_state["_pending_oac_bromine_name_in"]   = row.get("bromine_name") or ""
    st.session_state["_pending_oac_bromine_smiles_in"] = row.get("bromine_smiles") or ""
    st.session_state["_pending_oac_notes_in"]          = row.get("notes") or ""
    st.rerun()

def fetch_oac_joined(oac_code: str):
    """Fetch OAC row joined with phosphine info (from view)."""
    if not oac_code:
        return {"ok": False, "status": None, "rows": [], "msg": "empty code"}
    status, raw, parsed = sb_get_verbose(
        "sp_oac_with_phosphine",
        params={
            "select": "oac_code,phos_code,phosphine_name,phosphine_smiles,oac_smiles,bromine_name,bromine_smiles,notes",
            "oac_code": f"eq.{oac_code.strip()}",
            "limit": 1
        }
    )
    rows = parsed if isinstance(parsed, list) else []
    return {"ok": status in (200, 206), "status": status, "rows": rows, "msg": raw[:2000]}

def load_oac_into_form(oac_code: str):
    """Load an OAC (joined) row and schedule ALL OAC form fields to prefill next run."""
    res = fetch_oac_joined(oac_code)
    if not (res["ok"] and res["rows"]):
        st.warning("No OAC found for that code.")
        return
    row = res["rows"][0]
    st.session_state["_pending_oac_code_in"]           = row.get("oac_code","")
    st.session_state["_pending_oac_phos_code_ref"]     = row.get("phos_code","") or ""
    st.session_state["_pending_oac_phos_name_prefill"] = row.get("phosphine_name","") or ""
    st.session_state["_pending_oac_phos_smiles_prefill"] = row.get("phosphine_smiles","") or ""
    st.session_state["_pending_oac_smiles_in"]         = row.get("oac_smiles","") or ""
    st.session_state["_pending_oac_bromine_name_in"]   = row.get("bromine_name","") or ""
    st.session_state["_pending_oac_bromine_smiles_in"] = row.get("bromine_smiles","") or ""
    st.session_state["_pending_oac_notes_in"]          = row.get("notes","") or ""
    st.rerun()

# -------------------------
# UI Shell
# -------------------------
maybe_set_page_config()  # <- safe version
hdr_l, hdr_r = st.columns([4, 1.2])
with hdr_l:
    st.title(APP_TITLE)
with hdr_r:
    try: st.image("IMG_0405.jpeg", use_container_width=True)
    except Exception: pass

get_supabase()

# Session convenience
st.session_state.setdefault("last_phos_code", "")
st.session_state.setdefault("last_oac_code", "")

# -------------------------
# Tabs
# -------------------------
tab_phos, tab_oac, tab_coup = st.tabs(["Phosphine", "OAC Reaction", "Coupling Result"])

# ===== PHOSPHINE =====
with tab_phos:
    st.subheader("Phosphine")
    phos_code_in = st.text_input("Phosphine Code (required)", value=st.session_state.get("last_phos_code",""), key="phos_code_in")
    phos_name    = st.text_input("Phosphine Name", key="phos_name_in")
    phos_smiles  = st.text_input("Phosphine SMILES", key="phos_smiles_in")
    phos_notes   = st.text_area("Notes (optional)", key="phos_notes_in")

    if st.button("Save Phosphine", type="primary", key="save_phos_btn"):
        errs = []
        if not phos_code_in.strip():
            errs.append("Phosphine Code is required.")
        if not (phos_name or phos_smiles):
            errs.append("Enter at least a Name or a SMILES.")
        if errs:
            for e in errs: st.error(e)
        else:
            payload = {
                "phos_code": phos_code_in.strip(),
                "phosphine_name": phos_name or None,
                "phosphine_smiles": canon(phos_smiles or None),
                "notes": phos_notes or None
            }
            ins = sb_request("POST","sp_phosphines", json_body=payload)
            if not ins:
                upd = sb_request("PATCH","sp_phosphines", json_body=payload, params={"phos_code": f"eq.{phos_code_in.strip()}"})
                if not upd:
                    st.error("Failed to save phosphine.")
                else:
                    st.success(f"Updated phosphine {phos_code_in.strip()}.")
            else:
                st.success(f"Saved phosphine {phos_code_in.strip()}.")
            st.session_state.last_phos_code = phos_code_in.strip()

    st.divider()
    st.markdown("**Live Preview by Code**")
    phos_lookup_code = st.text_input("Type a Phosphine Code", value=st.session_state.last_phos_code, key="phos_lookup_in")
    if phos_lookup_code.strip():
        res = fetch_phosphine(phos_lookup_code.strip())
        if res["ok"] and len(res["rows"]) > 0:
            st.dataframe(pd.DataFrame([res["rows"][0]]), use_container_width=True, hide_index=True)
        elif res["ok"]:
            st.caption("No phosphine found.")
        else:
            st.caption(f"Lookup error (status {res['status']}).")

# ===== OAC =====
with tab_oac:
    st.subheader("OAC Reaction")

    with st.expander("Quick pick a Phosphine Code"):
        ph_quick = sb_request(
            "GET",
            "sp_phosphines",
            params={"select": "phos_code,phosphine_name,phosphine_smiles,created_at",
                    "order": "created_at.desc", "limit": 200}
        ) or []
        df_ph_quick = pd.DataFrame(ph_quick)

        if not df_ph_quick.empty:
            ph_labels = [
                f"{row['phos_code']} — {row.get('phosphine_name') or ''}"
                for _, row in df_ph_quick.iterrows()
            ]
            ph_idx = st.selectbox(
                "Pick Phosphine",
                options=list(range(len(df_ph_quick))),
                format_func=lambda i: ph_labels[i],
                key="pick_phos_idx_for_oac"
            )
            if st.button("Use this Phosphine in OAC tab", key="use_phos_for_oac_btn"):
                row = df_ph_quick.iloc[ph_idx]
                code = row["phos_code"]
                name = row.get("phosphine_name") or ""
                smi = row.get("phosphine_smiles") or ""
                st.session_state.last_phos_code = code
                st.session_state["_pending_oac_phos_code_ref"] = code
                st.session_state["_pending_oac_phos_name_prefill"] = name
                st.session_state["_pending_oac_phos_smiles_prefill"] = smi
                st.rerun()
        else:
            st.caption("No phosphines to pick yet.")

        # OAC code input — attach on_change callback
        oac_code_in = st.text_input(
            "OAC Reaction Code (required)",
            value=consume_pending_or_default("oac_code_in", st.session_state.get("last_oac_code", "")),
            key="oac_code_in",
            on_change=on_change_oac_code_autofill,
        )

        # Phosphine Code (can be filled either by user, quick pick, or by OAC fetch)
        phos_code_ref = st.text_input(
            "Phosphine Code (auto-fills name/SMILES)",
            value=consume_pending_or_default("oac_phos_code_ref", st.session_state.get("last_phos_code", "")),
            key="oac_phos_code_ref",
        )

        # Live auto-fill for phosphine display (as you type)
        ph_row = None
        if phos_code_ref.strip():
            res = fetch_phosphine(phos_code_ref.strip())
            if res["ok"] and len(res["rows"]) > 0:
                ph_row = res["rows"][0]

        cA, cB = st.columns(2)
        with cA:
            phos_name_prefill = st.text_input(
                "Phosphine Name (auto-filled)",
                value=consume_pending_or_default(
                    "oac_phos_name_prefill",
                    (ph_row or {}).get("phosphine_name", "")
                ),
                key="oac_phos_name_prefill",
            )
        with cB:
            phos_smiles_prefill = st.text_input(
                "Phosphine SMILES (auto-filled)",
                value=consume_pending_or_default(
                    "oac_phos_smiles_prefill",
                    (ph_row or {}).get("phosphine_smiles", "")
                ),
                key="oac_phos_smiles_prefill",
            )

        # These now “consume pending” values that the OAC code on_change callback sets
        oac_smiles = st.text_input(
            "OAC SMILES",
            value=consume_pending_or_default("oac_smiles_in", ""),
            key="oac_smiles_in",
        )
        bromine_name = st.text_input(
            "Bromine Name",
            value=consume_pending_or_default("oac_bromine_name_in", ""),
            key="oac_bromine_name_in",
        )
        bromine_smiles = st.text_input(
            "Bromine SMILES",
            value=consume_pending_or_default("oac_bromine_smiles_in", ""),
            key="oac_bromine_smiles_in",
        )
        oac_notes = st.text_area(
            "Notes (optional)",
            value=consume_pending_or_default("oac_notes_in", ""),
            key="oac_notes_in",
        )

        if st.button("Save OAC Reaction", type="primary", key="save_oac_btn"):
            errs = []
            if not oac_code_in.strip():
                errs.append("OAC Reaction Code is required.")
            if errs:
                for e in errs: st.error(e)
            else:
                payload = {
                    "oac_code": oac_code_in.strip(),
                    "phos_code": phos_code_ref.strip() or None,
                    "oac_smiles": canon(oac_smiles or None),
                    "bromine_name": bromine_name or None,
                    "bromine_smiles": canon(bromine_smiles or None),
                    "notes": oac_notes or None
                }
                ins = sb_request("POST", "sp_oac_reactions", json_body=payload)
                if not ins:
                    upd = sb_request("PATCH", "sp_oac_reactions", json_body=payload,
                                     params={"oac_code": f"eq.{oac_code_in.strip()}"})
                    if not upd:
                        st.error("Failed to save OAC reaction.")
                    else:
                        st.success(f"Updated OAC {oac_code_in.strip()}.")
                else:
                    st.success(f"Saved OAC {oac_code_in.strip()}.")
                st.session_state.last_oac_code = oac_code_in.strip()
                if phos_code_ref.strip():
                    st.session_state.last_phos_code = phos_code_ref.strip()

# ===== COUPLING =====
with tab_coup:
    st.subheader("Coupling Result")

    # Quick pick for OAC codes (from joined view, recent first)
    with st.expander("Quick pick an OAC Code"):
        oac_quick = sb_request(
            "GET", "sp_oac_with_phosphine",
            params={
                "select": "oac_code,phos_code,phosphine_name,oac_smiles,created_at",
                "order": "created_at.desc", "limit": 200
            }
        ) or []
        df_quick = pd.DataFrame(oac_quick)
        if not df_quick.empty:
            labels = [
                f"{row['oac_code']} — {row.get('phos_code') or ''} — {row.get('phosphine_name') or ''}"
                for _, row in df_quick.iterrows()
            ]
            idx = st.selectbox("Pick OAC", options=list(range(len(df_quick))),
                               format_func=lambda i: labels[i], key="pick_oac_idx")
            if st.button("Use this OAC in Coupling tab", key="use_oac_btn"):
                code = df_quick.iloc[idx]["oac_code"]
                st.session_state.last_oac_code = code
                push_for_next_run("cpl_oac_code_ref", code)
        else:
            st.caption("No OAC rows to pick yet.")

    # Live OAC code (typing or quick-pick both land here)
    oac_code_ref2 = st.text_input(
        "OAC Reaction Code (live context)",
        value=consume_pending_or_default("cpl_oac_code_ref", st.session_state.get("last_oac_code", "")),
        key="cpl_oac_code_ref",
    ).strip()

    # Pull joined context (includes phos name/SMILES)
    ocj_row = None
    if oac_code_ref2:
        res = fetch_oac_joined(oac_code_ref2)
        if res["ok"] and res["rows"]:
            ocj_row = res["rows"][0]

    # Auto-populating LOOKUP fields (read-only)
    dyn_suffix = oac_code_ref2 or "blank"
    cTop1, cTop2 = st.columns(2)
    with cTop1:
        st.text_input(
            "Lookup: Phosphine Code",
            value=(ocj_row or {}).get("phos_code", "") or "",
            disabled=True,
            key=f"lookup_phos_code_cpl::{dyn_suffix}",
        )
        st.text_input(
            "Lookup: Phosphine Name",
            value=(ocj_row or {}).get("phosphine_name", "") or "",
            disabled=True,
            key=f"lookup_phos_name_cpl::{dyn_suffix}",
        )
    with cTop2:
        st.text_input(
            "Lookup: Phosphine SMILES",
            value=(ocj_row or {}).get("phosphine_smiles", "") or "",
            disabled=True,
            key=f"lookup_phos_smiles_cpl::{dyn_suffix}",
        )
        st.text_input(
            "Lookup: OAC SMILES",
            value=(ocj_row or {}).get("oac_smiles", "") or "",
            disabled=True,
            key=f"lookup_oac_smiles_cpl::{dyn_suffix}",
        )

    # Editable Coupling inputs
    coupling_type = st.selectbox("Type of reaction", options=COUPLING_TYPES, index=0, key="cpl_type")
    solvent       = st.text_input("Solvent", key="cpl_solvent")
    base_name     = st.text_input("Base name (e.g., BTMG)", key="cpl_base_name")
    base_equiv    = st.number_input("Base equivalence (optional)", min_value=0.0, value=0.0, step=0.1, key="cpl_base_equiv")
    temperature_c = st.number_input("Temperature (°C)", value=25.0, step=0.5, key="cpl_temp_c")
    yrts_pct      = st.number_input("YRTS (%)", min_value=0.0, max_value=100.0, step=0.1, key="cpl_yrts_pct")
    assay_yield   = st.number_input("Assay Yield (%)", min_value=0.0, max_value=100.0, step=0.1, key="cpl_assay_yield")
    notes_cr      = st.text_area("Notes (optional)", key="cpl_notes_in")

    if st.button("Save Coupling Result", type="primary", key="save_cpl_btn"):
        errs = []
        if not solvent: errs.append("Solvent is required.")
        if temperature_c is None: errs.append("Temperature is required.")
        if errs:
            for e in errs: st.error(e)
        else:
            payload = {
                "oac_code":      oac_code_ref2 or None,
                "coupling_type": coupling_type,
                "solvent":       solvent,
                "base_name":     base_name or None,
                "base_equiv":    None if (base_equiv is None or base_equiv == 0.0) else float(base_equiv),
                "temperature_c": float(temperature_c),
                "yrts":          None if yrts_pct is None else float(yrts_pct),
                "assay_yield":   None if assay_yield is None else float(assay_yield),
                "notes":         notes_cr or None,
            }
            ins = sb_request("POST", "sp_coupling_results", json_body=payload)
            if not ins:
                st.error("Failed to save coupling result.")
            else:
                st.success("Saved coupling result.")
                if oac_code_ref2:
                    st.session_state.last_oac_code = oac_code_ref2

# -------------------------
# Tables (MASKED DISPLAY)
# -------------------------
st.markdown("---")
st.subheader("Phosphines")
ph_rows = sb_request(
    "GET",
    "sp_phosphines",
    params={"select":"phos_code,phosphine_name,phosphine_smiles,notes,created_at",
            "order":"created_at.desc","limit":200}
) or []
df_ph = pd.DataFrame(ph_rows)

if df_ph.empty:
    st.info("No phosphines yet.")
else:
    df_ph_view = df_ph.rename(columns={
        "phos_code":"Phos_Code",
        "phosphine_name":"Phosphine_Name",
        "phosphine_smiles":"Phosphine_SMILES",
        "notes":"Notes",
        "created_at":"Created_UTC"
    })
    st.dataframe(df_ph_view[["Phos_Code"]], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("OAC (joined with Phosphine)")
oac_joined = sb_request(
    "GET",
    "sp_oac_with_phosphine",
    params={"select":"oac_code,phos_code,phosphine_name,phosphine_smiles,oac_smiles,bromine_name,bromine_smiles,notes,created_at",
            "order":"created_at.desc","limit":200}
) or []
df_oacj = pd.DataFrame(oac_joined)

if df_oacj.empty:
    st.info("No OAC rows yet.")
else:
    df_oacj_view = df_oacj.rename(columns={
        "oac_code":"OAC_Code",
        "phos_code":"Phos_Code",
        "phosphine_name":"Phosphine_Name",
        "phosphine_smiles":"Phosphine_SMILES",
        "oac_smiles":"OAC_SMILES",
        "bromine_name":"Bromine_Name",
        "bromine_smiles":"Bromine_SMILES",
        "notes":"Notes",
        "created_at":"Created_UTC"
    })
    st.dataframe(df_oacj_view[["OAC_Code","Phos_Code"]], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Coupling Result (full joined)")
cpl_full = sb_request(
    "GET",
    "sp_coupling_full",
    params={"select":"id,oac_code,coupling_type,solvent,base_name,base_equiv,temperature_c,yrts,assay_yield,notes,created_at,phos_code,oac_smiles,bromine_name,bromine_smiles,phosphine_name,phosphine_smiles",
            "order":"created_at.desc","limit":200}
) or []
df_cplf = pd.DataFrame(cpl_full)

if df_cplf.empty:
    st.info("No coupling results yet.")
else:
    df_cplf_view = df_cplf.rename(columns={
        "id":"ID",
        "oac_code":"OAC_Code",
        "phos_code":"Phos_Code",
        "coupling_type":"Type",
        "solvent":"Solvent",
        "temperature_c":"Temp_C",
        "yrts":"YRTS_pct",
        "assay_yield":"Assay_Yield_pct",
        "notes":"Notes",
        "created_at":"Created_UTC",
        "oac_smiles":"OAC_SMILES",
        "bromine_name":"Bromine_Name",
        "bromine_smiles":"Bromine_SMILES",
        "phosphine_name":"Phosphine_Name",
        "phosphine_smiles":"Phosphine_SMILES"
    })
    st.dataframe(df_cplf_view[["OAC_Code","Phos_Code"]],
                 use_container_width=True, hide_index=True)

# -------------------------
# Sidebar exports (FULL, unmasked)
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Export")
    if not df_ph.empty:
        st.download_button("phosphines.csv", df_ph.to_csv(index=False).encode("utf-8"), "phosphines.csv", "text/csv")
    if not df_oacj.empty:
        st.download_button("oac_joined.csv", df_oacj.to_csv(index=False).encode("utf-8"), "oac_joined.csv", "text/csv")
    if not df_cplf.empty:
        st.download_button("coupling_full.csv", df_cplf.to_csv(index=False).encode("utf-8"), "coupling_full.csv", "text/csv")