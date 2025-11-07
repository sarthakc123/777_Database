# switchphos_independent.py
# Independent create/update; live auto-populating lookups; quick-picks; joined tables.
# Updates:
#  - Live preview on lookup (no buttons) for Phosphine & OAC
#  - YRTS is a percentage (0..100)
#  - OAC quick-pick pushes into Coupling tab

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
        # also update the "last_*" convenience if relevant
        st.session_state[target_key] = v  # sets the value BEFORE widget creation
        return v
    return st.session_state.get(target_key, default_value)

# -------------------------
# UI Shell
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=LOGO_FILE, layout="wide")
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
    st.subheader("OAC Reaction — Independent Create/Update (auto-populate from Phosphine Code)")

    # --- NEW: Quick pick a Phosphine Code (recent first) ---
    with st.expander("Quick pick a Phosphine Code"):
        ph_quick = sb_request(
            "GET",
            "sp_phosphines",
            params={"select": "phos_code,phosphine_name,created_at",
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
                code = df_ph_quick.iloc[ph_idx]["phos_code"]
                st.session_state.last_phos_code = code
                # schedule population of the OAC input on the next run
                push_for_next_run("oac_phos_code_ref", code)
        else:
            st.caption("No phosphines to pick yet.")

    # OAC inputs (note: use consume_pending_or_default for the phos code field)
    oac_code_in = st.text_input(
        "OAC Reaction Code (required)",
        value=st.session_state.get("last_oac_code",""),
        key="oac_code_in"
    )
    phos_code_ref = st.text_input(
        "Phosphine Code (auto-fills name/SMILES)",
        value=consume_pending_or_default("oac_phos_code_ref", st.session_state.get("last_phos_code","")),
        key="oac_phos_code_ref",
    )

    # Live auto-fill from phosphine
    ph_row = None
    if phos_code_ref.strip():
        res = fetch_phosphine(phos_code_ref.strip())
        if res["ok"] and len(res["rows"]) > 0:
            ph_row = res["rows"][0]

    cA, cB = st.columns(2)
    with cA:
        phos_name_prefill = st.text_input(
            "Phosphine Name (auto-filled)",
            value=(ph_row or {}).get("phosphine_name",""),
            key="oac_phos_name_prefill"
        )
    with cB:
        phos_smiles_prefill = st.text_input(
            "Phosphine SMILES (auto-filled)",
            value=(ph_row or {}).get("phosphine_smiles",""),
            key="oac_phos_smiles_prefill"
        )

    oac_smiles     = st.text_input("OAC SMILES", key="oac_smiles_in")
    bromine_name   = st.text_input("Bromine Name", key="oac_bromine_name_in")
    bromine_smiles = st.text_input("Bromine SMILES", key="oac_bromine_smiles_in")
    oac_notes      = st.text_area("Notes (optional)", key="oac_notes_in")

    if st.button("Save OAC Reaction", type="primary", key="save_oac_btn"):
        errs = []
        if not oac_code_in.strip():
            errs.append("OAC Reaction Code is required.")
        if errs:
            for e in errs: st.error(e)
        else:
            payload = {
                "oac_code":     oac_code_in.strip(),
                "phos_code":    phos_code_ref.strip() or None,  # soft link
                "oac_smiles":   canon(oac_smiles or None),
                "bromine_name": bromine_name or None,
                "bromine_smiles": canon(bromine_smiles or None),
                "notes":        oac_notes or None
            }
            ins = sb_request("POST","sp_oac_reactions", json_body=payload)
            if not ins:
                upd = sb_request("PATCH","sp_oac_reactions", json_body=payload, params={"oac_code": f"eq.{oac_code_in.strip()}"})
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
            "GET",
            "sp_oac_with_phosphine",
            params={"select":"oac_code,phos_code,phosphine_name,bromine_name,oac_smiles,created_at",
                    "order":"created_at.desc","limit":200}
        ) or []
        df_quick = pd.DataFrame(oac_quick)
        if not df_quick.empty:
            labels = [
                f"{row['oac_code']} — {row.get('phos_code') or ''} — {row.get('phosphine_name') or ''}"
                for _, row in df_quick.iterrows()
            ]
            idx = st.selectbox("Pick OAC", options=list(range(len(df_quick))), format_func=lambda i: labels[i], key="pick_oac_idx")
            if st.button("Use this OAC in Coupling tab", key="use_oac_btn"):
                code = df_quick.iloc[idx]["oac_code"]
                st.session_state.last_oac_code = code
                push_for_next_run("cpl_oac_code_ref", code)
        else:
            st.caption("No OAC rows to pick yet.")

    # Live context from OAC code as you type
    oac_code_ref2 = st.text_input(
        "OAC Reaction Code (live context)",
        value=consume_pending_or_default("cpl_oac_code_ref", st.session_state.get("last_oac_code", "")),
        key="cpl_oac_code_ref",
    )
    oc_row = None
    if oac_code_ref2.strip():
        oc_res = fetch_oac(oac_code_ref2.strip())
        if oc_res["ok"] and len(oc_res["rows"]) > 0:
            oc_row = oc_res["rows"][0]

    ph_row_cpl = None
    if oc_row and oc_row.get("phos_code"):
        ph_res2 = fetch_phosphine(oc_row["phos_code"])
        if ph_res2["ok"] and len(ph_res2["rows"]) > 0:
            ph_row_cpl = ph_res2["rows"][0]

    cTop1, cTop2 = st.columns(2)
    with cTop1:
        st.text_input("Lookup: Phosphine Code", value=(oc_row or {}).get("phos_code","") or "", disabled=True, key="lookup_phos_code_cpl")
        st.text_input("Lookup: Phosphine Name", value=(ph_row_cpl or {}).get("phosphine_name","") or "", disabled=True, key="lookup_phos_name_cpl")
    with cTop2:
        st.text_input("Lookup: Phosphine SMILES", value=(ph_row_cpl or {}).get("phosphine_smiles","") or "", disabled=True, key="lookup_phos_smiles_cpl")
        st.text_input("Lookup: OAC SMILES", value=(oc_row or {}).get("oac_smiles","") or "", disabled=True, key="lookup_oac_smiles_cpl")

    coupling_type = st.selectbox("Type of reaction", options=COUPLING_TYPES, index=0, key="cpl_type")
    solvent       = st.text_input("Solvent", key="cpl_solvent")
    base_name     = st.text_input("Base name (e.g., BTMG)", key="cpl_base_name")
    base_equiv    = st.number_input("Base equivalence (optional)", min_value=0.0, value=0.0, step=0.1, key="cpl_base_equiv")
    temperature_c = st.number_input("Temperature (°C)", value=25.0, step=0.5, key="cpl_temp_c")
    # YRTS as percentage
    yrts_pct      = st.number_input("YRTS (%)", min_value=0.0, max_value=100.0, step=0.1, key="cpl_yrts_pct")
    assay_yield   = st.number_input("Assay Yield (%)", min_value=0.0, max_value=100.0, step=0.1, key="cpl_assay_yield")
    notes_cr      = st.text_area("Notes (optional)", key="cpl_notes_in")

    if st.button("Save Coupling Result", type="primary", key="save_cpl_btn"):
        errs = []
        if not solvent:
            errs.append("Solvent is required.")
        if temperature_c is None:
            errs.append("Temperature is required.")
        if errs:
            for e in errs: st.error(e)
        else:
            payload = {
                "oac_code":      oac_code_ref2.strip() or None,
                "coupling_type": coupling_type,
                "solvent":       solvent,
                "base_name":     base_name or None,
                "base_equiv":    None if (base_equiv is None or base_equiv == 0.0) else float(base_equiv),
                "temperature_c": float(temperature_c),
                "yrts":          None if yrts_pct is None else float(yrts_pct),  # numeric %
                "assay_yield":   None if assay_yield is None else float(assay_yield),
                "notes":         notes_cr or None
            }
            ins = sb_request("POST","sp_coupling_results", json_body=payload)
            if not ins:
                st.error("Failed to save coupling result.")
            else:
                st.success("Saved coupling result.")
                if oac_code_ref2.strip():
                    st.session_state.last_oac_code = oac_code_ref2.strip()

# -------------------------
# Tables
# -------------------------
st.markdown("---")
st.subheader("Phosphines — `sp_phosphines`")
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
    st.dataframe(df_ph_view, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("OAC (joined with Phosphine) — `sp_oac_with_phosphine`")
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
    st.dataframe(
        df_oacj.rename(columns={
            "oac_code":"OAC_Code","phos_code":"Phos_Code","phosphine_name":"Phosphine_Name",
            "phosphine_smiles":"Phosphine_SMILES","oac_smiles":"OAC_SMILES",
            "bromine_name":"Bromine_Name","bromine_smiles":"Bromine_SMILES",
            "notes":"Notes","created_at":"Created_UTC"
        }),
        use_container_width=True, hide_index=True
    )

st.markdown("---")
st.subheader("Coupling Result (full joined) — `sp_coupling_full`")
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
    def _fmt_base(row):
        b, e = row.get("base_name"), row.get("base_equiv")
        if pd.isna(b) or not b: return None
        try:
            e_val = None if (e is None or (isinstance(e, float) and e == 0.0)) else float(e)
        except Exception:
            e_val = None
        return b if not e_val else f"{b} ({e_val:g}eq)"
    df_cplf["Base_Display"] = df_cplf.apply(_fmt_base, axis=1)

    st.dataframe(
        df_cplf.rename(columns={
            "id":"ID","oac_code":"OAC_Code","coupling_type":"Type","solvent":"Solvent",
            "temperature_c":"Temp_C","yrts":"YRTS_pct","assay_yield":"Assay_Yield_pct",
            "notes":"Notes","created_at":"Created_UTC","phos_code":"Phos_Code",
            "oac_smiles":"OAC_SMILES","bromine_name":"Bromine_Name","bromine_smiles":"Bromine_SMILES",
            "phosphine_name":"Phosphine_Name","phosphine_smiles":"Phosphine_SMILES"
        })[[
            "ID","OAC_Code","Phos_Code","Phosphine_Name","Phosphine_SMILES",
            "OAC_SMILES","Bromine_Name","Bromine_SMILES",
            "Type","Solvent","Base_Display","Temp_C","YRTS_pct","Assay_Yield_pct","Notes","Created_UTC"
        ]],
        use_container_width=True, hide_index=True
    )

# Sidebar exports
with st.sidebar:
    st.markdown("---")
    st.header("Export")
    if not df_ph.empty:
        st.download_button("phosphines.csv", df_ph.to_csv(index=False).encode("utf-8"), "phosphines.csv", "text/csv")
    if not df_oacj.empty:
        st.download_button("oac_joined.csv", df_oacj.to_csv(index=False).encode("utf-8"), "oac_joined.csv", "text/csv")
    if not df_cplf.empty:
        st.download_button("coupling_full.csv", df_cplf.to_csv(index=False).encode("utf-8"), "coupling_full.csv", "text/csv")