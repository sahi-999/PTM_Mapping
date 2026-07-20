import streamlit as st
import pandas as pd
import requests
import tempfile
import os
import json
import re
from typing import List, Tuple
from bs4 import BeautifulSoup
from Bio import SeqIO
import streamlit.components.v1 as components

# ====================== PAGE SETUP ======================
st.set_page_config(page_title="RCSB Proteomics Engine Pro", layout="wide")
st.title("🧬 Bi-Directional Parallel Mapping Workspace")
st.markdown("""
### 🔄 Complete Parallel Coordination Achieved:
1. **1D Sequence ➔ 3D Structure**: Hover or click a letter in the grid to highlight its exact spatial coordinate sphere and update the dashboard.
2. **3D Structure ➔ 1D Sequence**: Hover or click directly on any ribbon or atom in the 3D viewport. NGL captures the raycast event, **scrolls the 1D sequence grid directly to that exact residue, and highlights it** instantly.
3. **Peptide Selection**: Click any detected peptide (blue region) in the 3D structure to **zoom into it** in the inset panel, ghost the rest, and grey out the 1D sequence outside that peptide.

> ⚠️ **Note on PTM rendering**: AlphaFold structures model the *unmodified* protein — the atoms of a phosphate/acetyl/etc. group don't physically exist in the file. PTMs are rendered as a chemically-accurate **hyperball** (CPK-style) model at the real modification site on the existing residue atoms, colored and labeled using data fetched live from each modification's own UNIMOD record page.
""")

# ====================== FILE UPLOADS ======================
col1, col2 = st.columns(2)
with col1:
    peptide_file = st.file_uploader("Upload **peptide.csv**", type=["csv"])
with col2:
    fasta_file = st.file_uploader("Upload **FASTA Sequence** (Optional)", type=["fasta", "fa"])

if not peptide_file:
    st.info("👆 Please upload your peptide.csv file to begin.")
    st.stop()

# ====================== DATA PROCESSING ======================
@st.cache_data
def load_peptides(f):
    return pd.read_csv(f)

peptides = load_peptides(peptide_file)

def find_col(df, candidates):
    for cand in candidates:
        for col in df.columns:
            if cand.lower().replace(" ", "").replace(".", "") in col.lower().replace(" ", "").replace(".", ""):
                return col
    return None

protein_group_col = find_col(peptides, ["protein.group", "protein group", "protein"])
stripped_col      = find_col(peptides, ["stripped.sequence", "stripped sequence"])
modified_col      = find_col(peptides, ["modified.sequence", "modified sequence"])

if protein_group_col:
    selected_protein = st.sidebar.selectbox("Select Target Protein", sorted(peptides[protein_group_col].unique()))
else:
    selected_protein = st.sidebar.text_input("Enter Protein ID", value="P10636")

protein_df = peptides[peptides[protein_group_col] == selected_protein].copy() if protein_group_col else peptides

# ====================== DETECT PTM IDS IN THIS PROTEIN ======================
found_unimod_ids = set()
if modified_col:
    for _, row in protein_df.iterrows():
        mod_sequence_str = str(row.get(modified_col, ""))
        found_unimod_ids.update(
            m.lower() for m in re.findall(r"unimod:\d+", mod_sequence_str, re.IGNORECASE)
        )

# ====================== LIVE UNIMOD RECORD-PAGE FETCHING ======================
# Pulls identification directly from each modification's own record page:
#   https://www.unimod.org/modifications_view.php?editid1=<accession>
# instead of the bulk list API. Cached per-accession so each PTM is only
# fetched once per session, even across sidebar re-renders.

def _parse_unimod_view_page(html: str) -> dict:
    """Flatten the record-page's <td> label/value pairs into a dict."""
    soup = BeautifulSoup(html, "html.parser")
    cells = [td.get_text(strip=True) for td in soup.find_all("td")]

    def value_after(label):
        for i, c in enumerate(cells):
            if c == label and i + 1 < len(cells):
                return cells[i + 1]
        return ""

    psi_name      = value_after("PSI-MS Name")
    interim_name  = value_after("Interim Name")
    description   = value_after("Description")
    composition   = value_after("Composition")
    mono_mass_str = value_after("Monoisotopic")

    try:
        mono_mass = float(mono_mass_str)
    except ValueError:
        mono_mass = 0.0

    name = psi_name or interim_name or description or "Unknown"

    return {
        "name": name,
        "description": description,
        "mono_mass": mono_mass,
        "composition": composition,
    }


@st.cache_data(show_spinner=False)
def fetch_unimod_view_page(accession: str):
    """Fetch + parse a single UNIMOD modification record page by accession number."""
    url = f"https://www.unimod.org/modifications_view.php?editid1={accession}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text:
            return None
        parsed = _parse_unimod_view_page(resp.text)
        if not parsed["name"] or parsed["name"] == "Unknown":
            return None
        parsed["accession"]  = accession
        parsed["source_url"] = url
        return parsed
    except Exception:
        return None


def _auto_color(composition: str, mass: float, classification: str) -> str:
    """Derive a default hex color suggestion from chemical composition."""
    comp = (composition or "").upper()

    if "P" in comp:               return "#ef4444"   # phosphorus → red (phosphorylation)
    if classification == "Artefact": return "#6b7280" # grey for lab artefacts
    if "S" in comp and mass > 50: return "#a855f7"   # sulfur + heavy → purple (oxidation, alkylation)
    if "S" in comp:                return "#6b7280"  # sulfur light → grey (carbamidomethyl)
    if mass < 0:                   return "#0ea5e9"  # mass loss → blue (deamidation loss)
    if mass < 20:                  return "#10b981"  # tiny addition → green (methylation)
    if mass < 60:                  return "#eab308"  # medium → yellow (acetylation)
    if "N" in comp and "O" in comp: return "#f97316" # nitrogen+oxygen → orange (nitration)
    return "#94a3b8"                                 # default slate


@st.cache_data(show_spinner=False)
def fetch_unimod_data(unimod_ids: tuple) -> dict:
    """
    Build the enrichment dict for every detected UniMod ID by fetching each
    modification's own record page directly (real identification, not the
    bulk list endpoint).
    Returns: { "unimod:21": { name, mono_mass, composition, color, repr_type, source_url }, ... }
    """
    result = {}
    for uid in unimod_ids:
        accession = uid.split(":")[-1].strip()
        parsed = fetch_unimod_view_page(accession)
        if not parsed:
            continue
        comp = parsed["composition"]
        mass = parsed["mono_mass"]
        result[uid] = {
            "name":         parsed["name"],
            "mono_mass":    mass,
            "composition":  comp,
            "color":        _auto_color(comp, mass, ""),
            "repr_type":    "hyperball",   # chemically-accurate CPK-style model
            "source_url":   parsed["source_url"],
        }
    return result


unimod_enriched = fetch_unimod_data(tuple(sorted(found_unimod_ids)))

# ====================== PTM CUSTOMIZATION PANEL ======================
st.sidebar.subheader("🎨 PTM Color & Label Settings")

if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}

if found_unimod_ids:
    id_list = ", ".join(uid.split(':')[-1] for uid in sorted(found_unimod_ids))
    st.sidebar.write(f"**Detected UniMod IDs:** {id_list}")
else:
    st.sidebar.info("No PTMs detected in this protein.")

# Initialize / refresh default configs for detected PTMs using live UNIMOD data
for uid in found_unimod_ids:
    enriched = unimod_enriched.get(uid, {})
    auto_label = enriched.get("name", uid.upper())
    auto_color = enriched.get("color", "#436da9")

    if uid not in st.session_state.ptm_configs:
        st.session_state.ptm_configs[uid] = {
            'selected':   True,
            'label':      auto_label,
            'color':      auto_color,
            'auto_color': auto_color,   # remembered so "reset" always has somewhere to go back to
        }
    else:
        # keep the auto-color reference current even if UNIMOD data refreshes
        st.session_state.ptm_configs[uid]['auto_color'] = auto_color

# User controls for each PTM
for uid in sorted(found_unimod_ids):
    enriched = unimod_enriched.get(uid, {})
    cfg = st.session_state.ptm_configs[uid]

    cols = st.sidebar.columns([1, 1.3, 1, 0.6])
    with cols[0]:
        cfg['selected'] = st.checkbox(
            f"UniMod:{uid.split(':')[-1]}",
            value=cfg['selected'],
            key=f"cb_{uid}"
        )
    with cols[1]:
        cfg['label'] = st.text_input(
            "Label",
            value=cfg['label'],
            key=f"lbl_{uid}",
            label_visibility="collapsed"
        )
    with cols[2]:
        cfg['color'] = st.color_picker(
            "Color",
            value=cfg['color'],
            key=f"col_{uid}",
            label_visibility="collapsed"
        )
    with cols[3]:
        if st.button("↺", key=f"reset_{uid}", help="Reset to UNIMOD-derived auto-color"):
            st.session_state.ptm_configs[uid]['color'] = cfg['auto_color']
            st.rerun()

    if enriched:
        st.sidebar.caption(
            f"🔗 {enriched.get('name','?')} · {enriched.get('composition','?')} · "
            f"{enriched.get('mono_mass',0):.4f} Da · "
            f"[UNIMOD record]({enriched.get('source_url','https://www.unimod.org')})"
        )

# ====================== FASTA / API SEQUENCE RETRIEVAL ======================
full_sequence = ""
pdb_url       = ""

if fasta_file:
    fasta_bytes = fasta_file.getvalue().decode("utf-8")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w") as tmp:
        tmp.write(fasta_bytes)
        tmp_path = tmp.name
    try:
        for record in SeqIO.parse(tmp_path, "fasta"):
            seq_id = record.id.split('|')[-1] if '|' in record.id else record.id
            if selected_protein in seq_id or seq_id in selected_protein:
                full_sequence = str(record.seq).upper()
                break
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@st.cache_data
def fetch_alphafold_fallback(uniprot_id):
    clean_id = uniprot_id.split(';')[0].split('-')[0].strip()
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{clean_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return data.get("pdbUrl"), data.get("uniprotSequence")
    except Exception:
        pass
    return None, None

af_pdb_url, af_sequence = fetch_alphafold_fallback(selected_protein)
pdb_url = af_pdb_url

if not full_sequence:
    full_sequence = af_sequence

if not pdb_url or not full_sequence:
    st.error(f"Could not automatically locate structural data for: `{selected_protein}`")
    st.stop()

# ====================== PARSE pLDDT CONFIDENCE SCORES ======================
@st.cache_data
def extract_plddt_values(url_path, target_seq_len):
    scores = [80.0] * target_seq_len
    try:
        res = requests.get(url_path)
        if res.status_code == 200:
            lines = res.text.split('\n')
            for line in lines:
                if line.startswith("ATOM  ") and " CA " in line:
                    res_num = int(line[22:26].strip())
                    plddt   = float(line[60:66].strip())
                    if 0 < res_num <= target_seq_len:
                        scores[res_num - 1] = plddt
    except Exception:
        pass
    return scores

plddt_scores = extract_plddt_values(pdb_url, len(full_sequence))

# ====================== MAP PTMS ONTO THE SEQUENCE ======================
is_detected_residue = [False] * len(full_sequence)
ptm_metadata_map    = [None]  * len(full_sequence)

if stripped_col and modified_col:
    for _, row in protein_df.iterrows():
        pep = str(row[stripped_col]).upper().strip()
        mod_sequence_str = str(row[modified_col])

        start_idx = full_sequence.find(pep)
        while start_idx != -1:
            # Mark detected residues
            for i in range(start_idx, start_idx + len(pep)):
                is_detected_residue[i] = True

            # Parse PTMs with user custom color + label
            clean_mod_str = mod_sequence_str
            current_pep_idx = 0
            while clean_mod_str:
                match = re.match(r"^([A-Z])\((unimod:\d+)\)", clean_mod_str, re.IGNORECASE)
                if match:
                    aa, unimod_id = match.groups()
                    uid_lower = unimod_id.lower()

                    # Get user config or fallback
                    config = st.session_state.ptm_configs.get(uid_lower, {
                        'selected': True,
                        'label': uid_lower.upper(),
                        'color': "#436da9",
                    })

                    if config['selected']:
                        global_site = start_idx + current_pep_idx
                        if global_site < len(full_sequence):
                            enriched = unimod_enriched.get(uid_lower, {})
                            ptm_metadata_map[global_site] = {
                                "id":          uid_lower.upper(),
                                "name":        config['label'],                       # user custom label
                                "color":       config['color'],                       # user chosen color
                                "repr_type":   enriched.get("repr_type", "hyperball"), # chemically-accurate model
                                "mass":        enriched.get("mono_mass", 0.0),
                                "composition": enriched.get("composition", ""),
                                "source_url":  enriched.get("source_url", ""),
                            }
                    clean_mod_str = clean_mod_str[match.end():]
                    current_pep_idx += 1
                elif clean_mod_str and clean_mod_str[0].isalpha():
                    clean_mod_str = clean_mod_str[1:]
                    current_pep_idx += 1
                else:
                    clean_mod_str = clean_mod_str[1:]
            start_idx = full_sequence.find(pep, start_idx + 1)

# ====================== SIDEBAR ======================
st.sidebar.subheader("Visualization Modifiers")
enable_surface  = st.sidebar.toggle("Render Solvent Accessible Surface", value=False)
surface_opacity = st.sidebar.slider("Surface Opacity", 0.0, 1.0, 0.3) if enable_surface else 0.0

# ====================== BACKBONE RENDERING STYLE TOGGLE ======================
# Lets the user switch how the protein *backbone/chain* itself is drawn —
# either the classic ribbon (cartoon) representation, or the same
# chemically-accurate CPK/hyperball atomic style used for PTM sites.
# This only affects the backbone; PTM markers always keep their own
# hyperball representation regardless of this setting.
st.sidebar.subheader("🧬 Backbone Rendering Style")
backbone_style_choice = st.sidebar.radio(
    "Protein backbone representation",
    options=["🎀 Ribbon (Cartoon)", "⚛️ CPK / Hyperball (atomic)"],
    index=0,
    help=(
        "Switch how the protein backbone/chain is rendered. PTM sites are "
        "unaffected — they always render as chemically-accurate hyperball "
        "models regardless of this setting."
    ),
)
backbone_style = "cpk" if backbone_style_choice.startswith("⚛️") else "ribbon"

# ====================== MANUAL 3D REGION SELECTOR ======================
# Lets the user zoom into ANY residue range, not just auto-detected peptides.
st.sidebar.subheader("🎯 Manual Structure Region Selector")
st.sidebar.caption("Pick any residue range in the structure to zoom into — independent of detected peptides.")

if "manual_zoom" not in st.session_state:
    st.session_state.manual_zoom = None

seq_len = len(full_sequence)
default_start = st.session_state.manual_zoom["start"] if st.session_state.manual_zoom else 1
default_end   = st.session_state.manual_zoom["end"] if st.session_state.manual_zoom else min(20, seq_len)

msel_cols = st.sidebar.columns(2)
with msel_cols[0]:
    manual_start = st.number_input(
        "Start residue", min_value=1, max_value=seq_len,
        value=min(default_start, seq_len), key="manual_start_input"
    )
with msel_cols[1]:
    manual_end = st.number_input(
        "End residue", min_value=1, max_value=seq_len,
        value=min(default_end, seq_len), key="manual_end_input"
    )

zbtn_cols = st.sidebar.columns(2)
with zbtn_cols[0]:
    if st.button("🔍 Zoom to Range", use_container_width=True):
        lo, hi = sorted([int(manual_start), int(manual_end)])
        st.session_state.manual_zoom = {"start": lo, "end": hi}
        st.rerun()
with zbtn_cols[1]:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state.manual_zoom = None
        st.rerun()

# ====================== UNIFIED ENGINE HTML ======================
js_data = {
    "fullSeq": full_sequence,
    "detectionMap": is_detected_residue,
    "ptmMap": ptm_metadata_map,
    "plddtMap": plddt_scores,
    "PDB_URL": pdb_url or "",
    "ENABLE_SURF": bool(enable_surface),
    "SURF_OPACITY": float(surface_opacity),
    "backboneStyle": backbone_style,                # "ribbon" or "cpk"
    "manualSelection": st.session_state.manual_zoom  # {start, end} or null
}

custom_viewer_html = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  #root-layout {
    display: flex;
    flex-direction: column;
    gap: 16px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  #panels-row {
    display: flex;
    gap: 16px;
    width: 100%;
  }

  .panel-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }

  .panel-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    padding-left: 4px;
  }

  .viewport {
    width: 100%;
    height: 480px;
    background-color: #0b0f19;
    border: 1px solid #1e293b;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.5);
    position: relative;
    overflow: hidden;
  }

  #zoom-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #334155;
    font-size: 14px;
    gap: 10px;
    pointer-events: none;
    z-index: 5;
  }
  #zoom-placeholder svg { opacity: 0.4; }

  #hover-card {
    position: absolute;
    bottom: 16px;
    left: 16px;
    right: 16px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    color: #f8fafc;
    font-size: 13px;
    display: none;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3);
    z-index: 1000;
  }
  #hover-card .card-top {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
  }
  #hover-card .card-ptm-detail {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
  }
  #hover-card .card-ptm-detail a {
    color: #60a5fa;
    text-decoration: none;
  }

  #selection-badge {
    display: none;
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(244,63,94,0.18);
    border: 1px solid #f43f5e;
    border-radius: 6px;
    padding: 6px 10px;
    color: #fda4af;
    font-size: 12px;
    font-weight: 600;
    z-index: 999;
    cursor: pointer;
    user-select: none;
  }
  #selection-badge:hover { background: rgba(244,63,94,0.32); }

  #seq-row {
    width: 100%;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    overflow: hidden;
  }
  #seq-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-bottom: 2px solid #f1f5f9;
  }
  #seq-header h3 {
    margin: 0;
    color: #1e293b;
    font-size: 15px;
  }
  #seq-filter-note {
    font-size: 11px;
    color: #f43f5e;
    font-weight: 600;
    display: none;
    background: rgba(244,63,94,0.08);
    border: 1px solid #fecdd3;
    border-radius: 4px;
    padding: 3px 8px;
  }
  #sequence-container {
    height: 180px;
    overflow-y: auto;
    padding: 16px 20px;
    letter-spacing: 8px;
    line-height: 38px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
    font-size: 17px;
    word-break: break-all;
    user-select: none;
  }
</style>

<div id="root-layout">
  <div id="panels-row">
    <div class="panel-wrap">
      <div class="panel-label">🔭 Overview — Full Structure</div>
      <div class="viewport" id="viewport-overview">
        <div id="selection-badge" onclick="clearPeptideSelection()">
          ✕ &nbsp;<span id="badge-label">Peptide selected</span>
        </div>
        <div id="hover-card">
          <div class="card-top">
            <span id="card-residue" style="font-weight:bold;font-size:15px;color:#3b82f6;">Residue: --</span>
            <span id="card-plddt" style="font-weight:bold;padding:2px 6px;border-radius:4px;font-size:12px;">pLDDT: --</span>
          </div>
          <div id="card-status" style="margin-bottom:4px;">Status: <span style="color:#94a3b8;">--</span></div>
          <div id="card-ptm" style="font-weight:bold;display:none;">
            Modification: <span id="card-ptm-name">--</span>
            <div class="card-ptm-detail" id="card-ptm-detail"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-wrap">
      <div class="panel-label">🔬 Zoomed Inset — Selected Peptide</div>
      <div class="viewport" id="viewport-zoom">
        <div id="zoom-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            <path d="M11 8v6M8 11h6" stroke-width="1.5"/>
          </svg>
          <span>Click a detected peptide<br>in the overview to zoom in</span>
        </div>
      </div>
    </div>
  </div>

  <div id="seq-row">
    <div id="seq-header">
      <h3>Parallel Sequence Mapping Channel</h3>
      <span id="seq-filter-note">Showing selected peptide — click ✕ badge to reset</span>
    </div>
    <div id="sequence-container"></div>
  </div>
</div>

<script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<script>
const data = """ + json.dumps(js_data) + """;

const fullSeq      = data.fullSeq;
const detectionMap = data.detectionMap;
const ptmMap       = data.ptmMap;
const plddtMap     = data.plddtMap;
const PDB_URL          = data.PDB_URL;
const ENABLE_SURF      = data.ENABLE_SURF;
const SURF_OPACITY     = data.SURF_OPACITY;
const BACKBONE_STYLE   = data.backboneStyle;   // "ribbon" | "cpk"
const MANUAL_SELECTION = data.manualSelection;  // {start, end} from the sidebar range picker, or null

// ── PTM CHEMICALLY-ACCURATE STRUCTURAL RENDERER ──────────────────────────────
// Renders each PTM site using NGL's "hyperball" representation (a true
// CPK-style atomic model with real bond geometry and van-der-Waals-scaled
// spheres) applied to the residue atoms that actually exist in the
// structure, since AlphaFold models don't contain the modification's own
// atoms. Additional cues (anchor spheres, element highlights) are derived
// from the real composition formula fetched from UNIMOD.
//
// NOTE: this function is completely independent of BACKBONE_STYLE — PTM
// sites always render this way, whether the backbone toggle is set to
// Ribbon or CPK.
function addPtmStructuralReprs(comp, rn, ptm, withLabel = false) {
  if (!ptm) return;

  const baseSele = nglSele(rn, rn);
  const color    = ptm.color || "#436da9";
  const compStr  = (ptm.composition || "").toUpperCase();
  const mass     = ptm.mass || 0;
  const reprType = ptm.repr_type || "hyperball";

  // Primary chemically-accurate representation of the modified residue.
  // Scale kept modest (0.32) so this reads as an atomic-detail accent on
  // the residue, not a blob that swallows the surrounding backbone shape.
  comp.addRepresentation(reprType, {
    sele: baseSele + " AND sidechain",
    color: color,
    scale: 0.32,
    opacity: 1.0
  });

  // Phosphorus-bearing mods → single small anchor sphere on Cα (phosphorylation, etc.)
  // Only rendered for P-containing mods, so most PTMs add zero extra spheres.
  if (compStr.includes("P")) {
    comp.addRepresentation("spacefill", {
      sele: baseSele + " AND .CA", color: color, scale: 0.32, opacity: 0.6
    });
  }

  // Sulfur-bearing mods with meaningful mass shift → highlight SD/SG only
  if (compStr.includes("S") && mass > 30) {
    ["SD", "SG"].forEach(function(atom) {
      comp.addRepresentation("spacefill", {
        sele: baseSele + " AND ." + atom,
        color: "#f0abfc", scale: 0.4, opacity: 0.8
      });
    });
  }

  // Nitrogen+oxygen heavy mods (e.g. nitration) → hydroxyl oxygen only
  if (compStr.includes("N") && compStr.includes("O") && mass > 30) {
    comp.addRepresentation("spacefill", {
      sele: baseSele + " AND .OH", color: "#fed7aa", scale: 0.35, opacity: 0.8
    });
  }

  // NOTE: the old "always-present Cα marker" spacefill sphere was removed
  // here — it fired on every single PTM regardless of chemistry and was
  // the main reason the whole structure ended up reading as a cluster of
  // spheres. The hyperball sidechain representation above already marks
  // the site with real atomic geometry, so no generic filler sphere is needed.

  if (withLabel) {
    comp.addRepresentation("label", {
      sele: baseSele + " AND .CA",
      labelType: "text",
      labelText: ptm.name + (mass ? (" (+" + mass.toFixed(2) + " Da)") : ""),
      color: color,
      scale: 1.2,
      showBackground: true,
      backgroundColor: "black",
      backgroundOpacity: 0.6
    });
  }
}

// ── BACKBONE REPRESENTATION (Ribbon ⇄ CPK/Hyperball toggle) ──────────────────
// This is the single switch point for how the protein *chain itself* is
// drawn. When BACKBONE_STYLE === "cpk" the backbone is rendered with the
// same hyperball (CPK-style) atomic representation used for PTM sites, so
// the whole structure reads as one consistent atomic-detail model. When
// BACKBONE_STYLE === "ribbon" (default) it falls back to the classic
// cartoon ribbon. PTM markers (addPtmStructuralReprs above) are never
// touched by this — they always stay hyperball regardless of the toggle.
function addBackboneRepr(comp, sele, color, opacity, opts) {
  opts = opts || {};
  if (BACKBONE_STYLE === "cpk") {
    comp.addRepresentation("hyperball", {
      sele: sele,
      color: color,
      opacity: opacity,
      scale: opts.cpkScale || 0.22
    });
  } else {
    comp.addRepresentation(opts.reprType || "cartoon", {
      sele: sele,
      color: color,
      opacity: opacity
    });
  }
}

// ── COLOURS ──────────────────────────────────────────────────────────────────
const C_BACKBONE  = "#475569";
const C_PEPTIDE   = "#3b82f6";
const C_SELECT_HI = "#f43f5e";
const C_GHOST     = "#1e293b";

// ── STATE ─────────────────────────────────────────────────────────────────────
let overviewComp        = null;
let currentHighlightRep = null;
let selectedPeptide     = null;

function nglSele(start, end) {
  return start + "-" + end + ":A";
}

const peptideSegments = [];
(function() {
  let s = null;
  for (let i = 0; i < detectionMap.length; i++) {
    if (detectionMap[i]) {
      if (s === null) s = i + 1;
    } else {
      if (s !== null) { peptideSegments.push({ start: s, end: i }); s = null; }
    }
  }
  if (s !== null) peptideSegments.push({ start: s, end: detectionMap.length });
})();

function findSegmentForRes(resNum) {
  for (const seg of peptideSegments) {
    if (resNum >= seg.start && resNum <= seg.end) return seg;
  }
  return null;
}

const stageOverview = new NGL.Stage("viewport-overview", { backgroundColor: "#0b0f19" });
const stageZoom     = new NGL.Stage("viewport-zoom",     { backgroundColor: "#0b0f19" });

// Resolves a real AtomProxy from any NGL PickingProxy, whether the pick
// landed on an atom directly (cartoon/spacefill) or on a bond (hyperball).
function resolveAtom(proxy) {
  if (!proxy) return null;
  if (proxy.atom) return proxy.atom;
  if (proxy.closestBondAtom) return proxy.closestBondAtom;
  if (proxy.bond) return proxy.bond.atom1;
  return null;
}

// ── OVERVIEW LOAD ─────────────────────────────────────────────────────────────
stageOverview.loadFile(PDB_URL).then(function(o) {
  overviewComp = o;
  _applyOverviewReps(null);

  // ── HOVER = preview only (NO scrolling) ─────────────────────────────
  stageOverview.signals.hovered.add(function(proxy) {
    const atom = resolveAtom(proxy);
    if (!atom || !atom.resno) return;
    updateGlobalFocus(atom.resno, false); // false = don't scroll sequence
  });

  // ── CLICK = lock selection + scroll sequence ────────────────────────
  stageOverview.signals.clicked.add(function(proxy) {
    const atom = resolveAtom(proxy);

    if (!atom || !atom.resno) {
      clearPeptideSelection();
      return;
    }

    updateGlobalFocus(atom.resno, true); // true = scroll sequence to residue

    const seg = findSegmentForRes(atom.resno);
    if (seg) {
      selectPeptide(seg);
    } else {
      clearPeptideSelection();
    }
  });

  o.autoView();

  // A manual range picked in the sidebar reuses the exact same zoom path as
  // clicking a peptide — selectPeptide() doesn't care where the range came
  // from, it just needs {start, end}.
  if (MANUAL_SELECTION && MANUAL_SELECTION.start && MANUAL_SELECTION.end) {
    selectPeptide({ start: MANUAL_SELECTION.start, end: MANUAL_SELECTION.end });
  }
});

// ── APPLY OVERVIEW REPRESENTATIONS ───────────────────────────────────────────
function _applyOverviewReps(selSeg) {
  if (!overviewComp) return;

  overviewComp.removeAllRepresentations();
  currentHighlightRep = null;

  if (selSeg === null) {
    // Base backbone — ribbon or CPK depending on the sidebar toggle.
    addBackboneRepr(overviewComp, "polymer", C_BACKBONE, 0.5, { cpkScale: 0.18 });

    const allDetSele = peptideSegments.map(function(s) { return nglSele(s.start, s.end); }).join(" OR ");
    if (allDetSele) {
      addBackboneRepr(overviewComp, allDetSele, C_PEPTIDE, 0.95, { cpkScale: 0.24 });
    }

    ptmMap.forEach(function(ptm, idx) {
      if (ptm !== null) {
        const rn = idx + 1;
        addPtmStructuralReprs(overviewComp, rn, ptm, false);
      }
    });

    if (ENABLE_SURF) {
      overviewComp.addRepresentation("surface", {
        sele: "polymer", color: "electrostatic", surfaceType: "sas",
        opacity: SURF_OPACITY, useWorker: true
      });
    }

  } else {
    // Ghosted backbone — dimmer in CPK mode since hyperball atoms are
    // visually denser than a thin cartoon ribbon at the same opacity.
    addBackboneRepr(
      overviewComp, "polymer", C_GHOST,
      BACKBONE_STYLE === "cpk" ? 0.06 : 0.12,
      { cpkScale: 0.14 }
    );

    const selSele = nglSele(selSeg.start, selSeg.end);
    addBackboneRepr(overviewComp, selSele, C_SELECT_HI, 1.0, { cpkScale: 0.26 });

    // The extra "tube" accent only makes sense in ribbon mode — in CPK mode
    // the hyperball representation above already reads as a solid atomic
    // chain, so a tube on top would just add visual noise.
    if (BACKBONE_STYLE === "ribbon") {
      overviewComp.addRepresentation("tube", {
        sele: selSele, color: C_SELECT_HI, radius: 0.5, opacity: 0.5
      });
    }

    for (let i = selSeg.start - 1; i < selSeg.end; i++) {
      if (ptmMap[i] !== null) {
        const rn = i + 1;
        addPtmStructuralReprs(overviewComp, rn, ptmMap[i], false);
      }
    }
  }
}

// ── ZOOMED INSET ──────────────────────────────────────────────────────────────
function _loadZoomPanel(seg) {
  stageZoom.removeAllComponents();
  document.getElementById("zoom-placeholder").style.display = "none";

  stageZoom.loadFile(PDB_URL).then(function(o) {
    const selSele = nglSele(seg.start, seg.end);

    // NOTE: there is deliberately no "ghost" representation of the rest of
    // the structure here (the old version added a faint, 8%-opacity cartoon
    // covering the whole protein). Even at low opacity, that still visually
    // reads as "the whole structure is showing." The zoom panel now only
    // ever adds representations scoped to `selSele`, so nothing outside the
    // chosen residue range is rendered at all — not even faintly.

    addBackboneRepr(o, selSele, C_SELECT_HI, 1.0, { cpkScale: 0.3 });

    // Tube-on-backbone accent stays ribbon-mode only: ball+stick/hyperball
    // in CPK mode already draws every backbone atom with real geometry, so
    // stacking a tube underneath it would be redundant. In ribbon mode the
    // thin tube still follows the actual backbone path with a smooth pipe,
    // preserving chain shape while PTM sites keep their own atomic detail.
    if (BACKBONE_STYLE === "ribbon") {
      o.addRepresentation("tube", {
        sele: selSele + " AND backbone", color: C_SELECT_HI, radius: 0.3, opacity: 0.9
      });
    }

    for (let i = seg.start - 1; i < seg.end; i++) {
      if (ptmMap[i] !== null) {
        const rn = i + 1;
        addPtmStructuralReprs(o, rn, ptmMap[i], true);
      }
    }

    // Focus the camera tightly on just the selected range, not the full
    // structure's bounding box.
    o.autoView(selSele, 1500);
  });
}

// ── SELECT / CLEAR PEPTIDE ────────────────────────────────────────────────────
function selectPeptide(seg) {
  selectedPeptide = seg;
  _loadZoomPanel(seg);

  document.getElementById("selection-badge").style.display = "block";
  document.getElementById("badge-label").textContent =
    "Peptide " + seg.start + "–" + seg.end + "  ✕";

  _update1DForSelection(seg);
}

function clearPeptideSelection() {
  selectedPeptide = null;

  stageZoom.removeAllComponents();
  document.getElementById("zoom-placeholder").style.display = "flex";
  document.getElementById("selection-badge").style.display = "none";
  document.getElementById("seq-filter-note").style.display = "none";
  _restore1DNormal();
}

// ── 1D SEQUENCE SELECTION STATE ───────────────────────────────────────────────
function _update1DForSelection(seg) {
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(span) {
    const rn = parseInt(span.getAttribute("data-resnum"));
    if (rn >= seg.start && rn <= seg.end) {
      span.style.opacity         = "1";
      span.style.outline         = "2px solid " + C_SELECT_HI;
      span.style.outlineOffset   = "1px";
      span.style.backgroundColor = span.getAttribute("data-orig-bg") || "transparent";
    } else {
      span.style.opacity         = "0.2";
      span.style.outline         = "none";
      span.style.backgroundColor = "transparent";
    }
  });
  const anchor = document.getElementById("res-" + seg.start);
  if (anchor) anchor.scrollIntoView({ behavior: "smooth", block: "center" });
  document.getElementById("seq-filter-note").style.display = "inline-block";
}

function _restore1DNormal() {
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(span) {
    span.style.opacity         = "1";
    span.style.outline         = "none";
    span.style.backgroundColor = span.getAttribute("data-orig-bg") || "transparent";
  });
}

// ── HUD CARD ─────────────────────────────────────────────────────────────────
function populateHudCard(resNum, letter, isDetected, plddt, ptm) {
  document.getElementById("card-residue").innerText = "Residue: " + letter + resNum;
  const plddtEl = document.getElementById("card-plddt");
  plddtEl.innerText = "pLDDT: " + plddt.toFixed(1);
  if (plddt >= 90) {
    plddtEl.style.backgroundColor = "#1e3a8a"; plddtEl.style.color = "#93c5fd";
  } else if (plddt >= 70) {
    plddtEl.style.backgroundColor = "#065f46"; plddtEl.style.color = "#6ee7b7";
  } else {
    plddtEl.style.backgroundColor = "#9a3412"; plddtEl.style.color = "#fdba74";
  }
  document.getElementById("card-status").innerHTML =
    "Status: " + (isDetected
      ? '<span style="color:#60a5fa;font-weight:bold;">Detected in Assay</span>'
      : '<span style="color:#64748b;">Undetected Sequence</span>');

  const ptmEl     = document.getElementById("card-ptm");
  const ptmNameEl = document.getElementById("card-ptm-name");
  const ptmDetail = document.getElementById("card-ptm-detail");

  if (ptm) {
    ptmEl.style.display = "block";
    ptmNameEl.innerHTML = '<span style="color:' + ptm.color + ';">' + ptm.name + ' (' + ptm.id + ')</span>';

    const massTxt = ptm.mass ? ptm.mass.toFixed(4) + " Da" : "";
    const compTxt = ptm.composition ? ptm.composition : "";
    let detailParts = [];
    if (compTxt) detailParts.push(compTxt);
    if (massTxt) detailParts.push(massTxt);
    let detailHtml = detailParts.join(" · ");
    if (ptm.source_url) {
      detailHtml += ' &nbsp;<a href="' + ptm.source_url + '" target="_blank">UNIMOD record ↗</a>';
    }
    ptmDetail.innerHTML = detailHtml;
  } else {
    ptmEl.style.display = "none";
  }
  document.getElementById("hover-card").style.display = "block";
}

// ── CENTRAL HOVER ROUTER ───────────────────────────────────────────────────
function updateGlobalFocus(resNum, triggeredFromCanvas) {
  const idx = resNum - 1;
  if (idx < 0 || idx >= fullSeq.length) return;

  const letter     = fullSeq[idx];
  const isDetected = detectionMap[idx];
  const plddt      = plddtMap[idx];
  const ptm        = ptmMap[idx];

  // 3D crosshair — only when event came from 1D grid
  if (overviewComp && !triggeredFromCanvas) {
    if (currentHighlightRep) {
      try { overviewComp.removeRepresentation(currentHighlightRep); } catch(e) {}
    }
    currentHighlightRep = overviewComp.addRepresentation("spacefill", {
      sele: nglSele(resNum, resNum) + ".CA",
      color: "#f43f5e", scale: 1.6
    });
  }

  // Reset 1D highlights (only when no peptide selected)
  if (!selectedPeptide) {
    document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(s) {
      s.style.outline         = "none";
      s.style.transform       = "none";
      s.style.backgroundColor = s.getAttribute("data-orig-bg") || "transparent";
    });
  }

  const span = document.getElementById("res-" + resNum);
  if (span) {
    span.style.outline         = "3px solid #f43f5e";
    span.style.transform       = "scale(1.3)";
    span.style.zIndex          = "100";
    span.style.backgroundColor = "#fee2e2";
    if (triggeredFromCanvas) {
      span.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  populateHudCard(resNum, letter, isDetected, plddt, ptm);
}

// ── BUILD 1D SEQUENCE GRID ────────────────────────────────────────────────────
const container = document.getElementById("sequence-container");
detectionMap.forEach(function(isDetected, i) {
  const resNum = i + 1;
  const letter = fullSeq[i];
  const ptm    = ptmMap[i];

  const wrapper = document.createElement("div");
  wrapper.style.display  = "inline-block";
  wrapper.style.position = "relative";

  const span = document.createElement("span");
  span.innerText = letter;
  span.id = "res-" + resNum;
  span.setAttribute("data-resnum", resNum);
  span.style.padding      = "2px 4px";
  span.style.cursor       = "pointer";
  span.style.borderRadius = "4px";
  span.style.transition   = "all 0.12s ease";

  if (isDetected) {
    span.style.backgroundColor = "#dbeafe";
    span.setAttribute("data-orig-bg", "#dbeafe");
    span.style.color      = "#1e40af";
    span.style.fontWeight = "bold";
  } else {
    span.style.color = "#94a3b8";
    span.setAttribute("data-orig-bg", "transparent");
  }

  if (ptm) {
    const dot = document.createElement("div");
    dot.style.position        = "absolute";
    dot.style.top             = "-4px";
    dot.style.left            = "35%";
    dot.style.width           = "8px";
    dot.style.height          = "8px";
    dot.style.borderRadius    = "50%";
    dot.style.backgroundColor = ptm.color;
    dot.style.border          = "1.5px solid #ffffff";
    wrapper.appendChild(dot);
  }

  span.addEventListener("mouseenter", function() { updateGlobalFocus(resNum, false); });
  span.addEventListener("click", function() { updateGlobalFocus(resNum, false); });

  wrapper.appendChild(span);
  container.appendChild(wrapper);
});
</script>
"""

components.html(custom_viewer_html, height=800)

st.subheader("📋 Active Dataset Rows")
st.dataframe(protein_df, use_container_width=True)
