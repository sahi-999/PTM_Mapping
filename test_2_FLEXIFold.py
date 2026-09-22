import streamlit as st
import pandas as pd
import numpy as np
import requests
import tempfile
import os
import json
import re
import io
import base64
import zipfile
from typing import List, Tuple
from bs4 import BeautifulSoup
from Bio import SeqIO
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit.components.v1 as components

# ====================== PAGE SETUP ======================
st.set_page_config(page_title="Peptide3D Parallel Mapper Pro", page_icon="⚛️", layout="wide")

st.title("🧬 Peptide3D Parallel Mapper Pro")
st.markdown("""
### 🔄 Multi-Condition Parallel 1D/3D Synchronization
* **2–4 Condition Comparison**: Simultaneously view structural differences across up to 4 experimental groups (e.g., Control, Disease 1, Disease 2).
* **Z-Score Gradient Preserved**: Residues on both 1D sequence and 3D structures are colored according to quantitative Z-score intensity scales.
* **Synchronized Ghosting**: Clicking any residue or peptide segment highlights that region across **all 3D condition viewports simultaneously**, fading unselected regions to a ghosted 15% opacity.
* **Linked 3D Camera Controls**: Rotating or zooming in one panel updates the view angle across all structural panels instantly.
""")

# ====================== HELPER FUNCTIONS ======================
def z_score(intensities: np.ndarray) -> np.ndarray:
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    return np.zeros_like(log_int) if std_log == 0 else (log_int - mean_log) / std_log

def clean_and_find_mods(peptide: str):
    mod_list = []
    cleaned_seq = ""
    index = -1
    pattern = re.compile(r"([A-Z])(\(UniMod:(\d+)\))?", re.IGNORECASE)
    for match in pattern.finditer(peptide):
        aa, mod, num = match.groups()
        index += 1
        cleaned_seq += aa
        if num:
            mod_list.append((index, f"unimod:{num}"))
    return cleaned_seq, mod_list

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    ptm_positions = {}
    
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col].values)
    
    for idx, row in df.iterrows():
        pep = row.get('Stripped.Sequence', '')
        if not pep or pd.isna(pep):
            continue
        intensity = row.get(intensity_col, 0)
        if pd.isna(intensity) or intensity == 0:
            continue
            
        matches = list(re.finditer(re.escape(pep), protein_seq))
        if not matches:
            continue
            
        valid_found = False
        for match in matches:
            start = match.start()
            if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                continue
            valid_found = True
            end = start + len(pep)
            
            peptide_match = peptides[peptides['Stripped.Sequence'] == pep]
            if peptide_match.empty:
                continue
            pep_idx = peptide_match.index[0]
            z_val = float(z_scores[pep_idx])
            
            for i in range(start, end):
                if residue_vals[i] is None:
                    residue_vals[i] = [z_val]
                else:
                    residue_vals[i].append(z_val)
                    
            if ptm_col and ptm_col in row:
                ptm_seq = row[ptm_col]
                if pd.notna(ptm_seq) and '(UniMod:' in str(ptm_seq):
                    cleaned_pep, mods = clean_and_find_mods(str(ptm_seq))
                    if cleaned_pep == pep:
                        for rel_pos, unismod in mods:
                            abs_pos = start + rel_pos
                            if 0 <= abs_pos < seq_len:
                                if unismod not in ptm_positions:
                                    ptm_positions[unismod] = set()
                                ptm_positions[unismod].add(abs_pos)
                                
        if not valid_found:
            continue

    for i in range(seq_len):
        if residue_vals[i]:
            if overlap_strategy == 'merge':
                residue_vals[i] = float(np.mean(residue_vals[i]))
            elif overlap_strategy == 'highest':
                residue_vals[i] = float(np.max(residue_vals[i]))
            elif overlap_strategy in ['none', 'last']:
                residue_vals[i] = float(residue_vals[i][-1])
            else:
                residue_vals[i] = float(np.mean(residue_vals[i]))
        else:
            residue_vals[i] = None
            
    for k in ptm_positions:
        ptm_positions[k] = sorted(list(ptm_positions[k]))
        
    return residue_vals, ptm_positions

def generate_hex_colors(residue_vals, cmap_name='autumn', not_mapped_color='#d3d3d3'):
    cmap = colormaps[cmap_name]
    vals = [v for v in residue_vals if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    hex_colors = []
    for val in residue_vals:
        if val is None:
            hex_colors.append(not_mapped_color)
        else:
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            rgb = cmap(norm)[:3]
            hex_colors.append(mcolors.rgb2hex(rgb))
    return hex_colors, vmin, vmax

def _parse_unimod_view_page(html: str) -> dict:
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
def fetch_unimod_data(unimod_ids: tuple) -> dict:
    result = {}
    for uid in unimod_ids:
        accession = uid.split(":")[-1].strip()
        url = f"https://www.unimod.org/modifications_view.php?editid1={accession}"
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and resp.text:
                parsed = _parse_unimod_view_page(resp.text)
                if parsed["name"] and parsed["name"] != "Unknown":
                    parsed["accession"] = accession
                    parsed["source_url"] = url
                    result[uid.lower()] = parsed
        except Exception:
            pass
    return result

@st.cache_data
def extract_plddt_values(url_path, target_seq_len):
    scores = [80.0] * target_seq_len
    try:
        res = requests.get(url_path, timeout=10)
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

# ====================== DATA INPUT SECTION ======================
col1, col2 = st.columns(2)
with col1:
    csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"])
with col2:
    fasta_file = st.file_uploader("Upload FASTA Sequence", type=["fasta", "fa"])

if not csv_file or not fasta_file:
    st.info("👆 Please upload both your **Peptide CSV** and **FASTA Sequence** to begin.")
    st.stop()

df = pd.read_csv(csv_file)
fasta_str = fasta_file.getvalue().decode("utf-8")
seq_records = list(SeqIO.parse(io.StringIO(fasta_str), "fasta"))

if not seq_records:
    st.error("No valid sequence found in FASTA file.")
    st.stop()

# Detect Intensity Columns
intensity_cols = [c for c in df.columns if 'intensity' in c.lower() or 'sample' in c.lower() or 'area' in c.lower()]
if not intensity_cols:
    intensity_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not intensity_cols:
    st.error("Could not auto-detect intensity/quantitative columns in CSV.")
    st.stop()

# ====================== SIDEBAR CONFIGURATION ======================
st.sidebar.title("⚙️ Analysis Controls")

# Protein Selection
protein_col = 'Protein.Group' if 'Protein.Group' in df.columns else df.columns[0]
protein_options = sorted(df[protein_col].dropna().unique())
selected_protein = st.sidebar.selectbox("Target Protein Group", protein_options)

# Condition Mapping (2 to 4 Conditions)
st.sidebar.subheader("📊 Experimental Conditions Setup")
num_conditions = st.sidebar.slider("Number of Conditions to Compare", min_value=2, max_value=4, value=2)

conditions = {}
for i in range(num_conditions):
    c_cols = st.sidebar.columns([1, 1])
    with c_cols[0]:
        c_name = st.text_input(f"Name {i+1}", value=f"Group_{i+1}", key=f"cond_name_{i}")
    with c_cols[1]:
        default_idx = min(i, len(intensity_cols) - 1)
        c_col = st.selectbox(f"Column {i+1}", intensity_cols, index=default_idx, key=f"cond_col_{i}")
    conditions[c_name] = c_col

# Visualization Controls
st.sidebar.subheader("🎨 Gradient & Display Options")
selected_cmap = st.sidebar.selectbox("Z-Score Colormap", ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'], index=0)
not_mapped_color = st.sidebar.color_picker("Unmapped Residue Color", "#d3d3d3")
overlap_strategy = st.sidebar.selectbox("Peptide Overlap Strategy", ["merge", "highest", "last", "none"], index=0)
apply_tryptic = st.sidebar.checkbox("Apply Tryptic Rule (K/R Cleavage)", value=False)

st.sidebar.subheader("🧬 3D Backbone Style")
backbone_style_choice = st.sidebar.radio("Backbone Model", ["🎀 Cartoon Ribbon", "⚛️ CPK / Hyperball (Atomic)"], index=0)
backbone_style = "cpk" if "CPK" in backbone_style_choice else "ribbon"

enable_surface = st.sidebar.toggle("Render Solvent Accessible Surface", value=False)
surface_opacity = st.sidebar.slider("Surface Opacity", 0.0, 1.0, 0.3) if enable_surface else 0.0

# PDB Source Setup
st.sidebar.subheader("🏰 Structural Model (PDB)")
pdb_source = st.sidebar.selectbox("PDB Source", ["AlphaFold DB", "Upload Custom PDB"])
uploaded_pdb = None
if pdb_source == "Upload Custom PDB":
    uploaded_pdb = st.sidebar.file_uploader("Upload PDB File", type=["pdb"])

# ====================== DATA PROCESSING ======================
base_id = selected_protein.split('-')[0].split(';')[0].strip()

# Match FASTA
protein_seq = None
for rec in seq_records:
    if base_id in rec.id:
        protein_seq = str(rec.seq).upper()
        break
if not protein_seq:
    protein_seq = str(seq_records[0].seq).upper()

seq_len = len(protein_seq)

# Match PDB Data
pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
pdb_str = ""

if pdb_source == "AlphaFold DB":
    try:
        r = requests.get(pdb_url, timeout=10)
        if r.status_code == 200:
            pdb_str = r.text
        else:
            # Fallback to API check
            af_api = f"https://alphafold.ebi.ac.uk/api/prediction/{base_id}"
            res_api = requests.get(af_api, timeout=10).json()
            if res_api:
                pdb_url = res_api[0].get('pdbUrl', pdb_url)
                pdb_str = requests.get(pdb_url, timeout=10).text
    except Exception:
        pass
elif uploaded_pdb:
    pdb_str = uploaded_pdb.getvalue().decode("utf-8")

if not pdb_str:
    st.error(f"Could not retrieve 3D structure for `{base_id}`. Please upload a valid PDB file manually.")
    st.stop()

plddt_scores = extract_plddt_values(pdb_url, seq_len)

# Parse PTMs
ptm_col = 'PTM' if 'PTM' in df.columns else ('modified.sequence' if 'modified.sequence' in df.columns else None)
found_unimods = set()
if ptm_col:
    for val in df[ptm_col].dropna():
        for m in re.findall(r"unimod:\d+", str(val), re.IGNORECASE):
            found_unimods.add(m.lower())

unimod_data = fetch_unimod_data(tuple(sorted(found_unimods)))

# Process Each Condition Data
selected_df = df[df[protein_col] == selected_protein].copy()
condition_payloads = []

global_detected = [False] * seq_len
ptm_metadata_map = [None] * seq_len

for c_name, c_col in conditions.items():
    res_vals, ptms = map_peptides_to_residues(
        selected_df, protein_seq, c_col, overlap_strategy, ptm_col, apply_tryptic
    )
    hex_colors, vmin, vmax = generate_hex_colors(res_vals, selected_cmap, not_mapped_color)
    
    # Mark global detection
    for idx, v in enumerate(res_vals):
        if v is not None:
            global_detected[idx] = True

    # PTM mapping
    c_ptm_map = [None] * seq_len
    for u_id, pos_list in ptms.items():
        u_info = unimod_data.get(u_id.lower(), {})
        for pos in pos_list:
            ptm_obj = {
                "id": u_id.upper(),
                "name": u_info.get("name", u_id.upper()),
                "color": "#ef4444" if "p" in u_id else "#f97316",
                "mass": u_info.get("mono_mass", 0.0),
                "composition": u_info.get("composition", ""),
                "source_url": u_info.get("source_url", "")
            }
            c_ptm_map[pos] = ptm_obj
            ptm_metadata_map[pos] = ptm_obj

    condition_payloads.append({
        "name": c_name,
        "colors": hex_colors,
        "zscores": res_vals,
        "ptmMap": c_ptm_map,
        "minLog": float(vmin),
        "maxLog": float(vmax)
    })

# ====================== JS/HTML ENGINE PREPARATION ======================
js_data = {
    "fullSeq": protein_seq,
    "detectionMap": global_detected,
    "ptmMap": ptm_metadata_map,
    "plddtMap": plddt_scores,
    "PDB_DATA": pdb_str,
    "ENABLE_SURF": bool(enable_surface),
    "SURF_OPACITY": float(surface_opacity),
    "backboneStyle": backbone_style,
    "conditions": condition_payloads
}

custom_viewer_html = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  #root-layout { display: flex; flex-direction: column; gap: 16px; font-family: system-ui, -apple-system, sans-serif; }
  
  #panels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    width: 100%;
  }

  .panel-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }

  .panel-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #38bdf8;
    background: #0f172a;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #1e293b;
  }

  .viewport {
    width: 100%;
    height: 420px;
    background-color: #0b0f19;
    border: 1px solid #1e293b;
    border-radius: 10px;
    position: relative;
    overflow: hidden;
  }

  #hover-card {
    position: absolute;
    bottom: 12px; left: 12px; right: 12px;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    color: #f8fafc;
    font-size: 12px;
    display: none;
    backdrop-filter: blur(8px);
    z-index: 1000;
  }
  .card-top { display: flex; justify-content: space-between; margin-bottom: 4px; border-bottom: 1px solid #334155; padding-bottom: 4px; }
  .card-ptm { font-weight: bold; margin-top: 4px; }

  #selection-badge {
    display: none;
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid #f43f5e;
    border-radius: 6px;
    padding: 4px 10px;
    color: #fda4af;
    font-size: 12px;
    font-weight: 600;
    z-index: 999;
    cursor: pointer;
  }

  #seq-row {
    width: 100%;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #ffffff;
    overflow: hidden;
  }
  #seq-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
  }
  #seq-header h3 { margin: 0; color: #0f172a; font-size: 14px; }
  
  #sequence-container {
    height: 160px;
    overflow-y: auto;
    padding: 14px;
    letter-spacing: 6px;
    line-height: 32px;
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 15px;
    word-break: break-all;
    user-select: none;
  }
</style>

<div id="root-layout">
  <div id="selection-badge" onclick="clearSelection()">✕ &nbsp;<span id="badge-label">Selected</span></div>
  
  <div id="panels-grid"></div>

  <div id="hover-card">
    <div class="card-top">
      <span id="card-residue" style="font-weight:bold;font-size:14px;color:#38bdf8;">Residue: --</span>
      <span id="card-plddt" style="font-weight:bold;padding:2px 6px;border-radius:4px;">pLDDT: --</span>
    </div>
    <div id="card-scores" style="color:#cbd5e1;">Z-Scores: --</div>
    <div id="card-ptm" class="card-ptm" style="display:none;"></div>
  </div>

  <div id="seq-row">
    <div id="seq-header">
      <h3>Parallel Sequence Mapping Channel</h3>
      <span id="seq-filter-note" style="display:none;color:#f43f5e;font-size:11px;font-weight:bold;">Segment Filter Active — Click Badge to Reset</span>
    </div>
    <div id="sequence-container"></div>
  </div>
</div>

<script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<script>
const data = """ + json.dumps(js_data) + """;

const fullSeq        = data.fullSeq;
const detectionMap   = data.detectionMap;
const ptmMap         = data.ptmMap;
const plddtMap       = data.plddtMap;
const PDB_DATA       = data.PDB_DATA;
const ENABLE_SURF    = data.ENABLE_SURF;
const SURF_OPACITY   = data.SURF_OPACITY;
const BACKBONE_STYLE = data.backboneStyle;
const conditions     = data.conditions;

const stages = [];
const comps = [];
const schemeIds = [];
let selectedSeg = null;
let isUpdatingCamera = false;

// Initialize Dynamic CSS Grid for Condition Panels
const gridContainer = document.getElementById("panels-grid");
conditions.forEach((cond, idx) => {
  const wrap = document.createElement("div");
  wrap.className = "panel-wrap";
  
  const label = document.createElement("div");
  label.className = "panel-label";
  label.innerText = "🔬 " + cond.name;
  
  const viewport = document.createElement("div");
  viewport.className = "viewport";
  viewport.id = "viewport-cond-" + idx;
  
  wrap.appendChild(label);
  wrap.appendChild(viewport);
  gridContainer.appendChild(wrap);
});

// Helper for NGL residue selection string
function nglSele(start, end) {
  return start + "-" + end + ":A";
}

// Build Custom Z-Score Color Schemes in NGL Registry
conditions.forEach((cond, idx) => {
  const pairList = [];
  cond.colors.forEach((hex, i) => {
    pairList.push([hex, (i + 1) + ":A"]);
  });
  const schemeName = "zscore_scheme_" + idx;
  const schemeId = NGL.ColorMakerRegistry.addSelectionScheme(pairList, schemeName);
  schemeIds.push(schemeId);
});

// Instantiate NGL Stages & Load PDB Structure
conditions.forEach((cond, idx) => {
  const stage = new NGL.Stage("viewport-cond-" + idx, { backgroundColor: "#0b0f19" });
  stages.push(stage);

  // Load string PDB data
  const stringBlob = new Blob([PDB_DATA], { type: "text/plain" });
  stage.loadFile(stringBlob, { ext: "pdb" }).then(comp => {
    comps[idx] = comp;
    
    // Sync Camera Controls across all viewports
    stage.viewer.controls.addEventListener("change", () => {
      if (isUpdatingCamera) return;
      isUpdatingCamera = true;
      const orientation = stage.viewer.getOrientation();
      stages.forEach((stg, oIdx) => {
        if (oIdx !== idx) stg.viewer.setOrientation(orientation);
      });
      isUpdatingCamera = false;
    });

    // Handle 3D Pick / Click Event
    stage.signals.clicked.add(proxy => {
      if (!proxy || (!proxy.atom && !proxy.bond)) {
        clearSelection();
        return;
      }
      const atom = proxy.atom || (proxy.bond ? proxy.bond.atom1 : null);
      if (atom && atom.resno) {
        selectResidue(atom.resno);
      }
    });

    // Handle Hover Event
    stage.signals.hovered.add(proxy => {
      if (!proxy || (!proxy.atom && !proxy.bond)) return;
      const atom = proxy.atom || (proxy.bond ? proxy.bond.atom1 : null);
      if (atom && atom.resno) {
        updateHoverHUD(atom.resno);
      }
    });

    applyRepresentations();
    stage.autoView();
  });
});

// Apply / Update Structural Representations (Ghosting vs Default)
function applyRepresentations() {
  comps.forEach((comp, idx) => {
    if (!comp) return;
    comp.removeAllRepresentations();

    const schemeId = schemeIds[idx];
    const cond = conditions[idx];

    if (!selectedSeg) {
      // DEFAULT STATE: Full Z-Score heatmap
      comp.addRepresentation(BACKBONE_STYLE === "cpk" ? "hyperball" : "cartoon", {
        sele: "polymer",
        color: schemeId,
        opacity: 0.95
      });

      // Render PTM Hyperballs
      cond.ptmMap.forEach((ptm, i) => {
        if (ptm) {
          comp.addRepresentation("hyperball", {
            sele: (i + 1) + ":A AND sidechain",
            color: ptm.color,
            scale: 0.5
          });
        }
      });

      if (ENABLE_SURF) {
        comp.addRepresentation("surface", {
          sele: "polymer", color: "electrostatic", surfaceType: "sas", opacity: SURF_OPACITY
        });
      }
    } else {
      // SELECTED / GHOSTED STATE:
      const selSele = nglSele(selectedSeg.start, selectedSeg.end);
      const unselSele = "polymer AND NOT (" + selSele + ")";

      // 1. Ghost unselected backbone to 15% opacity dark slate
      comp.addRepresentation(BACKBONE_STYLE === "cpk" ? "hyperball" : "cartoon", {
        sele: unselSele,
        color: "#1e293b",
        opacity: 0.15
      });

      // 2. Full-opacity Z-score highlighted region
      comp.addRepresentation(BACKBONE_STYLE === "cpk" ? "hyperball" : "cartoon", {
        sele: selSele,
        color: schemeId,
        opacity: 1.0
      });

      if (BACKBONE_STYLE === "ribbon") {
        comp.addRepresentation("tube", {
          sele: selSele, color: schemeId, radius: 0.45, opacity: 0.9
        });
      }

      // Render PTMs in selected range
      for (let i = selectedSeg.start - 1; i < selectedSeg.end; i++) {
        if (cond.ptmMap[i]) {
          comp.addRepresentation("hyperball", {
            sele: (i + 1) + ":A AND sidechain",
            color: cond.ptmMap[i].color,
            scale: 0.7
          });
        }
      }
    }
  });
}

// 1D / 3D Selection Handlers
function selectResidue(resNum) {
  selectedSeg = { start: resNum, end: resNum };
  applyRepresentations();

  // Auto-focus cameras on selected range
  comps.forEach((comp, idx) => {
    if (comp) comp.autoView(nglSele(resNum, resNum), 1000);
  });

  document.getElementById("selection-badge").style.display = "block";
  document.getElementById("badge-label").innerText = "Residue " + resNum;
  document.getElementById("seq-filter-note").style.display = "inline";

  update1DHighlighting();
  updateHoverHUD(resNum);
}

function clearSelection() {
  selectedSeg = null;
  applyRepresentations();
  
  comps.forEach(comp => { if (comp) comp.autoView(1000); });

  document.getElementById("selection-badge").style.display = "none";
  document.getElementById("seq-filter-note").style.display = "none";
  restore1DHighlighting();
}

// HUD Hover Information
function updateHoverHUD(resNum) {
  const idx = resNum - 1;
  if (idx < 0 || idx >= fullSeq.length) return;

  const letter = fullSeq[idx];
  const plddt = plddtMap[idx] || 0;

  document.getElementById("card-residue").innerText = "Residue: " + letter + resNum;
  
  const plddtEl = document.getElementById("card-plddt");
  plddtEl.innerText = "pLDDT: " + plddt.toFixed(1);
  plddtEl.style.backgroundColor = plddt >= 70 ? "#065f46" : "#9a3412";

  let scoresHtml = conditions.map(c => {
    const val = c.zscores[idx];
    return "<b>" + c.name + ":</b> " + (val !== null ? val.toFixed(2) : "N/A");
  }).join(" &nbsp;|&nbsp; ");

  document.getElementById("card-scores").innerHTML = scoresHtml;

  const ptm = ptmMap[idx];
  const ptmEl = document.getElementById("card-ptm");
  if (ptm) {
    ptmEl.style.display = "block";
    ptmEl.innerHTML = "Modification: <span style='color:" + ptm.color + ";'>" + ptm.name + " (" + ptm.id + ")</span>";
  } else {
    ptmEl.style.display = "none";
  }

  document.getElementById("hover-card").style.display = "block";
}

// 1D Sequence Grid Rendering
const seqContainer = document.getElementById("sequence-container");
detectionMap.forEach((isDet, i) => {
  const resNum = i + 1;
  const letter = fullSeq[i];
  
  const wrapper = document.createElement("div");
  wrapper.style.display = "inline-block";
  wrapper.style.position = "relative";

  const span = document.createElement("span");
  span.innerText = letter;
  span.id = "res-" + resNum;
  span.setAttribute("data-resnum", resNum);
  span.style.padding = "2px 4px";
  span.style.cursor = "pointer";
  span.style.borderRadius = "4px";

  // Use base condition Z-score color as subtle background
  const baseColor = conditions[0].colors[i];
  span.style.backgroundColor = isDet ? baseColor + "33" : "transparent"; // 20% alpha
  span.setAttribute("data-orig-bg", span.style.backgroundColor);
  span.style.color = isDet ? "#0f172a" : "#94a3b8";
  span.style.fontWeight = isDet ? "bold" : "normal";

  if (ptmMap[i]) {
    const dot = document.createElement("div");
    dot.style.position = "absolute";
    dot.style.top = "-2px"; dot.style.left = "35%";
    dot.style.width = "6px"; dot.style.height = "6px";
    dot.style.borderRadius = "50%";
    dot.style.backgroundColor = ptmMap[i].color;
    wrapper.appendChild(dot);
  }

  span.addEventListener("mouseenter", () => updateHoverHUD(resNum));
  span.addEventListener("click", () => selectResidue(resNum));

  wrapper.appendChild(span);
  seqContainer.appendChild(wrapper);
});

function update1DHighlighting() {
  if (!selectedSeg) return;
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(span => {
    const rn = parseInt(span.getAttribute("data-resnum"));
    if (rn >= selectedSeg.start && rn <= selectedSeg.end) {
      span.style.outline = "2px solid #f43f5e";
      span.style.backgroundColor = "#ffe4e6";
    } else {
      span.style.opacity = "0.25";
      span.style.outline = "none";
    }
  });
}

function restore1DHighlighting() {
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(span => {
    span.style.opacity = "1";
    span.style.outline = "none";
    span.style.backgroundColor = span.getAttribute("data-orig-bg") || "transparent";
  });
}
</script>
"""

# Render Unified Component
components.html(custom_viewer_html, height=750)

# ====================== DATA TABLE & EXPORTS ======================
st.subheader("📋 Active Dataset & Export Center")

tab_table, tab_export = st.tabs(["Active Dataset Rows", "📦 Export Package (ZIP)"])

with tab_table:
    st.dataframe(selected_df, use_container_width=True)

with tab_export:
    if st.button("Generate Downloadable ZIP Package", use_container_width=True):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Write PDB
            zipf.writestr(f"{base_id}_structure.pdb", pdb_str)
            
            # Write PyMOL scripts for each condition
            for cond in condition_payloads:
                pml = f"load {base_id}_structure.pdb\nhide everything\nshow cartoon\ncolor gray80, all\n"
                for i, col in enumerate(cond["colors"]):
                    pml += f"color {col}, resi {i+1}\n"
                zipf.writestr(f"{base_id}_{cond['name']}_pymol.pml", pml)
                
        st.download_button(
            label="💾 Download ZIP File",
            data=zip_buffer.getvalue(),
            file_name=f"{base_id}_analysis_package.zip",
            mime="application/zip"
        )
