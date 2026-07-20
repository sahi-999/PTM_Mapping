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
st.title("🧬 FLEXI Parallel Mapping Workspace")
st.markdown("""
### 🔄 Complete Parallel Coordination:
1. **1D Sequence ➔ 3D Structure**: Hover or click a letter in the grid to highlight its exact spatial coordinate sphere and update the dashboard.
2. **3D Structure ➔ 1D Sequence**: Hover or click directly on any ribbon or atom in the 3D viewport.**scrolls the 1D sequence grid directly to that exact residue, and highlights it** instantly.
3. **Peptide Selection**: Click any detected peptide (blue region) in the 3D structure to **zoom into it** in the inset panel, ghost the rest, and grey out the 1D sequence outside that peptide.

> ⚠️ **Note on PTM rendering**: AlphaFold structures model the *unmodified* protein — the atoms of a phosphate/acetyl/etc. group don't physically exist in the file. PTMs are rendered as a chemically-accurate **hyperball** (CPK-style) model at the real modification site on the existing residue atoms, colored and labeled using data fetched live from each modification's own UNIMOD record page.
>
> The backbone can optionally be rendered in the **same hyperball stick-and-ball style** as the PTMs (see sidebar), so the whole structure reads as one consistent chemical model. Note that, like every molecular viewer (PyMOL, Chimera, Jmol, etc.), sphere sizes are intentionally shrunk below true van der Waals radii so bonds stay visible — true VdW spheres overlap and fuse into a blob. Bond lengths/angles and *relative* atom sizing (when using covalent/VdW radius type) are physically accurate; absolute sphere/stick thickness is a stylistic convention, not a measured quantity.
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
def fetch_unimod_view_page(accession: str):
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
    result = {}
    for uid in unimod_ids:
        accession = uid.split(":")[-1].strip()
        parsed = fetch_unimod_view_page(accession)
        if not parsed:
            return None
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

for uid in found_unimod_ids:
    enriched = unimod_enriched.get(uid, {})
    auto_label = enriched.get("name", uid.upper())
    auto_color = enriched.get("color", "#436da9")

    if uid not in st.session_state.ptm_configs:
        st.session_state.ptm_configs[uid] = {
            'selected':   True,
            'label':      auto_label,
            'color':      auto_color,
            'auto_color': auto_color,
        }
    else:
        st.session_state.ptm_configs[uid]['auto_color'] = auto_color

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
            for i in range(start_idx, start_idx + len(pep)):
                is_detected_residue[i] = True

            clean_mod_str = mod_sequence_str
            current_pep_idx = 0
            while clean_mod_str:
                match = re.match(r"^([A-Z])\((unimod:\d+)\)", clean_mod_str, re.IGNORECASE)
                if match:
                    aa, unimod_id = match.groups()
                    uid_lower = unimod_id.lower()

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
                                "name":        config['label'],
                                "color":       config['color'],
                                "repr_type":   enriched.get("repr_type", "hyperball"), 
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

# ====================== BACKBONE RENDERING STYLE ======================
st.sidebar.subheader("🧪 Backbone Rendering Style")

# Replaced the radio button with a streamlined toggle switch
use_hyperball_backbone = st.sidebar.toggle(
    "Show stick-and-ball backbone (instead of ribbon)",
    value=True,
    help="When enabled, renders the backbone atoms as a physical stick-and-ball (hyperball) model instead of the traditional ribbon/cartoon."
)

atom_scale = st.sidebar.slider(
    "Atom sphere scale (both backbone & PTMs)",
    min_value=0.15, max_value=0.60, value=0.32, step=0.01,
    help="Fraction of true van der Waals radius used for sphere size."
)

# ====================== MANUAL 3D REGION SELECTOR ======================
st.sidebar.subheader("🎯 Manual Structure Region Selector")
if "manual_zoom" not in st.session_state:
    st.session_state.manual_zoom = None

seq_len = len(full_sequence)
default_start = st.session_state.manual_zoom["start"] if st.session_state.manual_zoom else 1
default_end   = st.session_state.manual_zoom["end"] if st.session_state.manual_zoom else min(20, seq_len)

msel_cols = st.sidebar.columns(2)
with msel_cols[0]:
    manual_start = st.number_input("Start residue", min_value=1, max_value=seq_len, value=min(default_start, seq_len), key="manual_start_input")
with msel_cols[1]:
    manual_end = st.number_input("End residue", min_value=1, max_value=seq_len, value=min(default_end, seq_len), key="manual_end_input")

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
    "manualSelection": st.session_state.manual_zoom,
    "useHyperballBackbone": bool(use_hyperball_backbone),
    "atomScale": float(atom_scale),
}

custom_viewer_html = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  #root-layout { display: flex; flex-direction: column; gap: 16px; font-family: sans-serif; }
  #panels-row { display: flex; gap: 16px; width: 100%; }
  .panel-wrap { flex: 1; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
  .panel-label { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; padding-left: 4px; }
  .viewport { width: 100%; height: 480px; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 12px; position: relative; overflow: hidden; }
  #zoom-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #334155; font-size: 14px; gap: 10px; pointer-events: none; z-index: 5; }
  #hover-card { position: absolute; bottom: 16px; left: 16px; right: 16px; background: rgba(15, 23, 42, 0.95); border: 1px solid #334155; border-radius: 8px; padding: 14px; color: #f8fafc; font-size: 13px; display: none; z-index: 1000; }
  #hover-card .card-top { display: flex; justify-content: space-between; margin-bottom: 7px; border-bottom: 1px solid #334155; padding-bottom: 6px; }
  #selection-badge { display: none; position: absolute; top: 12px; left: 12px; background: rgba(244,63,94,0.18); border: 1px solid #f43f5e; border-radius: 6px; padding: 6px 10px; color: #fda4af; font-size: 12px; font-weight: 600; z-index: 999; cursor: pointer; }
  #seq-row { width: 100%; border: 1px solid #e2e8f0; border-radius: 12px; background: #ffffff; overflow: hidden; }
  #seq-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 2px solid #f1f5f9; }
  #sequence-container { height: 180px; overflow-y: auto; padding: 16px 20px; letter-spacing: 8px; line-height: 38px; font-family: monospace; font-size: 17px; word-break: break-all; }
</style>

<div id="root-layout">
  <div id="panels-row">
    <div class="panel-wrap">
      <div class="panel-label">🔭 Overview — Full Structure</div>
      <div class="viewport" id="viewport-overview">
        <div id="selection-badge" onclick="clearPeptideSelection()">✕ &nbsp;<span id="badge-label">Peptide selected</span></div>
        <div id="hover-card">
          <div class="card-top">
            <span id="card-residue" style="font-weight:bold;font-size:15px;color:#3b82f6;">Residue: --</span>
            <span id="card-plddt" style="font-weight:bold;padding:2px 6px;border-radius:4px;font-size:12px;">pLDDT: --</span>
          </div>
          <div id="card-status" style="margin-bottom:4px;">Status: <span style="color:#94a3b8;">--</span></div>
          <div id="card-ptm" style="font-weight:bold;display:none;">Modification: <span id="card-ptm-name">--</span></div>
        </div>
      </div>
    </div>
    <div class="panel-wrap">
      <div class="panel-label">🔬 Zoomed Inset — Selected Peptide</div>
      <div class="viewport" id="viewport-zoom">
        <div id="zoom-placeholder"><span>Click a detected peptide to zoom in</span></div>
      </div>
    </div>
  </div>
  <div id="seq-row">
    <div id="seq-header"><h3>Parallel Sequence Mapping Channel</h3><span id="seq-filter-note" style="display:none; color:#f43f5e;">Active Selection</span></div>
    <div id="sequence-container"></div>
  </div>
</div>

<script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<script>
const data = """ + json.dumps(js_data) + """;
const stageOverview = new NGL.Stage("viewport-overview", { backgroundColor: "#0b0f19" });
const stageZoom     = new NGL.Stage("viewport-zoom",     { backgroundColor: "#0b0f19" });

let overviewComp = null;
let currentHighlightRep = null;
let selectedPeptide = null;

function nglSele(start, end) { return start + "-" + end + ":A"; }

function addPtmStructuralReprs(comp, rn, ptm) {
  if (!ptm) return;
  comp.addRepresentation("hyperball", {
    sele: nglSele(rn, rn) + " AND sidechain",
    color: ptm.color,
    radiusType: "covalent",
    scale: data.atomScale
  });
}

function addBackboneHyperball(comp, sele, color, opacity, isOverview) {
  // Changed representation to NGL's 'ball+stick' and restricted ONLY to backbone atoms
  comp.addRepresentation("ball+stick", {
    sele: sele + " AND backbone", 
    color: color,
    radiusType: "vdw",       // Uses van der Waals radii for that authentic CPK look
    scale: data.atomScale,   // Controlled by your Streamlit slider
    opacity: opacity,
    sphereDetail: isOverview ? 1 : 2,
    radialSegments: isOverview ? 10 : 20
  });
}

const peptideSegments = [];
for (let i = 0; i < data.detectionMap.length; i++) {
  if (data.detectionMap[i]) {
    let s = i + 1;
    while(i < data.detectionMap.length && data.detectionMap[i]) { i++; }
    peptideSegments.push({ start: s, end: i });
  }
}

stageOverview.loadFile(data.PDB_URL).then(function(o) {
  overviewComp = o;
  _applyOverviewReps(null);
  
  stageOverview.signals.hovered.add(function(proxy) {
    let atom = proxy && (proxy.atom || proxy.closestBondAtom || (proxy.bond && proxy.bond.atom1));
    if (atom && atom.resno) updateGlobalFocus(atom.resno, false);
  });

  stageOverview.signals.clicked.add(function(proxy) {
    let atom = proxy && (proxy.atom || proxy.closestBondAtom || (proxy.bond && proxy.bond.atom1));
    if (!atom || !atom.resno) { clearPeptideSelection(); return; }
    updateGlobalFocus(atom.resno, true);
    let seg = peptideSegments.find(s => atom.resno >= s.start && atom.resno <= s.end);
    if (seg) selectPeptide(seg); else clearPeptideSelection();
  });

  o.autoView();
  if (data.manualSelection) selectPeptide(data.manualSelection);
});

function _applyOverviewReps(selSeg) {
  if (!overviewComp) return;
  overviewComp.removeAllRepresentations();
  
  if (selSeg === null) {
    if (data.useHyperballBackbone) {
      addBackboneHyperball(overviewComp, "polymer", "#475569", 0.55, true);
    } else {
      overviewComp.addRepresentation("cartoon", { color: "#475569", opacity: 0.5 });
    }

    let allDetSele = peptideSegments.map(s => nglSele(s.start, s.end)).join(" OR ");
    if (allDetSele) {
      if (data.useHyperballBackbone) {
        addBackboneHyperball(overviewComp, allDetSele, "#3b82f6", 0.95, true);
      } else {
        overviewComp.addRepresentation("cartoon", { sele: allDetSele, color: "#3b82f6", opacity: 0.95 });
      }
    }

    data.ptmMap.forEach((ptm, idx) => { if (ptm) addPtmStructuralReprs(overviewComp, idx + 1, ptm); });
  } else {
    if (data.useHyperballBackbone) {
      addBackboneHyperball(overviewComp, "polymer", "#1e293b", 0.1, true);
      addBackboneHyperball(overviewComp, nglSele(selSeg.start, selSeg.end), "#f43f5e", 1.0, true);
    } else {
      overviewComp.addRepresentation("cartoon", { color: "#1e293b", opacity: 0.12 });
      overviewComp.addRepresentation("cartoon", { sele: nglSele(selSeg.start, selSeg.end), color: "#f43f5e", opacity: 1.0 });
      overviewComp.addRepresentation("tube", { sele: nglSele(selSeg.start, selSeg.end), color: "#f43f5e", radius: 0.5, opacity: 0.5 });
    }
    for (let i = selSeg.start - 1; i < selSeg.end; i++) { if (data.ptmMap[i]) addPtmStructuralReprs(overviewComp, i + 1, data.ptmMap[i]); }
  }
}

function _loadZoomPanel(seg) {
  stageZoom.removeAllComponents();
  document.getElementById("zoom-placeholder").style.display = "none";
  stageZoom.loadFile(data.PDB_URL).then(function(o) {
    addBackboneHyperball(o, nglSele(seg.start, seg.end), "#f43f5e", 1.0, false);
    for (let i = seg.start - 1; i < seg.end; i++) { if (data.ptmMap[i]) addPtmStructuralReprs(o, i + 1, data.ptmMap[i]); }
    o.autoView(nglSele(seg.start, seg.end), 1500);
  });
}

function selectPeptide(seg) { 
  selectedPeptide = seg; 
  _loadZoomPanel(seg); 
  _applyOverviewReps(seg); 
  document.getElementById("selection-badge").style.display = "block";
  document.getElementById("badge-label").textContent = "Peptide " + seg.start + "–" + seg.end + "  ✕";
  _update1DForSelection(seg);
}

function clearPeptideSelection() { 
  selectedPeptide = null; 
  stageZoom.removeAllComponents(); 
  document.getElementById("zoom-placeholder").style.display = "flex"; 
  document.getElementById("selection-badge").style.display = "none";
  document.getElementById("seq-filter-note").style.display = "none";
  _applyOverviewReps(null); 
  _restore1DNormal();
}

function _update1DForSelection(seg) {
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(span) {
    const rn = parseInt(span.getAttribute("data-resnum"));
    if (rn >= seg.start && rn <= seg.end) {
      span.style.opacity = "1";
      span.style.outline = "2px solid #f43f5e";
      span.style.outlineOffset = "1px";
      span.style.backgroundColor = span.getAttribute("data-orig-bg") || "transparent";
    } else {
      span.style.opacity = "0.2";
      span.style.outline = "none";
      span.style.backgroundColor = "transparent";
    }
  });
  const anchor = document.getElementById("res-" + seg.start);
  if (anchor) anchor.scrollIntoView({ behavior: "smooth", block: "center" });
  document.getElementById("seq-filter-note").style.display = "inline-block";
}

function _restore1DNormal() {
  document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(span) {
    span.style.opacity = "1";
    span.style.outline = "none";
    span.style.backgroundColor = span.getAttribute("data-orig-bg") || "transparent";
  });
}

function populateHudCard(resNum, letter, isDetected, plddt, ptm) {
  document.getElementById("card-residue").innerText = "Residue: " + letter + resNum;
  const plddtEl = document.getElementById("card-plddt");
  plddtEl.innerText = "pLDDT: " + plddt.toFixed(1);
  if (plddt >= 90) { plddtEl.style.backgroundColor = "#1e3a8a"; plddtEl.style.color = "#93c5fd"; }
  else if (plddt >= 70) { plddtEl.style.backgroundColor = "#065f46"; plddtEl.style.color = "#6ee7b7"; }
  else { plddtEl.style.backgroundColor = "#9a3412"; plddtEl.style.color = "#fdba74"; }

  document.getElementById("card-status").innerHTML = "Status: " + (isDetected ? '<span style="color:#60a5fa;font-weight:bold;">Detected in Assay</span>' : '<span style="color:#64748b;">Undetected Sequence</span>');

  const ptmEl = document.getElementById("card-ptm");
  if (ptm) {
    ptmEl.style.display = "block";
    document.getElementById("card-ptm-name").innerHTML = '<span style="color:' + ptm.color + ';">' + ptm.name + '</span>';
  } else {
    ptmEl.style.display = "none";
  }
  document.getElementById("hover-card").style.display = "block";
}

function updateGlobalFocus(resNum, triggeredFromCanvas) {
  const idx = resNum - 1;
  if (idx < 0 || idx >= data.fullSeq.length) return;

  if (overviewComp && !triggeredFromCanvas) {
    if (currentHighlightRep) { try { overviewComp.removeRepresentation(currentHighlightRep); } catch(e) {} }
    currentHighlightRep = overviewComp.addRepresentation("spacefill", { sele: nglSele(resNum, resNum) + ".CA", color: "#f43f5e", scale: 1.6 });
  }

  if (!selectedPeptide) {
    document.querySelectorAll("#sequence-container span[data-resnum]").forEach(function(s) {
      s.style.outline = "none"; s.style.transform = "none"; s.style.backgroundColor = s.getAttribute("data-orig-bg") || "transparent";
    });
  }

  const span = document.getElementById("res-" + resNum);
  if (span) {
    span.style.outline = "3px solid #f43f5e"; span.style.transform = "scale(1.3)"; span.style.zIndex = "100"; span.style.backgroundColor = "#fee2e2";
    if (triggeredFromCanvas) span.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  populateHudCard(resNum, data.fullSeq[idx], data.detectionMap[idx], data.plddtMap[idx], data.ptmMap[idx]);
}

const container = document.getElementById("sequence-container");
data.detectionMap.forEach(function(isDetected, i) {
  const resNum = i + 1;
  const ptm = data.ptmMap[i];

  const wrapper = document.createElement("div");
  wrapper.style.display = "inline-block"; wrapper.style.position = "relative";

  const span = document.createElement("span");
  span.innerText = data.fullSeq[i]; span.id = "res-" + resNum; span.setAttribute("data-resnum", resNum);
  span.style.padding = "2px 4px"; span.style.cursor = "pointer"; span.style.borderRadius = "4px"; span.style.transition = "all 0.12s ease";

  if (isDetected) {
    span.style.backgroundColor = "#dbeafe"; span.setAttribute("data-orig-bg", "#dbeafe"); span.style.color = "#1e40af"; span.style.fontWeight = "bold";
  } else {
    span.style.color = "#94a3b8"; span.setAttribute("data-orig-bg", "transparent");
  }

  if (ptm) {
    const dot = document.createElement("div");
    dot.style.position = "absolute"; dot.style.top = "-4px"; dot.style.left = "35%"; dot.style.width = "8px"; dot.style.height = "8px";
    dot.style.borderRadius = "50%"; dot.style.backgroundColor = ptm.color; dot.style.border = "1.5px solid #ffffff";
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
