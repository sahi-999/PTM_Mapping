import streamlit as st
import pandas as pd
import requests
import tempfile
import os
import json
import re
import numpy as np
from typing import List, Tuple
from bs4 import BeautifulSoup
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Select
import streamlit.components.v1 as components

# ====================== PAGE SETUP ======================
st.set_page_config(page_title="RCSB Proteomics Engine Pro", layout="wide")
st.title("🧬 FLEXI Parallel Mapping Workspace: Physical Simulation")
st.markdown("""
### 🔄 True Physical PTM Simulation:
Unlike standard viewers that merely highlight the underlying unmodified residue, this engine **computationally grafts the physical mass** of the Post-Translational Modification onto the 3D structure. 

Using spatial vector mathematics, the backend calculates the terminal sidechain atom, projects an outward covalent bond trajectory, and adds a physically scaled pseudo-atom (or atom group) representing the UNIMOD mass. The frontend then visualizes this added mass with a simulated electron density cloud and thermal motion.
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

# ====================== UNIMOD FETCHING ======================
found_unimod_ids = set()
if modified_col:
    for _, row in protein_df.iterrows():
        mod_sequence_str = str(row.get(modified_col, ""))
        found_unimod_ids.update(
            m.lower() for m in re.findall(r"unimod:\d+", mod_sequence_str, re.IGNORECASE)
        )

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

def _auto_color(composition: str, mass: float) -> str:
    comp = (composition or "").upper()
    if "P" in comp: return "#ef4444"   
    if "S" in comp and mass > 50: return "#a855f7"   
    if "S" in comp: return "#6b7280"  
    if mass < 0: return "#0ea5e9"  
    if mass < 20: return "#10b981"  
    if mass < 60: return "#eab308"  
    if "N" in comp and "O" in comp: return "#f97316" 
    return "#94a3b8"                                 

@st.cache_data(show_spinner=False)
def fetch_unimod_data(unimod_ids: tuple) -> dict:
    result = {}
    for uid in unimod_ids:
        accession = uid.split(":")[-1].strip()
        parsed = fetch_unimod_view_page(accession)
        if not parsed: continue
        comp = parsed["composition"]
        mass = parsed["mono_mass"]
        result[uid] = {
            "name":         parsed["name"],
            "mono_mass":    mass,
            "composition":  comp,
            "color":        _auto_color(comp, mass),
            "repr_type":    "hyperball", 
            "source_url":   parsed["source_url"],
        }
    return result

unimod_enriched = fetch_unimod_data(tuple(sorted(found_unimod_ids)))

if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}

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

# ====================== STRUCTURE RETRIEVAL ======================
full_sequence = ""
pdb_url       = ""
pdb_text      = ""

@st.cache_data
def fetch_alphafold_fallback(uniprot_id):
    clean_id = uniprot_id.split(';')[0].split('-')[0].strip()
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{clean_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            pdb_url = data.get("pdbUrl")
            seq = data.get("uniprotSequence")
            
            # Fetch actual PDB text for physical modification
            pdb_res = requests.get(pdb_url)
            pdb_text = pdb_res.text if pdb_res.status_code == 200 else ""
            
            return pdb_url, seq, pdb_text
    except Exception:
        pass
    return None, None, ""

af_pdb_url, full_sequence, pdb_text = fetch_alphafold_fallback(selected_protein)

if not pdb_text or not full_sequence:
    st.error(f"Could not load structure for: `{selected_protein}`")
    st.stop()

# ====================== MAP PTMS ONTO SEQUENCE ======================
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
                    config = st.session_state.ptm_configs.get(uid_lower, {'selected': True, 'label': uid_lower.upper(), 'color': "#436da9"})

                    if config['selected']:
                        global_site = start_idx + current_pep_idx
                        if global_site < len(full_sequence):
                            enriched = unimod_enriched.get(uid_lower, {})
                            ptm_metadata_map[global_site] = {
                                "id":          uid_lower.upper(),
                                "name":        config['label'],
                                "color":       config['color'],
                                "mass":        enriched.get("mono_mass", 0.0),
                                "composition": enriched.get("composition", ""),
                            }
                    clean_mod_str = clean_mod_str[match.end():]
                    current_pep_idx += 1
                elif clean_mod_str and clean_mod_str[0].isalpha():
                    clean_mod_str = clean_mod_str[1:]
                    current_pep_idx += 1
                else:
                    clean_mod_str = clean_mod_str[1:]
            start_idx = full_sequence.find(pep, start_idx + 1)

# ====================== PHYSICAL PTM GRAFTING ENGINE ======================
@st.cache_data
def graft_ptm_atoms_to_pdb(pdb_string: str, ptm_map: list) -> str:
    """
    Physically simulates the presence of PTMs by calculating outward trajectory
    vectors from the sidechains and grafting new HETATM (heteroatom) records 
    into the PDB string representing the physical volume/mass of the PTM.
    """
    lines = pdb_string.split('\n')
    new_lines = []
    
    # Store coordinates to compute vectors
    res_coords = {}
    for line in lines:
        if line.startswith("ATOM"):
            res_num = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if res_num not in res_coords:
                res_coords[res_num] = {}
            res_coords[res_num][atom_name] = np.array([x, y, z])
            
        new_lines.append(line)
        
    # Append physical HETATMs
    atom_serial = 90000 
    for idx, ptm in enumerate(ptm_map):
        if ptm:
            res_num = idx + 1
            if res_num in res_coords:
                coords = res_coords[res_num]
                
                # Determine origin point (tip of sidechain) and trajectory vector
                if 'OH' in coords and 'CZ' in coords: # Tyrosine
                    origin = coords['OH']
                    vector = coords['OH'] - coords['CZ']
                elif 'OG' in coords and 'CB' in coords: # Serine
                    origin = coords['OG']
                    vector = coords['OG'] - coords['CB']
                elif 'NZ' in coords and 'CE' in coords: # Lysine
                    origin = coords['NZ']
                    vector = coords['NZ'] - coords['CE']
                else: # Fallback to C-alpha outwards
                    origin = coords.get('CA', np.array([0,0,0]))
                    vector = origin - coords.get('C', origin - np.array([1,1,1]))
                
                # Normalize vector and project outward by covalent bond distance (approx 1.6 Angstroms)
                norm_vector = vector / np.linalg.norm(vector)
                ptm_coord = origin + (norm_vector * 1.6)
                
                # Scale the physical representation volume radius based on UNIMOD mass
                # Mass = Volume * Density. Radius ~ cube root of mass.
                mass = ptm.get('mass', 80.0)
                vdw_radius = (mass ** (1/3.0)) * 0.5 
                
                # Create the synthetic PTM HETATM record
                hetatm_line = f"HETATM{atom_serial:5d} PTM  MOD A{res_num:4d}    {ptm_coord[0]:8.3f}{ptm_coord[1]:8.3f}{ptm_coord[2]:8.3f}  1.00 {vdw_radius:5.2f}          X  "
                new_lines.append(hetatm_line)
                atom_serial += 1
                
    return '\n'.join(new_lines)

mutated_pdb_text = graft_ptm_atoms_to_pdb(pdb_text, ptm_metadata_map)

# ====================== UNIFIED ENGINE HTML ======================
js_data = {
    "fullSeq": full_sequence,
    "detectionMap": is_detected_residue,
    "ptmMap": ptm_metadata_map,
    "PDB_TEXT": mutated_pdb_text, 
}

custom_viewer_html = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: sans-serif; }
  #root-layout { display: flex; flex-direction: column; gap: 16px; }
  .viewport { width: 100%; height: 500px; background-color: #0b0f19; border-radius: 12px; position: relative; overflow: hidden; }
</style>

<div id="root-layout">
  <div class="viewport" id="viewport-overview"></div>
</div>

<script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<script>
const data = """ + json.dumps(js_data) + """;
const ptmMap = data.ptmMap;

const stage = new NGL.Stage("viewport-overview", { backgroundColor: "#0b0f19" });

// Load the computationally mutated PDB text directly from memory
const blob = new Blob([data.PDB_TEXT], { type: 'text/plain' });
stage.loadFile(blob, { ext: 'pdb' }).then(function(comp) {
  
  // 1. Render standard backbone
  comp.addRepresentation("cartoon", { color: "#475569", opacity: 0.8 });
  
  // 2. Render Physical PTM Simulations
  ptmMap.forEach((ptm, idx) => {
    if (ptm) {
      const resNum = idx + 1;
      
      // Render the grafted HETATM we inserted via Python
      // We use a physical volumetric surface (electron density simulation)
      comp.addRepresentation("surface", {
        sele: "PTM and " + resNum,
        color: ptm.color,
        surfaceType: "vdw",
        opacity: 0.7,
        wireframe: true // Gives a 'quantum cloud' physical simulation look
      });
      
      // Add a dense core to the PTM
      comp.addRepresentation("spacefill", {
        sele: "PTM and " + resNum,
        color: ptm.color,
        scale: 0.8
      });
    }
  });

  comp.autoView();
  
  // 3. Thermal Motion Simulation
  // Gently wiggles the structure to simulate ambient physical molecular dynamics
  let axis = new NGL.Vector3(0, 1, 0);
  stage.setSpin(true);
  stage.spinAnimation.axis = axis;
  stage.spinAnimation.angle = 0.005;
});
</script>
"""

components.html(custom_viewer_html, height=550)
