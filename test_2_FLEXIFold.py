import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import py3Dmol
import io
import requests
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import zipfile
from streamlit.components.v1 import html
import json  # Keep only this import
from Bio.PDB import PDBParser
import re
import time

# Set wide layout #

# Page config
st.set_page_config(page_title="Peptide3D Mapper", page_icon="⚛️", layout="wide")
#st.title("🧬 Peptide3D Mapper")

# --- Helper Functions ---
def z_score(intensities):
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    return np.zeros_like(log_int) if std_log == 0 else (log_int - mean_log) / std_log

def clean_and_find_mods(peptide):
    """
    Cleans a peptide sequence and finds UniMod modification positions and types.
    Returns:
        cleaned_seq: sequence without UniMod tags
        mod_list: list of tuples (0-based position, unimod_type)
    """
    mod_list = []
    cleaned_seq = ""
    index = -1  # Start at -1 to make position 0-based after first increment
    pattern = re.compile(r"([A-Z])(\(UniMod:(\d+)\))?", re.IGNORECASE)
    for match in pattern.finditer(peptide):
        aa, mod, num = match.groups()
        index += 1
        cleaned_seq += aa
        if num:
            mod_list.append((index, num))  # 0-based position
    #st.write(f"Parsed PTM: {peptide} -> Cleaned: {cleaned_seq}, Mods: {mod_list}")  # Debug
    return cleaned_seq, mod_list

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    ptm_positions = {}
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col])
    
    for idx, row in df.iterrows():
        pep = row['Stripped.Sequence']
        intensity = row[intensity_col]
        # Skip if intensity is NaN, None, or 0
        if pd.isna(intensity) or intensity == 0:
            continue
        # Find all occurrences of the peptide
        matches = list(re.finditer(re.escape(pep), protein_seq))
        if not matches:
            continue
        valid_found = False
        for match in matches:
            start = match.start()
            # Apply tryptic rule if enabled
            if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                continue
            valid_found = True
            end = start + len(pep)
            # Intensity mapping
            peptide_match = peptides[peptides['Stripped.Sequence'] == pep]
            if peptide_match.empty:
                continue  # Skip if no matching peptide found
            pep_mean_intensity = peptide_match[intensity_col].values[0]
            z_val = z_scores[peptide_match.index[0]]  # Use index to get corresponding z-score
            for i in range(start, end):
                if residue_vals[i] is None:
                    residue_vals[i] = [z_val]
                else:
                    residue_vals[i].append(z_val)
            # PTM mapping (only if intensity is non-zero)
            if ptm_col and ptm_col in row:
                ptm_seq = row[ptm_col]
                if pd.notna(ptm_seq) and '(UniMod:' in ptm_seq:
                    cleaned_pep, mods = clean_and_find_mods(ptm_seq)
                    if cleaned_pep != pep:
                        continue
                    for rel_pos, unismod in mods:
                        abs_pos = start + rel_pos
                        if 0 <= abs_pos < seq_len:
                            if unismod not in ptm_positions:
                                ptm_positions[unismod] = set()
                            ptm_positions[unismod].add(abs_pos)
        if not valid_found:
            continue
    
    # Resolve overlaps for intensities
    for i in range(seq_len):
        if residue_vals[i]:
            if overlap_strategy == 'merge':
                residue_vals[i] = np.mean(residue_vals[i])
            elif overlap_strategy == 'highest':
                residue_vals[i] = np.max(residue_vals[i])
            elif overlap_strategy in ['none', 'last']:
                residue_vals[i] = residue_vals[i][-1]
            else:
                residue_vals[i] = np.mean(residue_vals[i])
        else:
            residue_vals[i] = None
    
    # Convert PTM sets to lists
    for k in ptm_positions:
        ptm_positions[k] = sorted(list(ptm_positions[k]))
    
    return residue_vals, ptm_positions

def generate_colormap(residue_vals, cmap_name='autumn', not_mapped_color='#d3d3d3'):
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

def extract_plddt_and_model(pdb_str, protein_seq):
    parser = PDBParser(QUIET=True)
    pdb_io = io.StringIO(pdb_str)
    structure = parser.get_structure('model', pdb_io)
    model_name = "Unknown Model"
    plddt_list = [None] * len(protein_seq)
    if 'HEADER' in pdb_str:
        header_match = re.search(r'HEADER\s+\S+\s+\S+\s+(.+?)\s+\d{2}', pdb_str, re.IGNORECASE)
        if header_match:
            model_name = header_match.group(1).strip()
    else:
        model_name = "AF-" + base_id + "-F1-model_v6" if 'base_id' in globals() else "AlphaFold Model"
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id[1] - 1
                    if 0 <= res_id < len(protein_seq):
                        b_factor = residue['CA'].get_bfactor()
                        plddt_list[res_id] = b_factor
                    #st.write(f"PDB residue: {res_id}, chain: {chain.id}, pLDDT: {b_factor}")  # Debug
    valid_plddt = [v for v in plddt_list if v is not None]
    mean_plddt = np.mean(valid_plddt) if valid_plddt else None
    return plddt_list, model_name, mean_plddt

def create_download_zip(protein_of_interest, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data=None, selected_df=None, protein_seq=None,apply_tryptic=None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(f"{protein_of_interest}_protein.pdb", pdb_str)
        for condition in conditions:
            peptide_csv = peptide_data[condition].to_csv(index=False)
            zipf.writestr(f"{protein_of_interest}_{condition}_peptides.csv", peptide_csv)
        
        cmap = colormaps[cmap_name]
        for condition in conditions:
            pml_content = f"load {protein_of_interest}_protein.pdb\nhide everything\nshow cartoon\ncolor gray90, all\nzoom\n"
            min_log, max_log = min_max_logs[condition]
            for i in range(seq_len):
                if residue_data[condition][i] is not None:
                    norm = (residue_data[condition][i] - min_log) / (max_log - min_log) if max_log > min_log else 0.5
                    color_hex = mcolors.rgb2hex(cmap(norm)[:3])
                    pml_content += f"color {color_hex}, resi {i+1}\n"
            # Add PTM spheres in PyMOL script

            if ptm_data and ptm_data[condition] and isinstance(ptm_data[condition], dict):
                for unismod, info in ptm_data[condition].items():
                    if info['selected']:
                        color_hex = info['color']
                        for pos in info['positions']:
                            resi = pos + 1
                            pml_content += f"pseudoatom ptm_{unismod}_{resi}, resi {resi} and name CA\n"
                            pml_content += f"show spheres, ptm_{unismod}_{resi}\n"
                            pml_content += f"set sphere_scale, 5.0, ptm_{unismod}_{resi}\n"
                            pml_content += f"color {color_hex}, ptm_{unismod}_{resi}\n"
            zipf.writestr(f"{protein_of_interest}_{condition}_pymol_script.pml", pml_content)
        
        for condition in conditions:
            fig_width = min(25, max(10, seq_len / 20))
            fig, ax = plt.subplots(figsize=(fig_width, 1), dpi=600)
            ax.add_patch(patches.Rectangle((0, 0), seq_len, 1, facecolor=not_mapped_color, edgecolor='none'))
            min_log, max_log = min_max_logs[condition]
            for i in range(seq_len):
                if residue_data[condition][i] is not None:
                    norm = (residue_data[condition][i] - min_log) / (max_log - min_log) if max_log > min_log else 0.5
                    ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=cmap(norm)[:3], edgecolor='none'))
            # --- Add PTM markers (overlay on linear plot) ---
            if ptm_data and condition in ptm_data and isinstance(ptm_data[condition], dict):
                for unimod, info in ptm_data[condition].items():
                    if info.get('selected'):
                        color_hex = info.get('color', '#ff0000')
                        for pos in info.get('positions', []):
                            ax.scatter(pos, 0.5, color=color_hex, s=60, edgecolors='black', linewidths=0.5, zorder=3)
                            ax.text(pos, 0.55, unimod, rotation=0, fontsize=10, ha='center', va='bottom')
            ax.set_xlim(0, seq_len)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel(f'Amino Acid Position ({condition})', fontsize=30)
            ax.tick_params(axis='x', labelsize=15)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpeg', dpi=600, bbox_inches='tight')
            img_buffer.seek(0)
            zipf.writestr(f"{protein_of_interest}_{condition}_linear.jpeg", img_buffer.read())
            plt.close(fig)
        
        # Add PTM positions CSV (inspired by first code)
        if selected_df is not None and protein_seq is not None:
            ptm_rows = []
            for idx, row in selected_df.iterrows():
                protein = row['Protein.Group']
                stripped = row['Stripped.Sequence'] if row['Stripped.Sequence'] else 'NA'
                ptm = row['PTM'] if 'PTM' in row and pd.notna(row['PTM']) else ''
                control_col = list(conditions.keys())[0]
                disease_col = list(conditions.keys())[1]
                control = row[control_col] if control_col in row else 'NA'
                disease = row[disease_col] if disease_col in row else 'NA'
                if ptm and '(UniMod:' in ptm:
                    cleaned, mods = clean_and_find_mods(ptm)
                    if cleaned != stripped:
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'Mismatch', 'NA', 'NA', 'NA'])
                        continue
                    matches = list(re.finditer(re.escape(cleaned), protein_seq))
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid_found = True
                        peptide_start = start + 1
                        peptide_end = start + len(cleaned)
                        for mod_pos, unismod in mods:
                            full_pos = start + mod_pos + 1
                            ptm_rows.append([protein, stripped, control, disease, ptm, full_pos, unismod, peptide_start, peptide_end])
                    if not valid_found:
                        for mod_pos, unismod in mods:
                            ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', unismod, 'NA', 'NA'])
                else:
                    matches = list(re.finditer(re.escape(stripped), protein_seq)) if stripped != 'NA' else []
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid_found = True
                        peptide_start = start + 1
                        peptide_end = start + len(stripped) if stripped != 'NA' else 'NA'
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'NA', 'NA', peptide_start, peptide_end])
                    if not valid_found:
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', 'NA', 'NA', 'NA'])

                    else:
                        matches = list(re.finditer(re.escape(stripped), protein_seq)) if stripped else []
                        valid_found = False
                        for match in matches:
                            start = match.start()
                            if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                                continue
                            valid_found = True
                            peptide_start = start + 1
                            peptide_end = start + len(stripped) if stripped else 'NA'
                            ptm_rows.append([protein, stripped, control, disease, ptm, 'NA', 'NA', peptide_start, peptide_end])
                        if not valid_found:
                            ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', 'NA', 'NA', 'NA'])
            
            ptm_df = pd.DataFrame(ptm_rows, columns=['Protein.Group', 'Stripped.Sequence', 'Control_Intensity', 'Disease_Intensity', 'PTM', 'PTM_position', 'UniMod_Type', 'Peptide_Start', 'Peptide_End'])
            ptm_csv = ptm_df.to_csv(index=False)
            zipf.writestr(f"{protein_of_interest}_modification_positions.csv", ptm_csv)
    
    zip_buffer.seek(0)
    return zip_buffer

def format_sequence_for_display(seq, residue_data, condition1_name, condition2_name, line_len=80, group=20):
    mapped_positions = set(i for i, v in enumerate(residue_data[condition1_name]) if v is not None)
    mapped_positions.update(i for i, v in enumerate(residue_data[condition2_name]) if v is not None)
    mapped_js = json.dumps(list(mapped_positions))
    lines = []
    seq_len = len(seq)
    for start in range(0, seq_len, line_len):
        end = min(start + line_len, seq_len)
        segment = seq[start:end]
        num_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 10px; color: #888;'>"
        seq_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 12px; line-height: 1.5;'>"
        if start > 0:
            num_line += f"<span style='position: relative;'><span style='margin-right: {group - 1}ch;'>{start}</span><span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
        for i in range(0, len(segment), group):
            pos = start + i + 1
            num_span = f"<span style='margin-right: {group - 1}ch;'>{pos}</span>" if i + group <= len(segment) else f"<span>{pos}</span>"
            num_line += f"<span style='position: relative;'>{num_span}<span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
            seq_segment = segment[i:i + group]
            segment_html = ""
            for j, aa in enumerate(seq_segment):
                abs_pos = start + i + j
                style = "cursor:pointer;color:blue;" if abs_pos in mapped_positions else "color:gray;"
                segment_html += f"<span class='aa' data-pos='{abs_pos}' style='{style}'>{aa}</span>"
            seq_line += f"<span>{segment_html}</span>"
        num_line += "</div>"
        seq_line += f"<span style='margin-left: auto;'>{end}</span></div>"
        lines.append(num_line + seq_line)
    seq_html = "<div id='seq-panel' style='padding:10px; background:#fafafa; border-radius:6px; border:1px solid #ddd;'>" + "".join(lines) + "</div>"
    return seq_html

def sequence_copy_component(seq):
    seq_escaped = seq.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <div style="display:flex; gap:10px; align-items:center;">
      <button id="copySeqBtn" style="padding:6px 10px; background:#2b8cff; color:white; border-radius:6px; border:none; cursor:pointer;">Copy sequence</button>
      <span id="copyMsg" style="color:green; font-size:13px; display:none;">Copied!</span>
    </div>
    <pre id="seqText" style="display:none;">{seq_escaped}</pre>
    <script>
      const btn = document.getElementById('copySeqBtn');
      const msg = document.getElementById('copyMsg');
      btn.addEventListener('click', () => {{
        const text = document.getElementById('seqText').innerText;
        navigator.clipboard.writeText(text).then(() => {{
          msg.style.display = 'inline';
          setTimeout(() => msg.style.display = 'none', 1500);
        }});
      }});
    </script>
    """
    return html

# --- Main App UI ---
html_content = """
<div style="position: relative; width: 100%; overflow: hidden; background-color: #1a1a2e; padding: 20px 0;">
    <h1 id="animated-title" style="font-family: 'Arial', sans-serif; font-size: 48px; color: #e94560; margin: 0; text-align: center; 
           background: linear-gradient(to right, #e94560, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
           position: relative; z-index: 1;">
        Peptide3D Mapper
    </h1>
    <div id="paint-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(to right, #00d4ff, #e94560, #ffffff); 
           z-index: 0; animation: paintEffect 2s ease-out forwards;">
    </div>
    <style>
        @keyframes paintEffect {
            0% { width: 0; }
            100% { width: 100%; opacity: 0; }
        }
        #animated-title {
            display: inline-block;
        }
        #paint-overlay {
            animation-fill-mode: forwards;
        }
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            setTimeout(() => {
                document.getElementById("paint-overlay").style.display = "none";
            }, 2000);
        });
    </script>
</div>
"""
st.components.v1.html(html_content, height=100)

st.markdown(
    """
    <p style='text-align: justify; font-size: 16px; color: #87CEEB;'>
   The Peptide3D Mapper is a web-based tool that visualizes peptide intensity data from proteomics experiments on 3D protein structures, highlighting post-translational modifications (PTMs) with 
   customizable annotations. Upload peptide CSV and FASTA files to compare conditions (e.g., control vs. disease) using z-score intensity scales, with options to apply tryptic cleavage rules and 
   handle protein isoforms. Automatically fetch AlphaFold structures with pLDDT confidence scores or upload custom PDB files. Explore residue-level differences through interactive 3D and linear 
   sequence views with clickable residue selection, customizable color schemes, and PhosphoSitePlus integration for PTM exploration. Export comprehensive outputs, including PDB files, PyMOL 
   scripts, peptide CSVs, linear plots, and PTM position data, for further analysis.
    </p>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'conditions_confirmed' not in st.session_state:
    st.session_state.conditions_confirmed = {'single': False, 'multiple': False}
if 'processed' not in st.session_state:
    st.session_state.processed = {'single': False, 'multiple': False}
if 'selected_residue' not in st.session_state:
    st.session_state.selected_residue = None
if 'ptm_enabled' not in st.session_state:
    st.session_state.ptm_enabled = False
if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}
if 'apply_tryptic' not in st.session_state:
    st.session_state.apply_tryptic = False
if 'pdb_source' not in st.session_state:
    st.session_state.pdb_source = 'AlphaFold'  # Default to AlphaFold
if 'uploaded_pdb' not in st.session_state:
    st.session_state.uploaded_pdb = None
if 'selected_conditions' not in st.session_state:
    st.session_state.selected_conditions = {'single': {}, 'multiple': []}
#creating two tabs
tab1, tab2 = st.tabs(["🔍 Qualitative Analysis ", "📊 Quantitative Analysis "])

with tab1:
    sub_tab1_1, sub_tab1_2 = st.tabs([" Single Condition", " Multiple Conditions"])
    st.markdown(
        """
        <p style='justify-content: center; font-size: 20px; color: #87CEEB;
        
        The Qualitative Analysis tab allows users to visualize peptide intensity data on 3D protein structures, focusing on the presence and locations of post-translational modifications (PTMs). 
        Users can upload peptide CSV and FASTA files, select proteins of interest, and customize PTM annotations. The tool fetches AlphaFold structures or accepts user-uploaded PDB files, 
        providing interactive 3D and linear sequence views for detailed exploration of residue-level modifications.
        </p>
        """,
        unsafe_allow_html=True
    )
    with sub_tab1_1:
        st.subheader("Single Condition")
        st.markdown(
            """
            <p style='justify-content: center; font-size: 18px; color: #FFD700;'>
            🚧 Coming Soon! The Qualitative Analysis tab will soon be available with exciting features to explore peptide intensity data on 3D protein structures. Stay tuned!
            </p>
            """,
            unsafe_allow_html=True
        )
        with sub_tab1_2:
            st.subheader("Multiple Conditions")
            st.markdown(
                """
                <p style='justify-content: center; font-size: 18px; color: #FFD700;'>
                🚧 Coming Soon! The Qualitative Analysis tab will soon be available with exciting features to explore peptide intensity data on 3D protein structures. Stay tuned!
                </p>
                """,
                unsafe_allow_html=True
            )
with tab2:
    sub_tab2_1, sub_tab2_2 = st.tabs([" Single Condition Quantitative Analysis", " Multiple Conditions Quantitative Analysis "])
    st.markdown(
        """
        <p style='justify-content: center; font-size: 20px; color: #87CEEB;

        The Quantitative Analysis tab enables users to compare peptide intensity data between two conditions (e.g., control vs. disease) on 3D protein structures. Users can upload peptide CSV and 
        FASTA files, define experimental conditions, and apply tryptic cleavage rules. The tool visualizes intensity differences using z-score scales, highlights PTMs, and provides interactive 
        3D and linear sequence views for in-depth analysis of residue-level changes between conditions.
        </p>
        """,
        unsafe_allow_html=True
    )
    with sub_tab2_1:
        st.subheader("Single Condition Quantitative Analysis")
        st.markdown(
            """
            <p style='justify-content: center; font-size: 18px; color: #FFD700;'>
            🚧 Coming Soon! The Quantitative Analysis tab will soon be available with exciting features to compare peptide intensity data on 3D protein structures. Stay tuned!
            </p>
            """,
            unsafe_allow_html=True
        )
        with sub_tab2_2:
            st.subheader("Multiple Conditions Quantitative Analysis")
            st.markdown(
                """
                <p style='justify-content: center; font-size: 18px; color: #FFD700;'>
                🚧 Coming Soon! The Quantitative Analysis tab will soon be available with exciting features to compare peptide intensity data on 3D protein structures. Stay tuned!
                </p>
                """,
                unsafe_allow_html=True
            )

# File upload
csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"], help="CSV with Protein.Group, Stripped.Sequence, PTM, and intensity columns")
fasta_file = st.file_uploader("Upload FASTA", type=["fasta"], help="FASTA with matching UniProt IDs")

if csv_file and fasta_file:
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    fasta_str = fasta_file.getvalue().decode("utf-8")
    fasta_handle = io.StringIO(fasta_str)
    seq_records = list(SeqIO.parse(fasta_handle, "fasta"))
    if not seq_records:
        st.error("No sequences found in FASTA file.")
        st.stop()

    # Extract UniMod IDs
    if 'PTM' in df.columns:
        all_unimods = set()
        for ptm_seq in df['PTM'].dropna():
            if '(UniMod:' in ptm_seq:
                matches = re.finditer(r'UniMod:(\d+)', ptm_seq, re.IGNORECASE)
                for match in matches:
                    all_unimods.add(match.group(1))
        st.session_state.all_unimods = sorted(list(all_unimods)) if all_unimods else []
    else:
        st.session_state.all_unimods = []

    # PTM selection
    if st.session_state.all_unimods:
        st.markdown("### Detected PTM UniMod IDs")
        st.write("The following UniMod IDs were detected in the PTM column. Select the ones to include:")
        if 'selected_unimods' not in st.session_state:
            st.session_state.selected_unimods = st.session_state.all_unimods.copy()
        selected_unimods = st.multiselect(
            "Select UniMod IDs to Include",
            options=st.session_state.all_unimods,
            default=st.session_state.selected_unimods,
            help="Choose the UniMod IDs you want to process for PTM annotation."
        )
        st.session_state.selected_unimods = selected_unimods
    else:
        st.session_state.selected_unimods = []

    # PTM and tryptic options
    has_ptm = bool(st.session_state.all_unimods)
    ptm_checkbox_disabled = not has_ptm
    # The 'key' parameter automatically syncs the checkbox with st.session_state.ptm_enabled
    st.checkbox(
        "Enable PTM Annotation", 
        disabled=ptm_checkbox_disabled, 
        value=False if ptm_checkbox_disabled else st.session_state.get('ptm_enabled', False),
        key="ptm_enabled" 
    )
    st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.apply_tryptic)

    # Condition setup
    intensity_cols = [c for c in df.columns if 'intensity' in c.lower()]
    if not 1 <= len(intensity_cols) <=4 :
        st.error(f"Expected 1 to 4 intensity columns, found: {len(intensity_cols)}")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        condition1_name = st.text_input("Name for Condition 1", value="Control")
        condition1_col = st.selectbox("Map Condition 1 to Column", intensity_cols, index=0)
    with col2:
        condition2_name = st.text_input("Name for Condition 2", value="Disease")
        condition2_col = st.selectbox("Map Condition 2 to Column", intensity_cols, index=0 if len(intensity_cols) < 2 else 1)
    
    if condition1_col == condition2_col:
        st.error("Intensity columns must be different.")
        st.stop()
    
    if st.button("Confirm Conditions", use_container_width=True):
        st.session_state.conditions_confirmed = True
        st.session_state.processed = False
        st.rerun()

    if st.session_state.conditions_confirmed:
        with st.container():
            st.info("✅ Conditions confirmed. Now select protein and options.")
            protein_options = sorted(df['Protein.Group'].unique())
            selected_protein = st.selectbox("Select Protein", protein_options)
            col3, col4 = st.columns([1, 1])
            with col3:
                combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"])
            with col4:
                overlap_strategy = st.selectbox("Overlap Strategy", ["none", "merge", "highest", "last"])
            
            # PDB source selection with hyperlinks
            st.markdown(
                f'<div style="text-align:left; margin-bottom:10px;">'
                f'<span style="font-size:16px; color:#FFFFFF;">Databases: </span>'
                f'<a href="https://alphafold.ebi.ac.uk/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">AlphaFold Database</a>'
                f' | '
                f'<a href="https://www.rcsb.org/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">RCSB PDB</a>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.session_state.pdb_source = st.selectbox(
                "Select PDB Source",
                ["AlphaFold", "Upload PDB"],
                help="Choose to fetch the structure from AlphaFold or upload a PDB file named as UniProtID.pdb"
            )
            if st.session_state.pdb_source == "Upload PDB":
                st.session_state.uploaded_pdb = st.file_uploader(
                    "Upload PDB File",
                    type=["pdb"],
                    help="Upload a PDB file named as {UniProt_ID}.pdb matching the selected protein"
                )
            
            if st.button("Process Protein", use_container_width=True):
                st.session_state.processed = True
                st.rerun()

        if st.session_state.processed:
            with st.container():
                st.info("🔄 Processing... (This may take a moment for PDB fetch or upload.)")
                base_id = selected_protein.split('-')[0]
                protein_seq = None
                for rec in seq_records:
                    parts = rec.id.split('|')
                    uniprot_candidate = None
                    if len(parts) >= 2:
                        if parts[0] in ['sp', 'tr']:
                            uniprot_candidate = parts[1]
                        else:
                            uniprot_candidate = parts[0]
                    else:
                        uniprot_candidate = rec.id.split()[0]
                    if uniprot_candidate == base_id:
                        protein_seq = str(rec.seq)
                        matched_header = rec.id
                        break
                if protein_seq is None:
                    st.info(f"No direct FASTA header match for {base_id}. Attempting peptide-based matching...")
                    peptides_unique = df[df['Protein.Group'] == selected_protein]['Stripped.Sequence'].dropna().unique().tolist()
                    if len(peptides_unique) == 0:
                        peptides_unique = df[df['Protein.Group'].str.contains(base_id)]['Stripped.Sequence'].dropna().unique().tolist()
                    best_count = -1
                    best_rec = None
                    for rec in seq_records:
                        rec_seq = str(rec.seq)
                        count = 0
                        for pep in peptides_unique:
                            if pep and pep in rec_seq:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_rec = rec
                    if best_count > 0 and best_rec is not None:
                        protein_seq = str(best_rec.seq)
                        matched_header = best_rec.id
                        st.info(f"Selected FASTA entry {matched_header} with {best_count} peptides matched.")
                    else:
                        if len(seq_records) == 1:
                            protein_seq = str(seq_records[0].seq)
                            matched_header = seq_records[0].id
                            st.info(f"No peptide matches found; using the single FASTA entry {matched_header}.")
                        else:
                            st.error("Protein sequence could not be unambiguously detected from FASTA (no header match and no peptide overlap).")
                            st.stop()

                seq_len = len(protein_seq)

                # Isoform handling
                isoforms = df[df['Protein.Group'].str.contains(selected_protein + r'(?:-\d+)?$', regex=True)]['Protein.Group'].unique()
                if len(isoforms) > 1 and combine_isoforms == "yes":
                    st.info("Isoforms Detected")
                    selected_groups = list(isoforms)
                elif len(isoforms) > 1 and combine_isoforms == "no":
                    selected_groups = st.multiselect("Select Isoforms", options=list(isoforms), default=list(isoforms))
                else:
                    selected_groups = list(isoforms)
                
                if not selected_groups:
                    st.error("No isoforms selected.")
                    st.stop()
                
                selected_df = df[df['Protein.Group'].isin(selected_groups)]
                conditions = {condition1_name: condition1_col, condition2_name: condition2_col}
                peptide_data = {}
                residue_data = {condition1_name: [None] * seq_len, condition2_name: [None] * seq_len}
                ptm_data = {condition1_name: {}, condition2_name: {}}
                min_max_logs = {}
                ptm_col = 'PTM' if st.session_state.ptm_enabled and has_ptm else None
                
                for condition, intensity_col in conditions.items():
                    residues, ptms = map_peptides_to_residues(
                        selected_df, protein_seq, intensity_col, overlap_strategy,
                        ptm_col, apply_tryptic=st.session_state.apply_tryptic
                    )
                    residue_data[condition] = residues
                    if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                        ptm_data[condition] = {um: pos for um, pos in ptms.items() if um in st.session_state.selected_unimods}
                    else:
                        ptm_data[condition] = ptms
                    covered = [v for v in residues if v is not None]
                    if not covered:
                        st.error(f"No peptides mapped for {condition}.")
                        st.stop()
                    min_max_logs[condition] = (min(covered), max(covered))
                    peptides = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                    peptide_data[condition] = peptides
                
                # PTM configuration with hyperlinks
                if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                    st.subheader("PTM Configuration")
                    st.write(f"Selected UniMods: {st.session_state.selected_unimods}")
                    if not st.session_state.selected_unimods:
                        st.warning("No selected UniMod annotations for this protein.")
                    else:
                        if 'ptm_configs' not in st.session_state or set(st.session_state.ptm_configs.keys()) != set(st.session_state.selected_unimods):
                            st.session_state.ptm_configs = {um: {'selected': True, 'label': f"{um}", 'color': "#3700FF"} for um in st.session_state.selected_unimods}
                        # Debug to confirm selected UniMods
                        st.write(f"Rendering PTM config for UniMods: {st.session_state.selected_unimods}")
                        for um in st.session_state.selected_unimods:
                            col_ptm1, col_ptm2, col_ptm3 = st.columns([1, 1, 1])
                            with col_ptm1:
                                st.markdown(
                                    f'<a href="https://www.unimod.org/modifications_view.php?editid1={um}" target="_blank" style="color: #2b8cff; text-decoration: underline; font-weight: bold;">UniMod:{um}</a>',
                                    unsafe_allow_html=True
                                )
                                st.session_state.ptm_configs[um]['selected'] = st.checkbox(
                                    f"Include {um}", value=st.session_state.ptm_configs[um]['selected'], key=f"checkbox_{um}"
                                )
                            with col_ptm2:
                                st.session_state.ptm_configs[um]['label'] = st.text_input(
                                    f"Label for {um}", value=st.session_state.ptm_configs[um]['label'], key=f"label_{um}"
                                )
                            with col_ptm3:
                                st.session_state.ptm_configs[um]['color'] = st.color_picker(
                                    f"Color for {um}", value=st.session_state.ptm_configs[um]['color'], key=f"color_{um}"
                                )
                        for cond in ptm_data:
                            for um in list(ptm_data[cond].keys()):
                                if um in st.session_state.ptm_configs:
                                    ptm_data[cond][um] = {
                                        'positions': ptm_data[cond][um],
                                        'selected': st.session_state.ptm_configs[um]['selected'],
                                        'label': st.session_state.ptm_configs[um]['label'],
                                        'color': st.session_state.ptm_configs[um]['color']
                                    }
                                else:
                                    del ptm_data[cond][um]
                else:
                    ptm_data = {condition1_name: None, condition2_name: None}
                
                st.subheader("Detected Sequence")
                st.markdown(f"**FASTA header:** {matched_header}")
                copy_html = sequence_copy_component(protein_seq)
                seq_html = format_sequence_for_display(protein_seq, residue_data, condition1_name, condition2_name, line_len=150, group=20)
                st.components.v1.html(copy_html + seq_html, height=320)
                
                # PDB fetching or upload
                if st.session_state.pdb_source == "AlphaFold":
                    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                    with st.spinner(f"Attempting to fetch AlphaFold v6 structure for {base_id}..."):
                        try:
                            r = requests.get(pdb_url, timeout=30)
                            if r.status_code == 200:
                                pdb_str = r.text
                            elif r.status_code == 404:
                                st.error("Model_v6 not found. The protein may not have a v6 structure yet.")
                                st.stop()
                            else:
                                st.error(f"PDB fetch failed (status {r.status_code}).")
                                st.stop()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to fetch PDB for {base_id}: {str(e)}.")
                            st.stop()
                else:  # Upload PDB
                    if st.session_state.uploaded_pdb is None:
                        st.error("No PDB file uploaded.")
                        st.stop()
                    # Validate filename
                    pdb_filename = st.session_state.uploaded_pdb.name
                    if not pdb_filename.endswith('.pdb'):
                        st.error("Uploaded file must have a .pdb extension.")
                        st.stop()
                    filename_id = pdb_filename[:-4]  # Remove .pdb extension
                    if filename_id != base_id:
                        st.error(f"PDB filename ({pdb_filename}) must match the selected protein's UniProt ID ({base_id}).")
                        st.stop()
                    try:
                        pdb_str = st.session_state.uploaded_pdb.getvalue().decode("utf-8")
                    except Exception as e:
                        st.error(f"Error reading uploaded PDB file: {e}")
                        st.stop()
                
                st.success(f"Loaded {'AlphaFold' if st.session_state.pdb_source == 'AlphaFold' else 'uploaded'} structure for {base_id} ({len(pdb_str)} bytes)")
                plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
                st.info(f"**Mean pLDDT:** {mean_plddt_display} (Overall Confidence)")
                bg_color = st.selectbox("Background Color", ["black", "white", "darkgrey"], index=0)
                cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")


                st.subheader("PhosphoSitePlus®")
                try:
                    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{base_id}"
                    response = requests.get(uniprot_url, timeout=10)
                    if response.status_code == 200:
                        uniprot_data = response.json()
                        gene_name = uniprot_data.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown')
                    else:
                        gene_name = 'Unknown'
                except Exception as e:
                    gene_name = 'Unknown'
                    st.warning(f"Failed to fetch gene name for {base_id}: {e}")
                if gene_name != 'Unknown':
                    phosphosite_url = f"https://www.phosphosite.org/simpleSearchSubmitAction.action?searchStr={gene_name}"
                    st.markdown(
                        f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                        f'<a href="{phosphosite_url}" target="_blank" style="font-size:20px; color:#87CEEB; text-decoration:underline;">{base_id}|{gene_name}</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                        f'<span style="font-size:20px; color:#FFFFFF;">No gene name found for {base_id}. Unable to generate PhosphoSitePlus link.</span>',
                        unsafe_allow_html=True
                    )
                
                
                with st.container():
                    st.subheader("3D Structure Visualizations")

                    peptide_atlas_url = f"https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetProtein?atlas_build_id=592&protein_name={base_id}&action=QUERY"
                    st.markdown(
                        f'<div style="text-align:right; margin-bottom:10px;">'
                        f'<a href="{peptide_atlas_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">Explore Peptides of {base_id} in Peptide Atlas</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # Above "3D Structure Visualizations" header
                    alphafold_url = f"https://alphafold.ebi.ac.uk/search/text/{base_id}"
                    st.markdown(
                        f'<div style="text-align:right; margin-bottom:10px;">'
                        f'<a href="{alphafold_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">View {base_id} in AlphaFold Database</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # ===== NGL BIDIRECTIONAL SYNC INSERTION POINT =====
                    coverage1 = (sum(1 for v in residue_data[condition1_name] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                    coverage2 = (sum(1 for v in residue_data[condition2_name] if v is not None) / seq_len * 100) if seq_len > 0 else 0

                    hex_colors1, _, _ = generate_colormap(residue_data[condition1_name], selected_cmap, selected_not_mapped_color)
                    hex_colors2, _, _ = generate_colormap(residue_data[condition2_name], selected_cmap, selected_not_mapped_color)

                    st.markdown("#### 🧬 3D Viewer Options")
                    opt_col1, opt_col2 = st.columns([1, 1])
                    with opt_col1:
                        backbone_style_choice = st.radio(
                            "Protein backbone representation",
                            options=["🎀 Ribbon (Cartoon)", "⚛️ CPK / Hyperball (atomic)"],
                            index=0,
                            key="backbone_style_choice",
                            help="PTM sites always render as hyperball models regardless of this setting."
                        )
                        backbone_style = "cpk" if backbone_style_choice.startswith("⚛️") else "ribbon"
                    with opt_col2:
                        enable_surface = st.toggle("Render Solvent Accessible Surface", value=False, key="enable_surface_toggle")
                        surface_opacity = st.slider("Surface Opacity", 0.0, 1.0, 0.3, key="surface_opacity_slider") if enable_surface else 0.0

                    st.caption("🎯 Manual Structure Region Selector — pick any residue range to zoom into, independent of detected peptides. Applies to both conditions.")
                    if 'manual_zoom' not in st.session_state:
                        st.session_state.manual_zoom = None
                    mz_default_start = st.session_state.manual_zoom["start"] if st.session_state.manual_zoom else 1
                    mz_default_end = st.session_state.manual_zoom["end"] if st.session_state.manual_zoom else min(20, seq_len)
                    mz_col1, mz_col2, mz_col3, mz_col4 = st.columns([1, 1, 1, 1])
                    with mz_col1:
                        manual_start = st.number_input("Start residue", min_value=1, max_value=seq_len, value=min(mz_default_start, seq_len), key="manual_start_input")
                    with mz_col2:
                        manual_end = st.number_input("End residue", min_value=1, max_value=seq_len, value=min(mz_default_end, seq_len), key="manual_end_input")
                    with mz_col3:
                        if st.button("🔍 Zoom to Range", use_container_width=True):
                            lo, hi = sorted([int(manual_start), int(manual_end)])
                            st.session_state.manual_zoom = {"start": lo, "end": hi}
                            st.rerun()
                    with mz_col4:
                        if st.button("✕ Clear Zoom", use_container_width=True):
                            st.session_state.manual_zoom = None
                            st.rerun()

                    render_bidirectional_ngl_view(
                        pdb_str=pdb_str,
                        seq_len=seq_len,
                        protein_seq=protein_seq,
                        plddt_list=plddt_list,
                        model_name=model_name,
                        mean_plddt=mean_plddt,
                        cond1_name=condition1_name,
                        cond2_name=condition2_name,
                        zvals1=residue_data[condition1_name],
                        zvals2=residue_data[condition2_name],
                        colors1=hex_colors1,
                        colors2=hex_colors2,
                        ptm_data1=ptm_data[condition1_name],
                        ptm_data2=ptm_data[condition2_name],
                        coverage1=coverage1,
                        coverage2=coverage2,
                        bg_color=bg_color,
                        backbone_style=backbone_style,
                        enable_surface=enable_surface,
                        surface_opacity=surface_opacity,
                        manual_zoom=st.session_state.manual_zoom,
                    )
                    # ===== END NGL BIDIRECTIONAL SYNC INSERTION POINT =====

                    # Colorbar
                    st.subheader("Colorbar")
                    overall_vmin = min(min_max_logs[condition1_name][0], min_max_logs[condition2_name][0])
                    overall_vmax = max(min_max_logs[condition1_name][1], min_max_logs[condition2_name][1])
                    fig, ax = plt.subplots(figsize=(8, 0.3))
                    norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                    sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
                    cbar.set_label('Z-Score Intensity', fontsize=10)
                    cbar.ax.tick_params(labelsize=9)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=False)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    plt.close(fig)
                    html_content = f"""
                    <div style="display: flex; justify-content: center; align-items: center; width: 100%; max-width: 600px; margin: 10px auto;">
                        <img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 500px; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    """
                    st.components.v1.html(html_content, height=80)


                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Download Files (ZIP)", use_container_width=True):
                        zip_buffer = create_download_zip(selected_protein, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, selected_cmap, selected_not_mapped_color, ptm_data 
                                                        if st.session_state.ptm_enabled 
                                                            else None, selected_df 
                                                        if st.session_state.ptm_enabled 
                                                            else None, protein_seq 
                                                        if st.session_state.ptm_enabled 
                                                            else None,st.session_state.apply_tryptic)
                        st.download_button(
                            label="Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"{selected_protein}_files.zip",
                            mime="application/zip"
                        )
                with col_btn2:
                    if st.button("Reset & Re-Process", use_container_width=True):
                        st.session_state.clear() # Clear all session state
                        st.session_state.conditions_confirmed = False
                        st.session_state.processed = False
                        st.session_state.selected_residue = None
                        st.session_state.ptm_enabled = False
                        st.session_state.ptm_configs = {}
                        st.session_state.apply_tryptic = False
                        st.session_state.pdb_source = 'AlphaFold'
                        st.session_state.uploaded_pdb = None
                        st.session_state.all_unimods = []
                        st.session_state.selected_unimods = []
                        st.rerun()
