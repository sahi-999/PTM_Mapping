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

def map_peptides_to_residues(df, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False):
    seq_len = len()
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
        matches = list(re.finditer(re.escape(pep), ))
        if not matches:
            continue
        valid_found = False
        for match in matches:
            start = match.start()
            # Apply tryptic rule if enabled
            if apply_tryptic and start > 0 and [start - 1] not in 'KR':
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

def extract_plddt_and_model(pdb_str, ):
    parser = PDBParser(QUIET=True)
    pdb_io = io.StringIO(pdb_str)
    structure = parser.get_structure('model', pdb_io)
    model_name = "Unknown Model"
    plddt_list = [None] * len()
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
                    if 0 <= res_id < len():
                        b_factor = residue['CA'].get_bfactor()
                        plddt_list[res_id] = b_factor
                    #st.write(f"PDB residue: {res_id}, chain: {chain.id}, pLDDT: {b_factor}")  # Debug
    valid_plddt = [v for v in plddt_list if v is not None]
    mean_plddt = np.mean(valid_plddt) if valid_plddt else None
    return plddt_list, model_name, mean_plddt

# In render_linear_plot function
def render_linear_plot(residue_vals, title, seq_len, vmin, vmax, model_name, plddt_list, mean_plddt,
                       cmap_name='viridis', not_mapped_color='#BEFDF9', highlight_residues=[], ptm_data=None):
    hex_colors, _, _ = generate_colormap(residue_vals, cmap_name, not_mapped_color)
    mapped = [i for i, v in enumerate(residue_vals) if v is not None]
    mapped_js = str(mapped)
    
    # Fixed pixel_per_res and total_width
    pixel_per_res = 2 # Fixed at 2px per residue for consistency
    total_width = max(1800, seq_len * pixel_per_res)  # Ensure minimum 1800px, scale with sequence
    bar_height = 40
    
    # Push the bar down by 70 pixels to leave room for lollipops at the top
    bar_y = 70
    
    label_offset = bar_y + bar_height + 15
    # Clean up the bottom whitespace
    total_height = label_offset + 30 
    
    bars = ""
    for i in range(seq_len):
        x = i * pixel_per_res
        color = hex_colors[i]
        width = pixel_per_res
        aa = [i] if i < len() else 'X'
        z_val = f"{residue_vals[i]:.2f}" if residue_vals[i] is not None else "N/A"
        tooltip = f"Pos {i+1} ({aa}): Z-Score={z_val}"
        is_mapped = i in mapped
        mapped_attr = 'True' if is_mapped else 'False'
        
        bars += f'<rect x="{x}" y="{bar_y}" width="{width}" height="{bar_height}" fill="{color}" '
        bars += f'stroke="#666" stroke-width="0.5" data-pos="{i}" data-mapped="{mapped_attr}" title="{tooltip}" />'

    ptm_lines = ""
    label_font_size = 10

    # Staggering logic for overlapping PTMs (replaces old ptm_y_offset logic entirely)
    if ptm_data:
        all_ptms = []
        for unismod, info in ptm_data.items():
            if info['selected']:
                for pos in info['positions']:
                    all_ptms.append({
                        'pos': pos, 
                        'label': info['label'], 
                        'color': info['color']
                    })
        
        all_ptms.sort(key=lambda item: item['pos'])
        
        for idx, ptm in enumerate(all_ptms):
            pos = ptm['pos']
            color = ptm['color']
            label = ptm['label']
            x = pos * pixel_per_res + (pixel_per_res / 2)
            
            stem_length = 25 if idx % 2 == 0 else 50
            ptm_top_y = bar_y - stem_length
            
            ptm_lines += f'<line x1="{x}" y1="{ptm_top_y}" x2="{x}" y2="{bar_y}" stroke="{color}" stroke-width="2"/>'
            ptm_lines += f'<circle cx="{x}" cy="{ptm_top_y}" r="6" fill="{color}" stroke="black" stroke-width="1.2"/>'
            ptm_lines += (
                f'<text x="{x}" y="{ptm_top_y - 4}" '
                f'font-size="{label_font_size}" text-anchor="middle" '
                f'fill="{color}" font-weight="bold">{label} ({pos+1})</text>'
            )

    label_step = max(1, int(50 / pixel_per_res))
    labels = ""
    for i in range(0, seq_len, label_step):
        x = i * pixel_per_res + (pixel_per_res / 2)
        labels += f'<text x="{x}" y="{label_offset}" font-size="12" text-anchor="middle" fill="#333">{i+1}</text>'
    
    mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
    title_html = f'<div style="text-align:center; font-size:18px; margin-bottom:5px; font-weight:bold; color:#87CEEB;">{title}<br><span style="font-size:12px; color:#87CEEB;">{model_name} | Mean pLDDT: {mean_plddt_display}</span></div>'
    svg = f'<svg width="{total_width + 20}" height="{total_height + 20}" style="overflow:visible; background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px;">{bars}{ptm_lines}{labels}</svg>'
    container_html = f'<div style="overflow-x:auto; max-width:100%; margin:10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{svg}</div>'
    
    js = f"""
    <script>
    const mapped = {mapped_js};
    const rects = document.querySelectorAll('rect[data-pos]');
    rects.forEach(el => {{
      const pos = parseInt(el.getAttribute('data-pos'));
      const isMapped = mapped.includes(pos);
      if (isMapped) {{
        el.style.cursor = 'pointer';
        el.addEventListener('click', e => {{
          const pos = parseInt(e.target.getAttribute('data-pos'));
          window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: pos }}, '*');
        }});
      }}
      el.addEventListener('mouseover', e => {{
        if (e.target.title) {{
          e.target.style.opacity = '0.7';
          e.target.style.strokeWidth = '1';
        }}
      }});
      el.addEventListener('mouseout', e => {{
        e.target.style.opacity = '1';
        e.target.style.strokeWidth = '0.5';
      }});
    }});
    </script>
    """
    html_output = title_html + container_html + js
    st.components.v1.html(html_output, height=200)
    return None, None

def add_ptm_spheres(viewer_idx, ptm_data, condition_name, view):
    if ptm_data:
        for unismod, info in ptm_data.items():
            if info['selected']:
                color = info['color']
                for pos in info['positions']:
                    resi_str = str(pos + 1)
                    spec = {
                        'center': {'resi': resi_str, 'atom': 'CA', 'chain': 'A'},
                        'radius': 2.0,  # Larger for visibility
                        'color': color,
                        'alpha': 1.0,
                        'zOffset': 10.0  # Push forward
                    }
                    try:
                        view.addSphere(spec, viewer=(0, viewer_idx))
                        #Fallback label
                        label_text = info.get('label', f"{unismod}") # Uses your custom label, or just the number
                        view.addLabel(label_text, {'fontSize': 12, 'fontColor': color, 'backgroundColor': 'white', 'backgroundOpacity': 0.5},
                                    {'resi': resi_str, 'chain': 'A'}, viewer=(0, viewer_idx))
                    except Exception as e:
                        st.error(f"Failed to add sphere for UniMod:{unismod} at {resi_str}: {e}")
        view.render()

def render_synced_viewers(pdb_str, residue_vals1, residue_vals2, bg_color, title1, title2, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data1=None, ptm_data2=None):
    # Use your original function to generate the exact Z-score hex colors
    hex_colors1, _, _ = generate_colormap(residue_vals1, cmap_name, not_mapped_color)
    hex_colors2, _, _ = generate_colormap(residue_vals2, cmap_name, not_mapped_color)
    
    bg_color_map = {'white': '#FFFFFF', 'black': '#000000', 'darkgrey': '#4A4A4A'}
    bg_color_hex = bg_color_map.get(bg_color.lower(), '#000000')
    
    # Package your exact data for the NGL Javascript engine
    js_data = {
        "pdb_str": pdb_str,
        "bg_color": bg_color_hex,
        "titles": [title1, title2],
        "colors": [hex_colors1, hex_colors2],
        "ptms": [ptm_data1 if ptm_data1 else {}, ptm_data2 if ptm_data2 else {}]
    }
    
    custom_viewer_html = """
    <style>
      .ngl-container { display: flex; width: 100%; gap: 10px; font-family: sans-serif; }
      .ngl-panel { flex: 1; height: 420px; position: relative; border: 1px solid #444; border-radius: 6px; overflow: hidden; }
      .ngl-title { position: absolute; top: 10px; left: 10px; z-index: 10; color: white; background: rgba(0,0,0,0.6); padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; }
    </style>
    
    <div class="ngl-container">
        <div id="viewport1" class="ngl-panel"><div class="ngl-title" id="title1"></div></div>
        <div id="viewport2" class="ngl-panel"><div class="ngl-title" id="title2"></div></div>
    </div>
    
    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script>
        const data = """ + json.dumps(js_data) + """;
        
        document.getElementById('title1').innerText = data.titles[0];
        document.getElementById('title2').innerText = data.titles[1];

        const stage1 = new NGL.Stage("viewport1", { backgroundColor: data.bg_color });
        const stage2 = new NGL.Stage("viewport2", { backgroundColor: data.bg_color });
        let comp1 = null; let comp2 = null;

        // 1. Sync Cameras across both panels
        let isUpdating = false;
        stage1.viewer.controls.addEventListener("change", () => {
            if (isUpdating) return;
            isUpdating = true;
            stage2.viewer.setOrientation(stage1.viewer.getOrientation());
            isUpdating = false;
        });
        stage2.viewer.controls.addEventListener("change", () => {
            if (isUpdating) return;
            isUpdating = true;
            stage1.viewer.setOrientation(stage2.viewer.getOrientation());
            isUpdating = false;
        });

        // 2. Create custom Z-score color schemes in NGL
        const schemeIds = [];
        data.colors.forEach((hexArr, idx) => {
            const pairs = [];
            hexArr.forEach((hex, i) => { pairs.push([hex, (i+1)+":A"]); });
            schemeIds.push(NGL.ColorMakerRegistry.addSelectionScheme(pairs, "scheme_" + idx));
        });

        // 3. Central Render Function (Handles Ghosting & Highlighting)
        function renderComp(comp, idx, highlightResi = null) {
            comp.removeAllRepresentations();
            const scheme = schemeIds[idx];
            const ptms = data.ptms[idx];

            if (highlightResi === null) {
                // Default State: Full Z-Score colored protein
                comp.addRepresentation("cartoon", { sele: "polymer", color: scheme, opacity: 1.0 });

                // Render all PTMs
                for (const [unimod, info] of Object.entries(ptms)) {
                    if (info.selected) {
                        info.positions.forEach(pos => {
                            comp.addRepresentation("spacefill", {
                                sele: (pos + 1) + ":A AND CA", color: info.color, scale: 2.0
                            });
                        });
                    }
                }
            } else {
                // Ghosted State: Fade everything grey, highlight just the selected residue
                const resiStr = highlightResi + ":A";
                comp.addRepresentation("cartoon", { sele: "polymer AND NOT " + resiStr, color: "#64748b", opacity: 0.15 });
                comp.addRepresentation("cartoon", { sele: resiStr, color: scheme, opacity: 1.0 });
                comp.addRepresentation("ball+stick", { sele: resiStr, color: scheme, scale: 2.0 });

                // Keep PTM visible if it's on the highlighted residue
                for (const [unimod, info] of Object.entries(ptms)) {
                    if (info.selected && info.positions.includes(highlightResi - 1)) {
                        comp.addRepresentation("spacefill", {
                            sele: highlightResi + ":A AND CA", color: info.color, scale: 3.0
                        });
                    }
                }
            }
        }

        // Load the PDB string
        const blob = new Blob([data.pdb_str], { type: 'text/plain' });
        Promise.all([
            stage1.loadFile(blob, { ext: 'pdb' }),
            stage2.loadFile(blob, { ext: 'pdb' })
        ]).then(([c1, c2]) => {
            comp1 = c1; comp2 = c2;
            renderComp(comp1, 0); renderComp(comp2, 1);
            stage1.autoView(); stage2.autoView();
        });

        // 4. Listen for clicks from your 1D Linear Sequence / SVG
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "SELECT_RESIDUE") {
                const resi = event.data.residue + 1; // Convert 0-based Python to 1-based PDB
                if (comp1 && comp2) {
                    renderComp(comp1, 0, resi);
                    renderComp(comp2, 1, resi);
                    // Zoom the camera to the residue
                    stage1.autoView(resi + ":A", 1000); 
                    // stage2 syncs automatically
                }
            }
        });

        // 5. Send clicks from 3D back to the 1D Sequence
        function handle3DClick(proxy) {
            if (proxy && proxy.atom) {
                window.parent.postMessage({ type: 'SELECT_RESIDUE', residue: proxy.atom.resno - 1 }, '*');
            } else {
                // Clicked background: reset zoom and ghosting
                renderComp(comp1, 0); renderComp(comp2, 1);
                stage1.autoView(1000);
            }
        }
        stage1.signals.clicked.add(handle3DClick);
        stage2.signals.clicked.add(handle3DClick);
    </script>
    """
    
    st.markdown(f"#### {title1} (Left) | {title2} (Right)")
    st.components.v1.html(custom_viewer_html, height=440)

def create_download_zip(protein_of_interest, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data=None, selected_df=None,apply_tryptic=None):
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
        if selected_df is not None and  is not None:
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
                    matches = list(re.finditer(re.escape(cleaned), ))
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and [start - 1] not in 'KR':
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
                    matches = list(re.finditer(re.escape(stripped), )) if stripped != 'NA' else []
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and [start - 1] not in 'KR':
                            continue
                        valid_found = True
                        peptide_start = start + 1
                        peptide_end = start + len(stripped) if stripped != 'NA' else 'NA'
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'NA', 'NA', peptide_start, peptide_end])
                    if not valid_found:
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', 'NA', 'NA', 'NA'])

                    else:
                        matches = list(re.finditer(re.escape(stripped), )) if stripped else []
                        valid_found = False
                        for match in matches:
                            start = match.start()
                            if apply_tryptic and start > 0 and [start - 1] not in 'KR':
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
    js = f"""
    <script>
    const mapped = {mapped_js};
    document.querySelectorAll('.aa').forEach(el => {{
        const pos = parseInt(el.getAttribute('data-pos'));
        if (mapped.includes(pos)) {{
            el.style.cursor = 'pointer';
            el.addEventListener('click', e => {{
                window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: pos }}, '*');
                console.log("Sequence click sent for pos:", pos);
            }});
        }}
    }});
    </script>
    """
    return seq_html + js

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

 # Extract UniMod IDs (Smarter Column Detection)
    # Automatically look for 'PTM' or 'Modified.Sequence' variations
    ptm_col_name = None
    for col in df.columns:
        if col.lower().replace(" ", "").replace(".", "") in ['ptm', 'modifiedsequence']:
            ptm_col_name = col
            break

    if ptm_col_name:
        all_unimods = set()
        for ptm_seq in df[ptm_col_name].dropna():
            # Made the UniMod search case-insensitive to catch 'unimod:' or 'UniMod:'
            if 'unimod:' in str(ptm_seq).lower():
                matches = re.finditer(r'unimod:(\d+)', str(ptm_seq), re.IGNORECASE)
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
                 = None
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
                         = str(rec.seq)
                        matched_header = rec.id
                        break
                if  is None:
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
                         = str(best_rec.seq)
                        matched_header = best_rec.id
                        st.info(f"Selected FASTA entry {matched_header} with {best_count} peptides matched.")
                    else:
                        if len(seq_records) == 1:
                             = str(seq_records[0].seq)
                            matched_header = seq_records[0].id
                            st.info(f"No peptide matches found; using the single FASTA entry {matched_header}.")
                        else:
                            st.error("Protein sequence could not be unambiguously detected from FASTA (no header match and no peptide overlap).")
                            st.stop()

                seq_len = len()

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
                        selected_df, , intensity_col, overlap_strategy,
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
                seq_html = format_sequence_for_display(, residue_data, condition1_name, condition2_name, line_len=150, group=20)
                copy_html = sequence_copy_component()
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
                plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, )
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
                    render_synced_viewers(pdb_str, residue_data[condition1_name], residue_data[condition2_name], bg_color, condition1_name, condition2_name, selected_cmap, selected_not_mapped_color, ptm_data[condition1_name], ptm_data[condition2_name])
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    
                    # Calculate coverage for each condition
                    coverage1 = (sum(1 for v in residue_data[condition1_name] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                    coverage2 = (sum(1 for v in residue_data[condition2_name] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                    
                    # Linear Sequence Visualizations
                    st.subheader("Linear Sequence Visualizations")
                    st.markdown(
                        f'<div style="font-size:16px; color:#87CEEB;margin-top:5px; margin-bottom:5px;">{condition1_name} (Coverage: {coverage1:.1f}%)</div>',
                        unsafe_allow_html=True
                    )
                    render_linear_plot(residue_data[condition1_name], condition1_name, seq_len,
                                      min_max_logs[condition1_name][0], min_max_logs[condition1_name][1], , model_name, plddt_list, mean_plddt, cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color, ptm_data=ptm_data[condition1_name])
                    st.markdown(
                        f'<div style="font-size:16px; color:#87CEEB; margin-top:5px;margin-bottom:5px;">{condition2_name} (Coverage: {coverage2:.1f}%)</div>',
                        unsafe_allow_html=True
                    )
                    render_linear_plot(residue_data[condition2_name], condition2_name, seq_len,
                                      min_max_logs[condition2_name][0], min_max_logs[condition2_name][1], , model_name, plddt_list, mean_plddt, cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color, ptm_data=ptm_data[condition2_name])
                    
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)


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
                                                            else None,  
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

