import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
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
import json
from Bio.PDB import PDBParser
import re
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Peptide3D Mapper", page_icon="⚛️", layout="wide")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def z_score(intensities):
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    return np.zeros_like(log_int) if std_log == 0 else (log_int - mean_log) / std_log

def clean_and_find_mods(peptide):
    mod_list = []
    cleaned_seq = ""
    index = -1
    pattern = re.compile(r"([A-Z])(\(UniMod:(\d+)\))?", re.IGNORECASE)
    for match in pattern.finditer(peptide):
        aa, mod, num = match.groups()
        index += 1
        cleaned_seq += aa
        if num:
            mod_list.append((index, num))
    return cleaned_seq, mod_list

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    ptm_positions = {}
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col])
    
    for idx, row in df.iterrows():
        pep = row.get('Stripped.Sequence', '')
        if not pep or pd.isna(pep): continue
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
            z_val = z_scores[peptide_match.index[0]]
            for i in range(start, end):
                if residue_vals[i] is None:
                    residue_vals[i] = [z_val]
                else:
                    residue_vals[i].append(z_val)
            if ptm_col and ptm_col in row:
                ptm_seq = row[ptm_col]
                if pd.notna(ptm_seq) and '(UniMod:' in str(ptm_seq):
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
        model_name = "AlphaFold Model"
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id[1] - 1
                    if 0 <= res_id < len(protein_seq):
                        b_factor = residue['CA'].get_bfactor()
                        plddt_list[res_id] = b_factor
    valid_plddt = [v for v in plddt_list if v is not None]
    mean_plddt = np.mean(valid_plddt) if valid_plddt else None
    return plddt_list, model_name, mean_plddt

def create_download_zip(protein_of_interest, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data=None, selected_df=None, protein_seq=None, apply_tryptic=None):
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
            
            if ptm_data and ptm_data[condition] and isinstance(ptm_data[condition], dict):
                for unismod, info in ptm_data[condition].items():
                    if isinstance(info, dict) and info.get('selected'):
                        color_hex = info['color']
                        for pos in info['positions']:
                            resi = pos + 1
                            pml_content += f"pseudoatom ptm_{unismod}_{resi}, resi {resi} and name CA\n"
                            pml_content += f"show spheres, ptm_{unismod}_{resi}\n"
                            pml_content += f"set sphere_scale, 5.0, ptm_{unismod}_{resi}\n"
                            pml_content += f"color {color_hex}, ptm_{unismod}_{resi}\n"
            zipf.writestr(f"{protein_of_interest}_{condition}_pymol_script.pml", pml_content)
        
        # Add PTM positions CSV safely across dynamic conditions
        if selected_df is not None and protein_seq is not None:
            ptm_rows = []
            for idx, row in selected_df.iterrows():
                protein = row['Protein.Group']
                stripped = row['Stripped.Sequence'] if row['Stripped.Sequence'] else 'NA'
                
                cond_names = list(conditions.keys())
                c1_name = cond_names[0]
                c2_name = cond_names[1] if len(cond_names) > 1 else c1_name
                
                # Dynamic PTM column detection by content
                ptm_col_name = None
                for col in row.index:
                    if pd.notna(row[col]) and isinstance(row[col], str) and 'unimod:' in row[col].lower():
                        ptm_col_name = col
                        break
                ptm = row[ptm_col_name] if ptm_col_name else ''
                
                control = row[conditions[c1_name]] if conditions[c1_name] in row else 'NA'
                disease = row[conditions[c2_name]] if conditions[c2_name] in row else 'NA'
                
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
            
            ptm_df = pd.DataFrame(ptm_rows, columns=['Protein.Group', 'Stripped.Sequence', 'Control_Intensity', 'Disease_Intensity', 'PTM', 'PTM_position', 'UniMod_Type', 'Peptide_Start', 'Peptide_End'])
            ptm_csv = ptm_df.to_csv(index=False)
            zipf.writestr(f"{protein_of_interest}_modification_positions.csv", ptm_csv)
    
    zip_buffer.seek(0)
    return zip_buffer

# ==========================================
# ADVANCED DYNAMIC NGL 3D/1D RENDERER 
# ==========================================
def render_bidirectional_ngl_view(pdb_str, seq_len, protein_seq, plddt_list, conditions_data, 
                                  bg_color, backbone_style, enable_surface, surface_opacity, manual_zoom):

    def prep_ptm_map(ptm_dict):
        arr = [None] * seq_len
        if ptm_dict and isinstance(ptm_dict, dict):
            for uid, info in ptm_dict.items():
                if isinstance(info, dict) and info.get('selected'):
                    for pos in info.get('positions', []):
                        if 0 <= pos < seq_len:
                            arr[pos] = {"name": info.get('label', uid), "color": info.get('color', '#ff0000')}
        return arr

    for c in conditions_data:
        c['ptm_arr'] = prep_ptm_map(c['ptms'])

    bg_color_map = {'white': '#FFFFFF', 'black': '#000000', 'darkgrey': '#4A4A4A'}
    bg_color_hex = bg_color_map.get(bg_color.lower(), '#0b0f19')

    js_data = {
        "pdb_str": pdb_str, "seq_len": seq_len, "protein_seq": protein_seq, "plddt": plddt_list,
        "conditions": conditions_data, "bg_color": bg_color_hex, "backbone_style": backbone_style,
        "enable_surface": bool(enable_surface), "surface_opacity": float(surface_opacity),
        "manual_zoom": manual_zoom
    }

    custom_viewer_html = """
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; }
        .ngl-container { display: flex; width: 100%; gap: 16px; position: relative; }
        .col-panel { flex: 1; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
        .panel-header { font-size: 13px; font-weight: bold; color: #38bdf8; background: #0f172a; padding: 6px 10px; border-radius: 6px; border: 1px solid #1e293b; display: flex; justify-content: space-between; }
        .viewport-ov { width: 100%; height: 380px; background-color: """ + bg_color_hex + """; border: 1px solid #1e293b; border-radius: 8px; position: relative; overflow: hidden; }
        .viewport-zm { width: 100%; height: 250px; background-color: """ + bg_color_hex + """; border: 1px solid #1e293b; border-radius: 8px; position: relative; overflow: hidden; }
        .vp-label { position: absolute; top: 6px; left: 8px; z-index: 10; color: #cbd5e1; font-size: 11px; background: rgba(15,23,42,0.7); padding: 2px 6px; border-radius: 4px; pointer-events: none; }
        .seq-container { height: 160px; overflow-y: auto; padding: 10px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; letter-spacing: 3px; line-height: 28px; font-family: monospace; font-size: 14px; word-break: break-all; user-select: none; }
        .aa-span { padding: 2px 4px; border-radius: 4px; cursor: pointer; position: relative; transition: all 0.1s; }
        .ptm-dot { position: absolute; top: -2px; left: 50%; transform: translateX(-50%); width: 6px; height: 6px; border-radius: 50%; border: 1px solid #fff; }
        .divider { width: 2px; background: #334155; border-radius: 1px; }
        #hover-hud { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(15,23,42,0.95); padding: 10px 20px; border-radius: 8px; color: white; font-size: 13px; z-index: 9999; display: none; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3); white-space: nowrap; }
        .hud-val { font-weight: bold; margin-left: 4px; }
        #selection-badge { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(244, 63, 94, 0.9); color: white; padding: 6px 12px; border-radius: 6px; cursor: pointer; z-index: 999; display: none; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    </style>

    <div style="position:relative;">
        <div id="selection-badge" onclick="clearSelection()">✕ Clear Selection</div>
        <div class="ngl-container" id="panels-grid"></div>
    </div>

    <div id="hover-hud">
        <div style="margin-bottom:4px; border-bottom:1px solid #475569; padding-bottom:4px;">
            <span style="color:#38bdf8; font-weight:bold;">Residue <span id="hud-resi">--</span></span>
            <span style="margin-left:15px; color:#cbd5e1;">pLDDT: <span id="hud-plddt" class="hud-val">--</span></span>
        </div>
        <div id="hud-scores"></div>
        <div id="hud-ptm" style="margin-top:4px; font-weight:bold; display:none;"></div>
    </div>

    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script>
        const data = """ + json.dumps(js_data) + """;
        const gridContainer = document.getElementById("panels-grid");
        
        let selectedSeg = null;
        const comps_ov = []; const comps_zm = [];
        const stages_ov = []; const stages_zm = [];
        const schemeIds = [];

        // Build Dynamic UI Elements
        data.conditions.forEach((cond, idx) => {
            const colPanel = document.createElement("div");
            colPanel.className = "col-panel";
            colPanel.innerHTML = `
                <div class="panel-header"><span>🔭 ${cond.name}</span> <span style="color:#94a3b8">Cov: ${cond.coverage.toFixed(1)}%</span></div>
                <div id="vp_${idx}_ov" class="viewport-ov"><div class="vp-label">Overview</div></div>
                <div id="vp_${idx}_zm" class="viewport-zm"><div class="vp-label">Zoomed Inset</div></div>
                <div id="seq_${idx}" class="seq-container"></div>
            `;
            gridContainer.appendChild(colPanel);
            
            if (idx < data.conditions.length - 1) {
                const div = document.createElement("div");
                div.className = "divider";
                gridContainer.appendChild(div);
            }
        });

        // Initialize 3D Viewers & Sequences
        data.conditions.forEach((cond, idx) => {
            const st_ov = new NGL.Stage("vp_" + idx + "_ov", {backgroundColor: data.bg_color});
            const st_zm = new NGL.Stage("vp_" + idx + "_zm", {backgroundColor: data.bg_color});
            stages_ov.push(st_ov); stages_zm.push(st_zm);

            const pairs = [];
            cond.colors.forEach((hex, i) => pairs.push([hex, (i+1)+":A"]));
            schemeIds.push(NGL.ColorMakerRegistry.addSelectionScheme(pairs, "zscore_scheme_" + idx));

            buildSeq("seq_" + idx, cond.colors, cond.ptm_arr, idx);
        });

        // Sync Cameras
        let isSync_ov = false, isSync_zm = false;
        stages_ov.forEach((st_ov, i) => {
            st_ov.viewer.controls.addEventListener("change", () => {
                if(isSync_ov) return;
                isSync_ov = true;
                stages_ov.forEach((other_st, j) => { if(i !== j) other_st.viewer.setOrientation(st_ov.viewer.getOrientation()); });
                isSync_ov = false;
            });
        });
        stages_zm.forEach((st_zm, i) => {
            st_zm.viewer.controls.addEventListener("change", () => {
                if(isSync_zm) return;
                isSync_zm = true;
                stages_zm.forEach((other_st, j) => { if(i !== j) other_st.viewer.setOrientation(st_zm.viewer.getOrientation()); });
                isSync_zm = false;
            });
        });

        function buildSeq(containerId, colors, ptmMap, condIdx) {
            const cont = document.getElementById(containerId);
            data.protein_seq.split('').forEach((letter, i) => {
                const span = document.createElement('span');
                span.innerText = letter; span.className = 'aa-span'; span.dataset.rn = i + 1;
                
                if (colors[i].toLowerCase() !== '#d3d3d3' && colors[i] !== data.bg_color) {
                    span.style.backgroundColor = colors[i] + '40'; 
                    span.style.color = '#000';
                    span.style.fontWeight = 'bold';
                    span.style.borderBottom = '3px solid ' + colors[i];
                } else {
                    span.style.color = '#94a3b8';
                }
                span.dataset.origBg = span.style.backgroundColor || 'transparent';
                span.dataset.origBorder = span.style.borderBottom || 'none';

                if (ptmMap[i]) {
                    const dot = document.createElement('div'); dot.className = 'ptm-dot';
                    dot.style.backgroundColor = ptmMap[i].color; span.appendChild(dot);
                }

                span.addEventListener('click', () => triggerSelect(i + 1));
                span.addEventListener('mouseenter', () => updateHUD(i));
                span.addEventListener('mouseleave', () => document.getElementById('hover-hud').style.display='none');
                cont.appendChild(span);
            });
        }

        function renderComp(comp, isZoom, condIdx) {
            comp.removeAllRepresentations();
            const schemeId = schemeIds[condIdx];
            const ptmMap = data.conditions[condIdx].ptm_arr;
            
            if (!selectedSeg) {
                if (isZoom) return;
                comp.addRepresentation(data.backbone_style === 'cpk' ? 'hyperball' : 'cartoon', { sele: "polymer", color: schemeId, opacity: 1.0 });
                if (data.enable_surface) comp.addRepresentation("surface", { sele: "polymer", color: "electrostatic", surfaceType: "sas", opacity: data.surface_opacity });
                ptmMap.forEach((ptm, i) => {
                    if (ptm) comp.addRepresentation("spacefill", { sele: (i+1)+":A AND .CA", color: ptm.color, scale: 0.8 });
                });
            } else {
                const selSele = selectedSeg.start + "-" + selectedSeg.end + ":A";
                const unselSele = "polymer AND NOT (" + selSele + ")";
                
                if (!isZoom) comp.addRepresentation(data.backbone_style === 'cpk' ? 'hyperball' : 'cartoon', { sele: unselSele, color: "#475569", opacity: 0.15 });
                
                comp.addRepresentation(data.backbone_style === 'cpk' ? 'hyperball' : 'cartoon', { sele: selSele, color: schemeId, opacity: 1.0 });
                if (data.backbone_style === 'ribbon') comp.addRepresentation("tube", { sele: selSele, color: schemeId, radius: 0.45 });

                for(let i = selectedSeg.start - 1; i < selectedSeg.end; i++) {
                    if (ptmMap[i]) {
                        comp.addRepresentation("hyperball", { sele: (i+1)+":A AND (sidechain OR .CA OR .C OR .N OR .O)", color: ptmMap[i].color, scale: 0.4 });
                        comp.addRepresentation("spacefill", { sele: (i+1)+":A AND .CA", color: ptmMap[i].color, scale: 0.6 });
                        if (isZoom) {
                            comp.addRepresentation("label", {
                                sele: (i+1)+":A AND .CA", labelText: ptmMap[i].name, color: ptmMap[i].color,
                                scale: 1.5, showBackground: true, backgroundColor: "black"
                            });
                        }
                    }
                }
            }
        }

        function applyReps() {
            comps_ov.forEach((c, idx) => renderComp(c, false, idx));
            comps_zm.forEach((c, idx) => renderComp(c, true, idx));
        }

        function updateHUD(i) {
            document.getElementById('hud-resi').innerText = data.protein_seq[i] + (i+1);
            document.getElementById('hud-plddt').innerText = (data.plddt[i] || 0).toFixed(1);
            
            let html = "";
            let ptmFound = null;
            data.conditions.forEach((cond, idx) => {
                const z = cond.zvals[i] !== null ? cond.zvals[i].toFixed(2) : 'N/A';
                html += `<span>${cond.name} Z-Score: <span class="hud-val" style="color:#10b981;">${z}</span></span>`;
                if (idx < data.conditions.length - 1) html += ` <span style="margin:0 10px; color:#475569;">|</span> `;
                if (cond.ptm_arr[i]) ptmFound = cond.ptm_arr[i];
            });
            document.getElementById('hud-scores').innerHTML = html;

            const ptmEl = document.getElementById('hud-ptm');
            if (ptmFound) { ptmEl.style.display = 'block'; ptmEl.innerHTML = `PTM: <span style='color:${ptmFound.color};'>${ptmFound.name}</span>`; }
            else { ptmEl.style.display = 'none'; }
            document.getElementById('hover-hud').style.display = 'block';
        }

        function triggerSelect(resi) {
            selectedSeg = {start: resi, end: resi};
            applyReps();
            comps_zm.forEach(c => c.autoView(resi+":A", 1000));
            
            document.querySelectorAll('.aa-span').forEach(span => {
                if (parseInt(span.dataset.rn) === resi) {
                    span.style.backgroundColor = '#ffe4e6';
                    span.style.borderBottom = '3px solid #e11d48';
                } else {
                    span.style.backgroundColor = span.dataset.origBg;
                    span.style.borderBottom = span.dataset.origBorder;
                }
            });
            document.getElementById("selection-badge").style.display = "block";
        }

        function clearSelection() {
            selectedSeg = null; applyReps();
            comps_ov.forEach(c => c.autoView(1000));
            document.querySelectorAll('.aa-span').forEach(span => {
                span.style.backgroundColor = span.dataset.origBg;
                span.style.borderBottom = span.dataset.origBorder;
            });
            document.getElementById("selection-badge").style.display = "none";
        }

        function handlePick(proxy) {
            if (proxy && (proxy.atom || proxy.bond)) {
                const atom = proxy.atom || proxy.bond.atom1;
                if (atom && atom.resno) triggerSelect(atom.resno);
            } else clearSelection();
        }

        // Load PDB
        const blob = new Blob([data.pdb_str], { type: 'text/plain' });
        const loadPromises = [];
        stages_ov.forEach(st => { st.signals.clicked.add(handlePick); loadPromises.push(st.loadFile(blob, {ext:'pdb'})); });
        stages_zm.forEach(st => { st.signals.clicked.add(handlePick); loadPromises.push(st.loadFile(blob, {ext:'pdb'})); });

        Promise.all(loadPromises).then(cs => {
            const half = cs.length / 2;
            for(let i=0; i<half; i++) { comps_ov.push(cs[i]); comps_zm.push(cs[i + half]); }
            
            if (data.manual_zoom && data.manual_zoom.start) {
                selectedSeg = {start: data.manual_zoom.start, end: data.manual_zoom.end};
                const sSele = selectedSeg.start + "-" + selectedSeg.end + ":A";
                setTimeout(() => { comps_zm.forEach(c => c.autoView(sSele, 1000)); }, 200);
            }
            applyReps();
            comps_ov.forEach(c => c.autoView());
        });
    </script>
    """
    
    components.html(custom_viewer_html, height=900)

# ==========================================
# MAIN APP UI 
# ==========================================
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
        #animated-title { display: inline-block; }
        #paint-overlay { animation-fill-mode: forwards; }
    </style>
</div>
"""
components.html(html_content, height=100)

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
    st.session_state.conditions_confirmed = False
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'selected_residue' not in st.session_state:
    st.session_state.selected_residue = None
if 'ptm_enabled' not in st.session_state:
    st.session_state.ptm_enabled = False
if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}
if 'apply_tryptic' not in st.session_state:
    st.session_state.apply_tryptic = False
if 'pdb_source' not in st.session_state:
    st.session_state.pdb_source = 'AlphaFold'
if 'uploaded_pdb' not in st.session_state:
    st.session_state.uploaded_pdb = None

tab1, tab2 = st.tabs(["🔍 Qualitative Analysis ", "📊 Quantitative Analysis "])

with tab1:
    sub_tab1_1, sub_tab1_2 = st.tabs([" Single Condition", " Multiple Conditions"])
    st.markdown("<p style='justify-content: center; font-size: 20px; color: #87CEEB;'>The Qualitative Analysis tab allows users to visualize peptide intensity data on 3D protein structures...</p>", unsafe_allow_html=True)
    with sub_tab1_1: st.markdown("<p style='color: #FFD700;'>🚧 Coming Soon!</p>", unsafe_allow_html=True)
    with sub_tab1_2: st.markdown("<p style='color: #FFD700;'>🚧 Coming Soon!</p>", unsafe_allow_html=True)

with tab2:
    sub_tab2_1, sub_tab2_2 = st.tabs([" Single Condition Quantitative Analysis", " Multiple Conditions Quantitative Analysis "])
    st.markdown("<p style='justify-content: center; font-size: 20px; color: #87CEEB;'>The Quantitative Analysis tab enables users to compare peptide intensity data...</p>", unsafe_allow_html=True)
    with sub_tab2_1: st.markdown("<p style='color: #FFD700;'>🚧 Coming Soon!</p>", unsafe_allow_html=True)
    with sub_tab2_2: st.markdown("<p style='color: #FFD700;'>🚧 Coming Soon!</p>", unsafe_allow_html=True)

# File upload
csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"])
fasta_file = st.file_uploader("Upload FASTA", type=["fasta"])

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

    # Automatically identify flexible PTM column names by searching data for "unimod:"
    ptm_col_name = None
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            sample = df[col].dropna().astype(str).head(30)
            if any('unimod:' in val.lower() for val in sample):
                ptm_col_name = col
                break
                
    if not ptm_col_name:
        for col in df.columns:
            if col.lower().replace(" ", "").replace(".", "") in ['ptm', 'modifiedsequence']:
                ptm_col_name = col
                break

    if ptm_col_name:
        all_unimods = set()
        for ptm_seq in df[ptm_col_name].dropna():
            if 'unimod:' in str(ptm_seq).lower():
                matches = re.finditer(r'unimod:(\d+)', str(ptm_seq), re.IGNORECASE)
                for match in matches:
                    all_unimods.add(match.group(1))
        st.session_state.all_unimods = sorted(list(all_unimods)) if all_unimods else []
    else:
        st.session_state.all_unimods = []

    # PTM selection
    if st.session_state.all_unimods:
        st.markdown(f"### Detected PTM UniMod IDs (from column `{ptm_col_name}`)")
        if 'selected_unimods' not in st.session_state:
            st.session_state.selected_unimods = st.session_state.all_unimods.copy()
        selected_unimods = st.multiselect("Select UniMod IDs to Include", options=st.session_state.all_unimods, default=st.session_state.selected_unimods)
        st.session_state.selected_unimods = selected_unimods
    else:
        st.session_state.selected_unimods = []

    has_ptm = bool(st.session_state.all_unimods)
    ptm_checkbox_disabled = not has_ptm
    st.checkbox("Enable PTM Annotation", disabled=ptm_checkbox_disabled, value=False if ptm_checkbox_disabled else st.session_state.get('ptm_enabled', False), key="ptm_enabled")
    st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.apply_tryptic)

    # Condition setup - Auto Detects Intensity Columns
    intensity_cols = [c for c in df.columns if 'intensity' in c.lower() or 'sample' in c.lower() or 'area' in c.lower()]
    if not intensity_cols:
        intensity_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    num_cols = min(len(intensity_cols), 4)
    if num_cols < 2:
        st.error("Expected at least 2 intensity columns for comparison.")
        st.stop()
        
    conditions = {}
    st.markdown(f"**Auto-detected {num_cols} conditions from data:**")
    cols = st.columns(num_cols)
    for i in range(num_cols):
        with cols[i]:
            guess_name = intensity_cols[i].replace("Intensity", "").replace("Area", "").replace("_", " ").strip()
            c_name = st.text_input(f"Name {i+1}", value=guess_name or f"Group {i+1}")
            c_col = st.selectbox(f"Column {i+1}", intensity_cols, index=i)
            conditions[c_name] = c_col

    if st.button("Confirm Conditions", use_container_width=True):
        st.session_state.conditions_confirmed = True
        st.session_state.processed = False
        st.rerun()

    if st.session_state.conditions_confirmed:
        with st.container():
            st.info("✅ Conditions confirmed. Now select protein and options.")
            protein_col = 'Protein.Group' if 'Protein.Group' in df.columns else df.columns[0]
            protein_options = sorted(df[protein_col].dropna().unique())
            selected_protein = st.selectbox("Select Protein", protein_options)
            col3, col4 = st.columns([1, 1])
            with col3: combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"])
            with col4: overlap_strategy = st.selectbox("Overlap Strategy", ["none", "merge", "highest", "last"])
            
            st.markdown(
                f'<div style="text-align:left; margin-bottom:10px;">'
                f'<span style="font-size:16px; color:#FFFFFF;">Databases: </span>'
                f'<a href="https://alphafold.ebi.ac.uk/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">AlphaFold Database</a> | '
                f'<a href="https://www.rcsb.org/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">RCSB PDB</a>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.session_state.pdb_source = st.selectbox("Select PDB Source", ["AlphaFold", "Upload PDB"])
            if st.session_state.pdb_source == "Upload PDB":
                st.session_state.uploaded_pdb = st.file_uploader("Upload PDB File", type=["pdb"])
            
            if st.button("Process Protein", use_container_width=True):
                st.session_state.processed = True
                st.rerun()

        if st.session_state.processed:
            with st.container():
                st.info("🔄 Processing... (This may take a moment)")
                base_id = selected_protein.split('-')[0].split(';')[0].strip()
                protein_seq = None
                for rec in seq_records:
                    if base_id in rec.id:
                        protein_seq = str(rec.seq).upper()
                        matched_header = rec.id
                        break
                if protein_seq is None:
                    protein_seq = str(seq_records[0].seq).upper()

                seq_len = len(protein_seq)
                isoforms = df[df[protein_col].str.contains(selected_protein + r'(?:-\d+)?$', regex=True, na=False)][protein_col].unique()
                if len(isoforms) > 1 and combine_isoforms == "yes":
                    selected_groups = list(isoforms)
                elif len(isoforms) > 1 and combine_isoforms == "no":
                    selected_groups = st.multiselect("Select Isoforms", options=list(isoforms), default=list(isoforms))
                else:
                    selected_groups = list(isoforms)
                
                selected_df = df[df[protein_col].isin(selected_groups)]
                
                peptide_data = {}
                residue_data = {c: [None] * seq_len for c in conditions}
                ptm_data = {c: {} for c in conditions}
                min_max_logs = {}
                ptm_col = ptm_col_name if st.session_state.ptm_enabled and has_ptm else None
                
                for condition, intensity_col in conditions.items():
                    residues, ptms = map_peptides_to_residues(selected_df, protein_seq, intensity_col, overlap_strategy, ptm_col, st.session_state.apply_tryptic)
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
                    peptide_data[condition] = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                
                if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                    st.subheader("PTM Configuration")
                    if 'ptm_configs' not in st.session_state or set(st.session_state.ptm_configs.keys()) != set(st.session_state.selected_unimods):
                        st.session_state.ptm_configs = {um: {'selected': True, 'label': f"{um}", 'color': "#3700FF"} for um in st.session_state.selected_unimods}
                    for um in st.session_state.selected_unimods:
                        col_ptm1, col_ptm2, col_ptm3 = st.columns([1, 1, 1])
                        with col_ptm1:
                            st.session_state.ptm_configs[um]['selected'] = st.checkbox(f"Include {um}", value=st.session_state.ptm_configs[um]['selected'], key=f"cb_{um}")
                        with col_ptm2:
                            st.session_state.ptm_configs[um]['label'] = st.text_input(f"Label for {um}", value=st.session_state.ptm_configs[um]['label'], key=f"lbl_{um}")
                        with col_ptm3:
                            st.session_state.ptm_configs[um]['color'] = st.color_picker(f"Color for {um}", value=st.session_state.ptm_configs[um]['color'], key=f"col_{um}")
                    for cond in ptm_data:
                        for um in list(ptm_data[cond].keys()):
                            if um in st.session_state.ptm_configs:
                                ptm_data[cond][um] = {
                                    'positions': ptm_data[cond][um],
                                    'selected': st.session_state.ptm_configs[um]['selected'],
                                    'label': st.session_state.ptm_configs[um]['label'],
                                    'color': st.session_state.ptm_configs[um]['color']
                                }
                
                if st.session_state.pdb_source == "AlphaFold":
                    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                    try:
                        r = requests.get(pdb_url, timeout=15)
                        if r.status_code == 200: pdb_str = r.text
                        else: st.error("Model_v6 not found."); st.stop()
                    except Exception as e: st.error(f"Fetch failed: {e}"); st.stop()
                else:  
                    if st.session_state.uploaded_pdb is None: st.error("No PDB file uploaded."); st.stop()
                    try: pdb_str = st.session_state.uploaded_pdb.getvalue().decode("utf-8")
                    except Exception as e: st.error(f"Read error: {e}"); st.stop()
                
                st.success(f"Loaded structure for {base_id} ({len(pdb_str)} bytes)")
                plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                st.info(f"**Mean pLDDT:** {mean_plddt:.1f}" if mean_plddt else "N/A")
                
                bg_color = st.selectbox("Background Color", ["black", "white", "darkgrey"], index=0)
                selected_cmap = st.selectbox("Select Color Gradient", ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'], index=0)
                selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")
                
                with st.container():
                    st.subheader("3D Structure Visualizations")
                    
                    # Prepare Conditions Data Dynamically
                    conditions_data = []
                    for c_name in conditions:
                        hex_colors, _, _ = generate_colormap(residue_data[c_name], selected_cmap, selected_not_mapped_color)
                        cov = (sum(1 for v in residue_data[c_name] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                        conditions_data.append({
                            "name": c_name,
                            "zvals": residue_data[c_name],
                            "colors": hex_colors,
                            "ptms": ptm_data[c_name],
                            "coverage": cov
                        })

                    st.markdown("#### 🧬 3D Viewer Options")
                    opt_col1, opt_col2 = st.columns([1, 1])
                    with opt_col1:
                        backbone_style_choice = st.radio("Backbone representation", ["🎀 Ribbon (Cartoon)", "⚛️ CPK / Hyperball (atomic)"], index=0)
                        backbone_style = "cpk" if backbone_style_choice.startswith("⚛️") else "ribbon"
                    with opt_col2:
                        enable_surface = st.toggle("Render Solvent Accessible Surface", value=False)
                        surface_opacity = st.slider("Surface Opacity", 0.0, 1.0, 0.3) if enable_surface else 0.0

                    if 'manual_zoom' not in st.session_state: st.session_state.manual_zoom = None
                    mz_col1, mz_col2, mz_col3, mz_col4 = st.columns([1, 1, 1, 1])
                    with mz_col1: manual_start = st.number_input("Start residue", 1, seq_len, min(st.session_state.manual_zoom["start"] if st.session_state.manual_zoom else 1, seq_len))
                    with mz_col2: manual_end = st.number_input("End residue", 1, seq_len, min(st.session_state.manual_zoom["end"] if st.session_state.manual_zoom else 20, seq_len))
                    with mz_col3:
                        if st.button("🔍 Zoom Range", use_container_width=True): st.session_state.manual_zoom = {"start": sorted([int(manual_start), int(manual_end)])[0], "end": sorted([int(manual_start), int(manual_end)])[1]}; st.rerun()
                    with mz_col4:
                        if st.button("✕ Clear Zoom", use_container_width=True): st.session_state.manual_zoom = None; st.rerun()

                    # Render Unified 1D + 3D Multi-Panel Viewer
                    render_bidirectional_ngl_view(
                        pdb_str=pdb_str,
                        seq_len=seq_len,
                        protein_seq=protein_seq,
                        plddt_list=plddt_list,
                        conditions_data=conditions_data,
                        bg_color=bg_color,
                        backbone_style=backbone_style,
                        enable_surface=enable_surface,
                        surface_opacity=surface_opacity,
                        manual_zoom=st.session_state.manual_zoom,
                    )

                    st.subheader("Colorbar")
                    overall_vmin = min(min_max_logs[c][0] for c in conditions)
                    overall_vmax = max(min_max_logs[c][1] for c in conditions)
                    fig, ax = plt.subplots(figsize=(8, 0.8)) # Fixed constraint height
                    norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                    sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
                    cbar.set_label('Z-Score Intensity', fontsize=10)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=False)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    plt.close(fig)
                    components.html(f'<div style="display:flex; justify-content:center;"><img src="data:image/png;base64,{img_str}" style="width:100%; max-width:500px;"></div>', height=120)

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Download Files (ZIP)", use_container_width=True):
                        zip_buffer = create_download_zip(selected_protein, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, selected_cmap, selected_not_mapped_color, ptm_data if st.session_state.ptm_enabled else None, selected_df if st.session_state.ptm_enabled else None, protein_seq if st.session_state.ptm_enabled else None, st.session_state.apply_tryptic)
                        st.download_button("Download ZIP", zip_buffer.getvalue(), f"{selected_protein}_files.zip", "application/zip")
                with col_btn2:
                    if st.button("Reset & Re-Process", use_container_width=True):
                        st.session_state.clear() 
                        st.rerun()
