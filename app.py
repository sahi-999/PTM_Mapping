import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="True Physical PTM Modeler", layout="wide")

st.title("⚛️ True-to-Scale Physical PTM Modeler")
st.markdown("""
This engine uses the **MMFF94 physics force-field** to computationally construct the PTM. 
It calculates the exact covalent bond lengths, van der Waals atomic radii, and torsion angles to simulate the real physical mass of the modification.
""")

# ==========================================
# CUSTOM NATIVE RENDERER
# ==========================================
def render_mol(view, height=500):
    """Renders a py3Dmol viewer natively in Streamlit, bypassing stmol entirely."""
    view_html = view._make_html()
    components.html(view_html, height=height)

# ==========================================
# CHEMICAL REACTION DEFINITION
# ==========================================
# FIX 1: We explicitly define the hydrogens on the new Phosphate oxygens ([OH1]) 
# so the physics engine doesn't miscalculate their valence bonds.
PHOSPHORYLATION_SMARTS = '[CH2:1]-[OH1:2] >> [CH2:1]-[O:2]-P(=O)([OH1])[OH1]'
rxn = AllChem.ReactionFromSmarts(PHOSPHORYLATION_SMARTS)

@st.cache_data
def generate_physical_ptm(smiles_sequence):
    """Calculates the 3D physics for the unmodified and modified peptide."""
    
    # Step 1: Build the raw, unmodified amino acid
    base_mol = Chem.MolFromSmiles(smiles_sequence)
    base_mol = Chem.AddHs(base_mol) # Add Hydrogens for true physical volume
    
    # Step 2: Calculate 3D coordinates using the physics engine
    AllChem.EmbedMolecule(base_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(base_mol) 
    
    # Step 3: Apply the chemical reaction to graft the PTM atoms
    products = rxn.RunReactants((base_mol,))
    if not products:
        return base_mol, None
    
    # Step 4: Extract the newly modified molecule
    modified_mol = products[0][0]
    
    # FIX 2: Sanitize the newly created molecule to recalculate correct chemical 
    # properties and ring valences BEFORE passing it to the 3D embedding engine.
    Chem.SanitizeMol(modified_mol)
    modified_mol = Chem.AddHs(modified_mol) 
    
    # Step 5: Physics simulation to calculate true scale and angles for the new PTM
    AllChem.EmbedMolecule(modified_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(modified_mol)
    
    return base_mol, modified_mol

def mol_to_pdb_block(mol):
    return Chem.MolToPDBBlock(mol) if mol else ""

# ==========================================
# USER INTERFACE
# ==========================================

# A simple Serine molecule SMILES (N-C-C backbone with -CH2OH sidechain)
test_smiles = "NCC(CO)C(=O)O" 

st.subheader("Simulating Phosphorylation on Serine")

with st.spinner("Calculating molecular physics and true scale..."):
    unmod_mol, mod_mol = generate_physical_ptm(test_smiles)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Unmodified Residue")
    st.caption("Standard predicted structure.")
    
    view_unmod = py3Dmol.view(width=500, height=500)
    view_unmod.addModel(mol_to_pdb_block(unmod_mol), "pdb")
    view_unmod.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}}) 
    view_unmod.zoomTo()
    
    render_mol(view_unmod, height=500)

with col2:
    st.markdown("### 2. Modified Residue (True Scale)")
    st.caption("Physics engine has calculated exact atomic volume and bonds of the Phosphate.")
    
    if mod_mol:
        view_mod = py3Dmol.view(width=500, height=500)
        view_mod.addModel(mol_to_pdb_block(mod_mol), "pdb")
        view_mod.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
        
        # Draw a translucent actual-size physical surface over it to show real mass
        view_mod.addSurface(py3Dmol.VDW, {'opacity': 0.6, 'color': 'white'})
        view_mod.zoomTo()
        
        render_mol(view_mod, height=500)
    else:
        st.error("The chemical reaction failed to process.")
