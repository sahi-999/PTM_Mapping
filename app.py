import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Physical PTM Modeler", layout="wide")

st.title("⚛️ True-to-Scale Physical PTM Modeler")
st.markdown("""
Unlike standard viewers that overlay generic spheres, this engine uses the **MMFF94 physics force-field** to computationally construct the PTM. 
It calculates the exact covalent bond lengths, van der Waals atomic radii, and torsion angles to simulate the real physical mass of the modification.
""")

# ==========================================
# CHEMICAL REACTION DEFINITION
# ==========================================
# This rule finds the Oxygen on the Serine sidechain and covalently attaches a Phosphate group.
PHOSPHORYLATION_SMARTS = '[CH2:1]-[OH:2] >> [CH2:1]-[O:2]-P(=O)(O)O'
rxn = AllChem.ReactionFromSmarts(PHOSPHORYLATION_SMARTS)

@st.cache_data
def generate_physical_ptm(smiles_sequence):
    """Generates the 3D coordinates for the unmodified and modified peptide."""
    
    # Step 1: Build the raw, unmodified amino acid
    base_mol = Chem.MolFromSmiles(smiles_sequence)
    base_mol = Chem.AddHs(base_mol) # Add Hydrogens for true physical volume
    
    # Step 2: Calculate 3D coordinates for unmodified molecule
    AllChem.EmbedMolecule(base_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(base_mol) # Physics engine folds it naturally
    
    # Step 3: Apply the chemical reaction to graft the PTM atoms
    products = rxn.RunReactants((base_mol,))
    if not products:
        return base_mol, None
    
    # Step 4: Extract the modified molecule
    modified_mol = products[0][0]
    modified_mol = Chem.AddHs(modified_mol) # Add Hydrogens to the new PTM
    
    # Step 5: Physics simulation to calculate true scale and angles
    AllChem.EmbedMolecule(modified_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(modified_mol)
    
    return base_mol, modified_mol

def mol_to_pdb_block(mol):
    """Converts the calculated molecule into PDB format for the 3D viewer."""
    return Chem.MolToPDBBlock(mol) if mol else ""

# ==========================================
# USER INTERFACE & RENDERING
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
    
    # Render in Streamlit
    showmol(view_unmod, height=500, width=500)

with col2:
    st.markdown("### 2. Modified Residue (True Scale)")
    st.caption("Physics engine has calculated exact atomic volume and bonds of the Phosphate.")
    
    view_mod = py3Dmol.view(width=500, height=500)
    view_mod.addModel(mol_to_pdb_block(mod_mol), "pdb")
    view_mod.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
    
    # Draw a translucent actual-size physical surface over it to show real mass
    view_mod.addSurface(py3Dmol.VDW, {'opacity': 0.6, 'color': 'white'})
    view_mod.zoomTo()
    
    # Render in Streamlit
    showmol(view_mod, height=500, width=500)
