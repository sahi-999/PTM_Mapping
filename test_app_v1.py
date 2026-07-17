"""
ptm_builder.py

Builds REAL 3D atoms for common post-translational modifications and grafts
them onto an existing protein structure (e.g. an AlphaFold model), using
internal-coordinate (NeRF) placement -- the same technique used to build
sidechains in homology modeling (Modeller, SCWRL, PyRosetta).

Given three existing atoms A-B-C and a target (bond length, bond angle,
dihedral) for a new atom D, NeRF computes D's real Cartesian position such
that the bond geometry is chemically correct. Chaining these lets us build
a whole modification group (e.g. phosphate: P + 3 O's) atom by atom, each
one referencing either the original residue atoms or previously-placed new
atoms.

SCOPE / HONESTY NOTE: this produces idealized, textbook bond geometry
(correct bond lengths/angles from standard chemistry references). It does
NOT perform steric clash relief or energy minimization -- a newly built
group could geometrically overlap a nearby loop. That would be a
legitimate v2 addition (e.g. a light local minimization restricted to the
new atoms only), not implemented here.

Reference for the placement algorithm: Parsons et al., "Practical
conversion from torsion space to Cartesian space for in silico protein
synthesis," J. Comput. Chem. 2005 (the "NeRF" method).
"""

import numpy as np


# ============================================================
# CORE GEOMETRY: NeRF internal-coordinate atom placement
# ============================================================

def place_atom(a, b, c, bond_length, bond_angle_deg, dihedral_deg):
    """
    Given 3 existing atom coordinates A, B, C (numpy arrays, shape (3,)),
    compute the Cartesian position of a new atom D such that:
      - the bond length C-D equals `bond_length`
      - the bond angle B-C-D equals `bond_angle_deg`
      - the dihedral angle A-B-C-D equals `dihedral_deg`

    This is the standard NeRF (Natural Extension Reference Frame) formula.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    bond_angle = np.radians(bond_angle_deg)
    dihedral = np.radians(dihedral_deg)

    bc = c - b
    bc_hat = bc / np.linalg.norm(bc)

    ab = b - a
    n = np.cross(ab, bc_hat)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6:
        # A-B is parallel to B-C (degenerate case) -- pick an arbitrary
        # perpendicular vector so the placement doesn't blow up.
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(bc_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = np.cross(arbitrary, bc_hat)
        n_norm = np.linalg.norm(n)
    n_hat = n / n_norm
    m_hat = np.cross(n_hat, bc_hat)

    d_local = np.array([
        -bond_length * np.cos(bond_angle),
        bond_length * np.sin(bond_angle) * np.cos(dihedral),
        bond_length * np.sin(bond_angle) * np.sin(dihedral),
    ])

    # Columns of M are the local frame's basis vectors expressed in the
    # global frame: bc_hat (x), m_hat (y), n_hat (z)
    M = np.array([bc_hat, m_hat, n_hat]).T
    d = c + M.dot(d_local)
    return d


# ============================================================
# PTM STRUCTURE LIBRARY
# ============================================================
# Each entry defines, per applicable residue type:
#   - "attach": name of the existing atom the modification attaches to
#   - "chain":  the 2 atoms immediately preceding the attach atom in the
#               existing residue (used as initial reference frame)
# ...and a shared "new_atoms" recipe: a list of atoms to build in order.
# Each new atom's "ref" is a 3-item list naming which 3 atoms (existing
# residue atoms, or names of previously-built new atoms in this same
# recipe) serve as A, B, C for its placement.
#
# Bond lengths (Angstrom) and angles (degrees) are standard textbook
# values for the relevant functional groups, not experimentally refined
# for this specific protein.

PTM_LIBRARY = {

    # ---------------- Phosphorylation (UNIMOD:21) ----------------
    "unimod:21": {
        "display_name": "Phospho",
        "residues": {
            "SER": {"attach": "OG",  "chain": ["CA", "CB"]},
            "THR": {"attach": "OG1", "chain": ["CA", "CB"]},
            "TYR": {"attach": "OH",  "chain": ["CE1", "CZ"]},
        },
        "new_atoms": [
            {"name": "P",   "element": "P", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.61, "angle": 119.0, "dihedral": 180.0},
            {"name": "O1P", "element": "O", "ref": ["chain1", "attach", "P"],
             "bond": 1.48, "angle": 109.5, "dihedral": 60.0},
            {"name": "O2P", "element": "O", "ref": ["chain1", "attach", "P"],
             "bond": 1.48, "angle": 109.5, "dihedral": 180.0},
            {"name": "O3P", "element": "O", "ref": ["chain1", "attach", "P"],
             "bond": 1.57, "angle": 109.5, "dihedral": 300.0},
        ],
    },

    # ---------------- Acetylation (UNIMOD:1) — Lys side chain ----------------
    "unimod:1": {
        "display_name": "Acetyl",
        "residues": {
            "LYS": {"attach": "NZ", "chain": ["CD", "CE"]},
        },
        "new_atoms": [
            {"name": "CH", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.33, "angle": 120.0, "dihedral": 180.0},
            {"name": "OH", "element": "O", "ref": ["chain1", "attach", "CH"],
             "bond": 1.23, "angle": 120.0, "dihedral": 0.0},
            {"name": "CH3", "element": "C", "ref": ["chain1", "attach", "CH"],
             "bond": 1.51, "angle": 116.0, "dihedral": 180.0},
        ],
    },

    # ---------------- Methylation (UNIMOD:34) — Lys side chain ----------------
    "unimod:34": {
        "display_name": "Methyl",
        "residues": {
            "LYS": {"attach": "NZ", "chain": ["CD", "CE"]},
        },
        "new_atoms": [
            {"name": "CM1", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 180.0},
        ],
    },

    # ---------------- Dimethylation (UNIMOD:36) — Lys side chain ----------------
    "unimod:36": {
        "display_name": "Dimethyl",
        "residues": {
            "LYS": {"attach": "NZ", "chain": ["CD", "CE"]},
        },
        "new_atoms": [
            {"name": "CM1", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 60.0},
            {"name": "CM2", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 180.0},
        ],
    },

    # ---------------- Trimethylation (UNIMOD:37) — Lys side chain ----------------
    "unimod:37": {
        "display_name": "Trimethyl",
        "residues": {
            "LYS": {"attach": "NZ", "chain": ["CD", "CE"]},
        },
        "new_atoms": [
            {"name": "CM1", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 60.0},
            {"name": "CM2", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 180.0},
            {"name": "CM3", "element": "C", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.47, "angle": 109.5, "dihedral": 300.0},
        ],
    },

    # ---------------- Oxidation (UNIMOD:35) — Met sulfur ----------------
    "unimod:35": {
        "display_name": "Oxidation",
        "residues": {
            "MET": {"attach": "SD", "chain": ["CB", "CG"]},
        },
        "new_atoms": [
            {"name": "OD1", "element": "O", "ref": ["chain0", "chain1", "attach"],
             "bond": 1.50, "angle": 106.0, "dihedral": 60.0},
        ],
    },
}


def resolve_ref_atom(name, existing_atoms, built_atoms, chain_names):
    """
    Resolves a reference-atom placeholder ('chain0', 'chain1', 'attach',
    or the literal name of a previously-built new atom) to a coordinate.
    """
    if name == "chain0":
        return existing_atoms[chain_names[0]]
    if name == "chain1":
        return existing_atoms[chain_names[1]]
    if name == "attach":
        return existing_atoms[chain_names[2]]
    if name in built_atoms:
        return built_atoms[name]
    if name in existing_atoms:
        return existing_atoms[name]
    raise KeyError(f"Cannot resolve reference atom '{name}'")


def build_ptm_atoms(unimod_id: str, residue_name: str, existing_atoms: dict):
    """
    Build real 3D coordinates for a PTM's new atoms on a specific residue.

    Args:
        unimod_id: e.g. "unimod:21"
        residue_name: 3-letter residue code, e.g. "SER"
        existing_atoms: dict of {atom_name: np.array([x,y,z])} for the
                         real atoms already present on that residue
                         (must include at least the attach atom + its
                         2 preceding chain atoms)

    Returns:
        list of (atom_name, element, coord) tuples for the new atoms,
        or None if this PTM/residue combination isn't in the library
        (caller should fall back gracefully -- e.g. keep the old
        decorative marker -- rather than crash).
    """
    entry = PTM_LIBRARY.get(unimod_id.lower())
    if not entry:
        return None
    res_spec = entry["residues"].get(residue_name.upper())
    if not res_spec:
        return None

    chain_names = [res_spec["chain"][0], res_spec["chain"][1], res_spec["attach"]]
    for cn in chain_names:
        if cn not in existing_atoms:
            return None  # incomplete residue (missing expected atom) -- bail safely

    built_atoms = {}
    output = []
    for spec in entry["new_atoms"]:
        ref_a = resolve_ref_atom(spec["ref"][0], existing_atoms, built_atoms, chain_names)
        ref_b = resolve_ref_atom(spec["ref"][1], existing_atoms, built_atoms, chain_names)
        ref_c = resolve_ref_atom(spec["ref"][2], existing_atoms, built_atoms, chain_names)
        coord = place_atom(ref_a, ref_b, ref_c, spec["bond"], spec["angle"], spec["dihedral"])
        built_atoms[spec["name"]] = coord
        output.append((spec["name"], spec["element"], coord))

    return output


def supported_ptms():
    """Returns {unimod_id: {display_name, residues: [3-letter codes]}} for UI display."""
    return {
        uid: {"display_name": v["display_name"], "residues": sorted(v["residues"].keys())}
        for uid, v in PTM_LIBRARY.items()
    }
