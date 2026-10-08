#!/usr/bin/env python3

"""
Identify void regions and characterise their size and surface composition.

Input:
    One or more OVITO-readable ``.cfg.gz`` atomic configurations passed as
    command-line arguments. Each filename must contain the iteration number.

Method:
    Construct an alpha-shape surface mesh, identify empty spatial regions,
    determine the atoms associated with each void surface, and calculate the
    void volume, periodic centroid, surface composition, and enrichment relative
    to the bulk composition.

Outputs:
    ``void_volume_histogram.dat``: void-volume distribution and volume fraction.
    ``surface_atoms_vs_volume.dat``: number of surface atoms for each void.
    ``surface_composition_vs_volume.dat``: surface fractions and enrichments.
    ``void_stat_<iteration>.xyz``: optional void-centroid file for OVITO.
"""

# ============================================================
# 1. Imports
# ============================================================
import sys, re 
import numpy as np

from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier

# ============================================================
# 2. Initialise timing
# ============================================================
from datetime import datetime
print("DATE AND TIME: ", datetime.now())

s_time = datetime.now()
start_time = datetime.now()

# ============================================================
# 3. Build and sort the input file list
# ============================================================
newlist=list()
readlist=list()

for a in sys.argv[1:]: newlist.append(a.rsplit('.cfg.gz', 1)[0])
newlist.sort(key=lambda x: int(re.search(r'\d+', x).group()))
for a in newlist: readlist.append(a + ".cfg.gz")

# ============================================================
# 4. Define analysis parameters
# ============================================================
cutoff_surf_mesh = 1.9
write_void_xyz = False

# ============================================================
# 5. Initialise output files
# ============================================================

histo = open("void_volume_histogram.dat", "w")
histo.write("##LOOP NB\tVolume_A3\tNumber_of_voids\tRatio_tot_volume(%)\n")

sa_vol = open("surface_atoms_vs_volume.dat", "w")
sa_vol.write("##LOOP NB\tVolume_A3\tNsurface\n")

sc_vol = open("surface_composition_vs_volume.dat", "w")
sc_vol.write("##LOOP NB\tVolume_A3\tXFe\tXCr\tXNi\tRFe\tRCr\tRNi\n")

# ============================================================
# 6. Analyse void regions in each configuration
# ============================================================
for file in readlist:
	
    node = None
    iteration=re.search(r'\d+', file).group()
    print(file)

    # ============================================================
    # 6.1 Read the atomic configuration
    # ============================================================
    e_time = datetime.now()

    diff = (e_time-s_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Lecture')
    s_time = datetime.now()

    node = import_file(file)

    # ============================================================
    # 6.2 Construct the alpha-shape surface mesh
    # ============================================================
    e_time = datetime.now()
    diff = e_time-s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Constructing surface mesh')
    s_time = datetime.now()

    surface_mod = ConstructSurfaceModifier()

    surface_mod.method = ConstructSurfaceModifier.Method.AlphaShape
    surface_mod.radius = cutoff_surf_mesh       # to be adjusted
    surface_mod.identify_regions = True         # map objects empty/filled
    surface_mod.select_surface_particles = True # identify atoms at surface
    surface_mod.map_particles_to_regions = True # index the spatial regions each particle is located in

    node.modifiers.append(surface_mod) # Insert the modifier into the pipeline

    data = node.compute() # Evaluate the pipeline to compute results

    e_time = datetime.now()
    diff = e_time-s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Analysing void regions')
    s_time = datetime.now()

    # ============================================================
    # 6.3 Extract simulation-cell and particle data
    # ============================================================

    # Simulation cell
    box_lengths = np.diag(data.cell.matrix[:, :3])
    pbc_origin = data.cell.matrix[:, 3]  # origin of box cell

    Lx = box_lengths[0]
    Ly = box_lengths[1]
    Lz = box_lengths[2]

    # Get all type (ID -> Name)
    atom_types_property = data.particles['Particle Type']
    atom_positions = data.particles['Position'] # Array (N_atoms, 3)

    type_mapping = {t.id: t.name for t in atom_types_property.type_list} # Create a dictionnary {ID_numerical: "type_name"}

    # ============================================================
    # 6.4 Calculate the bulk chemical composition
    # ============================================================

    all_type_ids = np.asarray(atom_types_property)

    unique_ids, global_counts = np.unique(all_type_ids,return_counts=True)

    global_counts_dict = {type_mapping[tid]: count for tid, count in zip(unique_ids, global_counts)}

    N_total = len(all_type_ids)

    N_Fe_bulk = global_counts_dict.get('Fe', 0)
    N_Cr_bulk = global_counts_dict.get('Cr', 0)
    N_Ni_bulk = global_counts_dict.get('Ni', 0)

    X_Fe_bulk = N_Fe_bulk / N_total
    X_Cr_bulk = N_Cr_bulk / N_total
    X_Ni_bulk = N_Ni_bulk / N_total

    print("\n===================================================")
    print("GLOBAL CHEMICAL COMPOSITION")
    print("===================================================")

    print(f"Total atoms : {N_total}")
    print(f"Fe : {N_Fe_bulk:10d}   X = {X_Fe_bulk:.6f}")
    print(f"Cr : {N_Cr_bulk:10d}   X = {X_Cr_bulk:.6f}")
    print(f"Ni : {N_Ni_bulk:10d}   X = {X_Ni_bulk:.6f}")

    print("===================================================\n")

    # ============================================================
    # 6.5 Access the surface mesh and spatial regions
    # ============================================================
    memberships = data.attributes['ConstructSurfaceMesh.region_memberships']
    mesh = data.surfaces['surface']
    regions = mesh.regions

    # ============================================================
    # 6.6 Initialise storage for void statistics
    # ============================================================

    void_ids = []
    void_volumes = []
    void_nsurf = []

    void_centroid = []

    void_nfe = []
    void_ncr = []
    void_nni = []

    void_xfe = []
    void_xcr = []
    void_xni = []

    void_rfe = []
    void_rcr = []
    void_rni = []

    # ============================================================
    # 6.7 Identify and characterise individual voids
    # ============================================================
    for region_id, is_filled in enumerate(regions['Filled']):

        if is_filled:   # if region is filled: this is not a void, we pass
            continue

        void_volume = regions['Volume'][region_id] # Calculate volume of the void

        # ============================================================
        # 6.7.1 Identify atoms associated with the void surface
        # ============================================================

        # Get the raw neighborhood atoms from the mesh
        raw_atoms = np.asarray(memberships[region_id])

        if len(raw_atoms) == 0:
            continue

        # Keep only atoms whose official primary region ID matches the current region_id
        surface_atoms_of_void = np.unique(raw_atoms)

        if len(surface_atoms_of_void) == 0:
            continue

        # Atom coordinates
        void_atom_coords = atom_positions[surface_atoms_of_void]   # coordinates of surface atoms

        # Atom types
        void_atom_types_ids = np.asarray(atom_types_property[surface_atoms_of_void]) # Type of surface atoms
        void_atom_types_names = [type_mapping[tid] for tid in void_atom_types_ids]   # 

        # Count chemical species
        unique_types, counts = np.unique(void_atom_types_names, return_counts=True) # count type of atom per region
        type_distribution = dict(zip(unique_types, counts))

        n_fe = type_distribution.get('Fe', 0)
        n_cr = type_distribution.get('Cr', 0)
        n_ni = type_distribution.get('Ni', 0)

        n_surface = len(surface_atoms_of_void)

        # Chemical fractions at void surface
        if n_surface > 0:
            x_fe = n_fe / n_surface
            x_cr = n_cr / n_surface
            x_ni = n_ni / n_surface
        else:
            x_fe = 0.0
            x_cr = 0.0
            x_ni = 0.0

        # Enrichement relative to bulk stoichiometry
        r_fe = x_fe/X_Fe_bulk if X_Fe_bulk > 0 else 0.0
        r_cr = x_cr/X_Cr_bulk if X_Cr_bulk > 0 else 0.0
        r_ni = x_ni/X_Ni_bulk if X_Ni_bulk > 0 else 0.0

        # ============================================================
        # 6.7.2 Calculate the periodic centroid of the void
        # ============================================================

        # ALGORITHM for circular centroid, PBC 
        shifted_coords = void_atom_coords - pbc_origin        # 1. Translate coordinates to match [0, L]
        thetas = (shifted_coords / box_lengths) * 2.0 * np.pi # 2. Convert  coordinates in θi ([0, 2*pi])
        mean_cos = np.mean(np.cos(thetas), axis=0)            # 3. Calculate cosine average for each dimension (X, Y, Z)
        mean_sin = np.mean(np.sin(thetas), axis=0)            # 4. Calculate sine average for each dimension (X, Y, Z)
        mean_thetas = np.arctan2(-mean_sin, -mean_cos) + np.pi# 5. Calculate average angle with atan2
        shifted_centroid = (mean_thetas / (2.0 * np.pi)) * box_lengths  # 6. Convert back average angle into real coordinate (Å)
        centroid = shifted_centroid + pbc_origin  # 7. Convert back coordinates with center of boxcell in the middle

        # ============================================================
        # 6.7.3 Calculate surface-atom density relative to void volume
        # ============================================================

        nsurf_per_volume = (n_surface/void_volume if void_volume > 0 else 0.0)

        # ============================================================
        # 6.7.4 Store the void properties
        # ============================================================

        void_ids.append(region_id)

        void_volumes.append(void_volume)
        void_nsurf.append(n_surface)

        void_centroid.append(centroid)

        void_nfe.append(n_fe)
        void_ncr.append(n_cr)
        void_nni.append(n_ni)

        void_xfe.append(x_fe)
        void_xcr.append(x_cr)
        void_xni.append(x_ni)

        void_rfe.append(r_fe)
        void_rcr.append(r_cr)
        void_rni.append(r_ni)

        # ============================================================
        # 6.7.5 Report the individual void properties
        # ============================================================

        print(f"--- VOID {region_id} ---")
        print(f"Volume : {void_volume:.2f} Å³")
        print(f"Centroid : "f"[{centroid[0]:.3f}, "f"{centroid[1]:.3f}, "f"{centroid[2]:.3f}]")
        print(f"Surface atoms : {n_surface}")
        print(f"Composition : "f"Fe={x_fe:.4f}  "f"Cr={x_cr:.4f}  "f"Ni={x_ni:.4f}")
        print(f"Enrichment : "f"Fe={r_fe:.3f}  "f"Cr={r_cr:.3f}  "f"Ni={r_ni:.3f}")
        print()

    e_time = datetime.now()
    diff = e_time-s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Computing global void statistics')
    s_time = datetime.now()

    # ============================================================
    # 6.8 Calculate global void statistics
    # ============================================================

    # Convert to numpy array
    void_ids = np.asarray(void_ids)

    void_volumes = np.asarray(void_volumes)

    void_nsurf = np.asarray(void_nsurf)

    void_nfe = np.asarray(void_nfe)
    void_ncr = np.asarray(void_ncr)
    void_nni = np.asarray(void_nni)

    void_xfe = np.asarray(void_xfe)
    void_xcr = np.asarray(void_xcr)
    void_xni = np.asarray(void_xni)

    void_rfe = np.asarray(void_rfe)
    void_rcr = np.asarray(void_rcr)
    void_rni = np.asarray(void_rni)

    # ============================================================
    # 6.8.1 Classify voids by volume
    # ============================================================

    # Rejected: V < 33 Å^3
    rejected_volumes = void_volumes[void_volumes < 33.0]

    # Mono-vacancy: 33 <= V < 38 Å^3
    mono_volumes = void_volumes[(void_volumes >= 33.0) & (void_volumes < 38.0)]

    # Bi-vacancy: 58 <= V <= 75 Å^3
    bi_volumes = void_volumes[(void_volumes >= 58.0) & (void_volumes <= 75.0)]

    # Other: V > 75 Å^3
    other_volumes = void_volumes[void_volumes > 75.0]

    # Volumes between 38 and 58 Å^3 or safety
    intermediate_volumes = void_volumes[(void_volumes >= 38.0) & (void_volumes < 58.0)]

    # Total volume used for percentages
    accepted_volumes = np.concatenate([mono_volumes,bi_volumes,other_volumes])
    total_volume = np.sum(accepted_volumes)
    discarted_volume = np.sum(rejected_volumes) + np.sum(intermediate_volumes)

    mono_total = np.sum(mono_volumes)
    bi_total = np.sum(bi_volumes)
    other_total = np.sum(other_volumes)

    if total_volume > 0:
        pct_mono = 100.0 * mono_total / total_volume
        pct_bi   = 100.0 * bi_total / total_volume
        pct_other= 100.0 * other_total / total_volume
    else:
        pct_mono = pct_bi = pct_other = 0.0

    # ============================================================
    # 6.8.2 Report global void statistics
    # ============================================================

    print("===================================================")
    print("VOID STATISTICS")
    print("===================================================")

    print("\n===== VOID VOLUME STATISTICS =====")

    print(f"Number of voids : {len(void_ids)}")

    print(f"Rejected V < 33 Å³    : {len(rejected_volumes)} voids")
    print(f"Intermediate 38-58 Å³ : {len(intermediate_volumes)} voids")
    print(f"Discarted volume (<33 + 38-58 Å³): "f"{discarted_volume:.3f} Å³")
    print(f"Mono 33-38 Å³         : {len(mono_volumes)} mono-vacancies")
    print(f"Bi 58-75 Å³           : {len(bi_volumes)} bi-vacancies")
    print(f"Other V > 75 Å³       : {len(other_volumes)} other voids")

    if len(void_volumes) > 0:

        print(f"Minimum volume : "f"{np.min(void_volumes):.3f} Å³")
        print(f"Maximum volume : "f"{np.max(void_volumes):.3f} Å³")
        print(f"Mean volume    : "f"{np.mean(void_volumes):.3f} Å³")
        print(f"Median volume  : "f"{np.median(void_volumes):.3f} Å³")
        print(f"Total void volume : "f"{np.sum(void_volumes):.3f} Å³")

    print("\n===== VOLUME CONTRIBUTION =====")

    print(f"Total volume considered  : {total_volume:.3f} Å³")
    print(f"Mono volume               : {mono_total:.3f} Å³  "f"({pct_mono:.2f} %)")
    print(f"Bi volume                 : {bi_total:.3f} Å³  "f"({pct_bi:.2f} %)")
    print(f"Other volume              : {other_total:.3f} Å³  "f"({pct_other:.2f} %)")
    print(f"Sum                       : "f"{pct_mono + pct_bi + pct_other:.2f} %")

    e_time = datetime.now()
    diff = e_time-s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Writing output files')
    s_time = datetime.now()

    # ============================================================
    # 6.9 Write optional void-centroid files for OVITO
    # ============================================================

    if write_void_xyz:
        filename = f"void_stat_{iteration}.xyz"

        with open(filename, "w") as f:

            f.write(f"{len(void_volumes)}\n")

            f.write(f"{'region_id':<10} {'X':>12} {'Y':>12} {'Z':>12} "
                    f"{'Volume':>12} "
                    f"{'r_Fe':>8} {'r_Cr':>8} {'r_Ni':>8}\n")

            for i in range(len(void_volumes)):

                f.write(f"{void_ids[i]:<10d} "
                        f"{void_centroid[i][0]:12.4f} "
                        f"{void_centroid[i][1]:12.4f} "
                        f"{void_centroid[i][2]:12.4f} "
                        f"{void_volumes[i]:12.4f} "
        #               f"{void_nsurf[i]:8d} "
                        f"{void_rfe[i]:10.6f} "
                        f"{void_rcr[i]:10.6f} "
                        f"{void_rni[i]:10.6f}\n")


    # ============================================================
    # 6.10 Save the void-volume histogram
    # ============================================================

    if len(void_volumes) > 0:

        # Bin width in A^3
        bin_width = 10.0

        # Define bin edges
        vmin = np.floor(np.min(void_volumes)/bin_width) * bin_width
        vmax = np.ceil(np.max(void_volumes)/bin_width) * bin_width

        bins = np.arange(vmin,vmax + bin_width,bin_width)

        # Number of voids in each interval
        counts, edges = np.histogram(void_volumes,bins=bins)

        # Bin centers
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Su of the void volumes in each interval
        volume_sums = np.zeros(len(bin_centers))

        for i in range(len(bin_centers)):
            volume_sums[i] = np.sum(void_volumes[(void_volumes >= edges[i]) & (void_volumes < edges[i+1])])

        total_void_volume = np.sum(void_volumes)
        vol_pct = 100.0 * volume_sums / total_void_volume

        for volume, count, percent in zip(bin_centers,counts,vol_pct):
            print(f"{iteration}\t{volume:.6f}\t{count:d}\t{percent:.6f}", file=histo)
        

    # ============================================================
    # 6.11 Save the number of surface atoms versus void volume
    # ============================================================
        for volume, nsurf in zip(void_volumes,void_nsurf):
            print(f"{iteration}\t{volume:.6f}\t{nsurf:d}", file=sa_vol)

    # ============================================================
    # 6.12 Save surface composition and enrichment versus void volume
    # ============================================================
        for i in range(len(void_volumes)):

            print(
                f"{iteration}\t"
                f"{void_volumes[i]:.6f} "
                f"{void_xfe[i]:.6f} "
                f"{void_xcr[i]:.6f} "
                f"{void_xni[i]:.6f} "
                f"{void_rfe[i]:.6f} "
                f"{void_rcr[i]:.6f} "
                f"{void_rni[i]:.6f}",
                file=sc_vol
            )
                
    print(' ',file=histo)
    print(' ',file=sc_vol)
    print(' ',file=sa_vol)
    histo.flush()
    sc_vol.flush()
    sa_vol.flush()

    if write_void_xyz:
        print("\nStatistics of individual void written to:")
        print(f"  {filename}")
    print("\nVoid volume histogram written to:")
    print(f"void_volume_histogram.dat")
    print("\nSpecies at surface statistics written to:")
    print(f"surface_atoms_vs_volume.dat"f" surface_composition_vs_volume.dat")

    # ============================================================
    # 6.13 Report analysis and total elapsed times
    # ============================================================
    e_time = datetime.now()
    diff = e_time-s_time
    diff_tot = e_time-start_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Finished; total elapsed time ------>', diff_tot.total_seconds())
    print()
    s_time = datetime.now()

histo.close()
sc_vol.close()
sa_vol.close()
