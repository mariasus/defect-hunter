#!/usr/bin/env ovitos

"""
Identify and geometrically characterise stacking-fault tetrahedra (SFTs) in FCC atomic configurations.

Input:
    One or more OVITO-readable ``.cfg.gz`` atomic configurations passed as command-line arguments.
    Each filename must contain the iteration number.

Method:
    Select HCP atoms using CNA and group them using two clustering cutoffs. Candidate clusters are
    filtered according to size, connectivity, PCA shape, and association with stair-rod dislocations.
    Duplicate detections between the two clustering passes are removed.

Outputs:
    ``sft_map.tmp``: equivalent-volume distribution for each iteration.
    ``sft_nb.tmp``: number of identified SFTs for each iteration.
    ``sft_per_sft.tmp``: HCP atom count, equivalent volume, and equivalent edge length for each SFT.
    ``sft_av.tmp``: mean and SEM of the equivalent volume and edge length for each iteration.
"""

# ============================================================
# 1. Imports
# ============================================================
import re
import sys
from math import pi

import numpy as np
from ovito.data import CutoffNeighborFinder
from ovito.io import import_file
from ovito.modifiers import ClusterAnalysisModifier, CommonNeighborAnalysisModifier, DislocationAnalysisModifier, SelectParticleTypeModifier


# ============================================================
# 2. Principal-component analysis
# ============================================================
def pca(X):
    n, m = X.shape
    cov = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    singular = [np.sqrt((n-1)*evals[0]), np.sqrt((n-1)*evals[1]), np.sqrt((n-1)*evals[2])]
    plane = np.cross(evecs[0], evecs[1])
    return singular, plane


# ============================================================
# 3. Connectivity check
# ============================================================
def mostly_connected(cluster_indices, cluster_id, l_dis, data, cutoff_link, alpha=1.6, keep_frac=0.85):
    """Test whether an HCP cluster is dominated by one connected component."""

    N = len(cluster_indices)
    if N < 5:
        return True

    global_to_local = {g: i for i, g in enumerate(cluster_indices)}

    r_probe = max(1.5*cutoff_link, 1e-6)
    finder_probe = CutoffNeighborFinder(r_probe, data)
    nn = np.full(N, np.inf)

    for local_i, global_i in enumerate(cluster_indices):
        best2 = np.inf
        for neigh in finder_probe.find(int(global_i)):
            j = neigh.index
            if j == global_i:
                continue
            if l_dis[j] != cluster_id:
                continue
            d2 = np.dot(neigh.delta, neigh.delta)
            if d2 < best2:
                best2 = d2
        nn[local_i] = np.sqrt(best2) if np.isfinite(best2) else np.inf

    valid_nn = nn[np.isfinite(nn)]
    if len(valid_nn) < max(3, int(0.3*N)):
        return False

    r_conn = max(alpha*np.median(valid_nn), 0.9*cutoff_link)
    finder = CutoffNeighborFinder(r_conn, data)
    deg = np.zeros(N, dtype=int)

    for local_i, global_i in enumerate(cluster_indices):
        c = 0
        for neigh in finder.find(int(global_i)):
            j = neigh.index
            if j == global_i:
                continue
            if l_dis[j] == cluster_id:
                c += 1
        deg[local_i] = c

    start_local = int(np.argmax(deg))
    start_global = int(cluster_indices[start_local])

    seen = np.zeros(N, dtype=bool)
    stack = [start_global]
    seen[start_local] = True
    count = 1

    while stack:
        gi = stack.pop()
        for neigh in finder.find(int(gi)):
            gj = neigh.index
            if gj == gi:
                continue
            if l_dis[gj] != cluster_id:
                continue
            lj = global_to_local.get(gj, None)
            if lj is None or seen[lj]:
                continue
            seen[lj] = True
            count += 1
            stack.append(gj)

    return count/float(N) >= keep_frac

# ============================================================
# 4. Initialise timing
# ============================================================
from datetime import datetime
print("DATE AND TIME: ", datetime.now())

s_time = datetime.now()
start_time = datetime.now()

# ============================================================
# 5. Define analysis parameters
# ============================================================
trial_circuit_length = 14
circuit_stretchability = 9
defect_mesh_smoothing_level = 4
line_point_separation = 2.5

cutoff_clus_sft_small = 4.6
cutoff_clus_sft_big = 3.4
cutoff_thick = 1.0


# ============================================================
# 6. Build and sort the input file list
# ============================================================
newlist = []
readlist = []

for a in sys.argv[1:]:
    newlist.append(a.rsplit(".cfg.gz", 1)[0])

newlist.sort(key=lambda x: int(re.search(r"\d+", x).group()))

for a in newlist:
    readlist.append(a + ".cfg.gz")

# ============================================================
# 7. Initialise output files
# ============================================================

sft_map = open("sft_map.tmp", "w")
sft_map.write("# ITERATION\tV_EQ BIN LOWER EDGE\tSFT COUNT\n")

sft_nb = open("sft_nb.tmp", "w")
sft_nb.write("# ITERATION\tSFT COUNT\n")

sft_per_sft = open("sft_per_sft.tmp", "w")
sft_per_sft.write("# ITERATION\tHCP ATOM COUNT\tV_EQ\tL_EQ\n")

sft_av = open("sft_av.tmp", "w")
sft_av.write("# ITERATION\tSFT COUNT\tV_EQ MEAN\tV_EQ SEM\tL_EQ MEAN\tL_EQ SEM\n")

# ============================================================
# 8. Analyse SFTs in each configuration
# ============================================================
for file in readlist:
    iteration = re.search(r"\d+", file).group()
    print(file)
    
    # ============================================================
    # 8.1 Read the atomic configuration
    # ============================================================
    e_time = datetime.now()

    diff = (e_time-s_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Reading configuration')
    s_time = datetime.now()

    node = import_file(file)
    
    # ============================================================
    # 8.2 Extract the stair-rod dislocation network
    # ============================================================
    e_time = datetime.now()
    diff = (e_time-s_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Analysing dislocation network')
    s_time = datetime.now()

    # Extract stair-rod dislocations using DXA
    dislo = DislocationAnalysisModifier()
    dislo.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
    dislo.trial_circuit_length = trial_circuit_length
    dislo.circuit_stretchability = circuit_stretchability
    dislo.defect_mesh_smoothing_level = defect_mesh_smoothing_level
    dislo.line_point_separation = line_point_separation
    dislo.line_coarsening_enabled = True
    dislo.line_smoothing_enabled = True
    node.modifiers.append(dislo)
    data = node.compute()

    pos_dis = []
    network = node.output.dislocations

    for segment in network.segments:
        b_x = segment.true_burgers_vector[0]
        b_y = segment.true_burgers_vector[1]
        b_z = segment.true_burgers_vector[2]
        abs_x = abs(b_x)
        abs_y = abs(b_y)
        abs_z = abs(b_z)

        if (((abs_x <= 0.0001) and (abs(abs_y-0.1666) <= 0.0001) and (abs(abs_z-0.1666) <= 0.0001)) or
            ((abs_y <= 0.0001) and (abs(abs_x-0.1666) <= 0.0001) and (abs(abs_z-0.1666) <= 0.0001)) or
            ((abs_z <= 0.0001) and (abs(abs_y-0.1666) <= 0.0001) and (abs(abs_x-0.1666) <= 0.0001))):

            pts = np.asarray(segment.points)
            if len(pts) < 2:
                continue

            sr_len = np.sum(np.linalg.norm(pts[1:]-pts[:-1], axis=1))
            mean_dis = np.mean(pts, axis=0)
            pos_dis.append([segment.id, mean_dis[0], mean_dis[1], mean_dis[2], sr_len])

    # Select HCP atoms using CNA
    sft = CommonNeighborAnalysisModifier()
    node.modifiers.append(sft)
    selection = SelectParticleTypeModifier(property="Structure Type", types={CommonNeighborAnalysisModifier.Type.HCP})
    node.modifiers.append(selection)
    data = node.compute()

    sft_clus = []

    # ============================================================
    # 8.3 Identify small SFT candidates
    # ============================================================
    e_time = datetime.now()
    diff = (e_time-s_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Analysing small SFTs')
    s_time = datetime.now()
    
    # Identify small SFTs using the larger cutoff
    clus = ClusterAnalysisModifier(cutoff=cutoff_clus_sft_small, sort_by_size=True, only_selected=True, unwrap_particles=True)
    node.modifiers.append(clus)
    data = node.compute()
    l_dis = data.particle_properties["Cluster"].array

    if len(l_dis) > 0:
        for i in range(1, max(l_dis)+1):
            npart = len(np.where(l_dis == i)[0])

            if npart < 10:
                continue

            i_int = np.where(l_dis == i)[0]
            p = node.output.particle_properties["Position"].array[l_dis == i].copy()

            if npart > 50:
                if not mostly_connected(i_int, i, l_dis, data, cutoff_link=cutoff_clus_sft_small, alpha=1.6, keep_frac=0.85):
                    continue

            pa = np.mean(p, axis=0)
            pa_init = np.mean(p, axis=0)
            dist = max(np.linalg.norm(p-pa, axis=-1))

            if dist > max(node.output.cell.matrix[0])/2.0:
                continue
            if dist < 0.1:
                continue
            if dist > max(node.output.cell.matrix[0])/2.0:
                continue

            singular, plane = pca(p-pa)
            peri = 2.*pi*(0.5*(singular[0]**1.5+singular[1]**1.5))**(2./3.)
            ave_d = peri/pi

            if singular[2] < 4.:
                continue
            if singular[1]/singular[0] < 0.55:
                continue
            if singular[2]/singular[0] < 0.55:
                continue

            av_len = (singular[0]+singular[1]+singular[2])/3.
            sym = max(singular[0]/av_len, singular[1]/av_len, singular[2]/av_len)
            sym_2 = max(singular[0], singular[1], singular[2])/min(singular[0], singular[1], singular[2])

            nb_stair = 0
            for j in range(len(pos_dis)):
                diff = np.array(pos_dis[j][1:4])-pa_init
                d = np.linalg.norm(diff)
                sr_len = pos_dis[j][4]
                if (d < 0.9*av_len) and (abs(sym-sym_2) < cutoff_thick) and (sr_len > 0.30*av_len):
                    nb_stair += 1

            if nb_stair > 3:
                sft_clus.append([npart, singular, pa_init, i])

    # Remove the first clustering modifier
    del node.modifiers[3]

    # ============================================================
    # 8.4 Identify large SFT candidates
    # ============================================================
    e_time = datetime.now()
    diff = (e_time-s_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Analysing big SFTs')
    s_time = datetime.now()

    # Identify larger SFTs using the tighter cutoff
    clus = ClusterAnalysisModifier(cutoff=cutoff_clus_sft_big, sort_by_size=True, only_selected=True, unwrap_particles=True)
    node.modifiers.append(clus)
    data = node.compute()
    l_dis = data.particle_properties["Cluster"].array

    if len(l_dis) > 0:
        for i in range(1, max(l_dis)+1):
            npart = len(np.where(l_dis == i)[0])

            if npart < 10:
                continue

            i_int = np.where(l_dis == i)[0]
            p = node.output.particle_properties["Position"].array[l_dis == i].copy()

            if npart > 50:
                if not mostly_connected(i_int, i, l_dis, data, cutoff_link=cutoff_clus_sft_big, alpha=1.6, keep_frac=0.85):
                    continue

            pa = np.mean(p, axis=0)
            pa_init = np.mean(p, axis=0)
            dist = max(np.linalg.norm(p-pa, axis=-1))

            if dist > max(node.output.cell.matrix[0])/2.0:
                continue
            if dist < 0.1:
                continue
            if dist > max(node.output.cell.matrix[0])/2.0:
                continue

            singular, plane = pca(p-pa)
            peri = 2.*pi*(0.5*(singular[0]**1.5+singular[1]**1.5))**(2./3.)
            ave_d = peri/pi

            if singular[2] < 4.0:
                continue
            if singular[1]/singular[0] < 0.55:
                continue
            if singular[2]/singular[0] < 0.55:
                continue

            av_len = (singular[0]+singular[1]+singular[2])/3.
            sym = max(singular[0]/av_len, singular[1]/av_len, singular[2]/av_len)
            sym_2 = max(singular[0], singular[1], singular[2])/min(singular[0], singular[1], singular[2])

            nb_stair = 0
            for j in range(len(pos_dis)):
                diff = np.array(pos_dis[j][1:4])-pa_init
                d = np.linalg.norm(diff)
                sr_len = pos_dis[j][4]
                if (d < 0.9*av_len) and (abs(sym-sym_2) < cutoff_thick) and (sr_len > 0.3*av_len):
                    nb_stair += 1

            if (singular[1]/singular[0] < 0.5) or (singular[2]/singular[0] < 0.5):
                continue

            # Remove duplicates between the clustering passes
            if nb_stair > 3:
                d_min = 100000.
                for k in range(len(sft_clus)):
                    d_min = min(d_min, np.linalg.norm(sft_clus[k][2]-pa_init))
                if d_min > 5.:
                    sft_clus.append([npart, singular, pa_init, i])

    # ============================================================
    # 8.5 Calculate and save equivalent SFT quantities
    # ============================================================
    Veq_list = []
    Leq_list = []

    if len(sft_clus) == 0:
        print(iteration, 0, f"{0.0:.4f}", f"{0.0:.4f}", file=sft_per_sft)

    pref = np.sqrt(2)/12*np.power(3/np.sqrt(6), 3)

    for i in range(len(sft_clus)):
        npart = sft_clus[i][0]
        h_max = sft_clus[i][1][0]
        h_med = sft_clus[i][1][1]
        h_min = sft_clus[i][1][2]

        Vmax = pref*h_max**3
        Vmed = pref*h_med**3
        Vmin = pref*h_min**3
        Veq = (Vmax*Vmed*Vmin)**(1.0/3.0) if (Vmax > 0 and Vmed > 0 and Vmin > 0) else 0.0
        Leq = (6*np.sqrt(2)*Veq)**(1.0/3.0) if Veq > 0 else 0.0

        Veq_list.append(Veq)
        Leq_list.append(Leq)

        print(iteration, npart, f"{Veq:.4f}", f"{Leq:.4f}", file=sft_per_sft)

    print(" ", file=sft_per_sft)
    sft_per_sft.flush()

    nb_tot = len(Veq_list)
    print(iteration, nb_tot, file=sft_nb)
    sft_nb.flush()

    # Save the equivalent-volume distribution
    dis_sfteq = np.histogram(Veq_list, bins=1000, range=(0., 1000000.))

    for i in range(300):
        print(iteration, dis_sfteq[1][i], dis_sfteq[0][i], file=sft_map)

    print(" ", file=sft_map)
    sft_map.flush()

    # Calculate mean values and standard errors
    N = len(Veq_list)
    Veq_mean = float(np.mean(Veq_list)) if N else 0.0
    Veq_sem = float(np.std(Veq_list, ddof=1)/np.sqrt(N)) if N > 1 else 0.0
    Leq_mean = float(np.mean(Leq_list)) if N else 0.0
    Leq_sem = float(np.std(Leq_list, ddof=1)/np.sqrt(N)) if N > 1 else 0.0

    print(iteration, N, Veq_mean, Veq_sem, Leq_mean, Leq_sem, file=sft_av)
    sft_av.flush()
    
    # ============================================================
    # 8.6 Report analysis and total elapsed times
    # ============================================================
    e_time = datetime.now()
    diff = (e_time-s_time)
    diff_tot =(e_time-start_time)
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Finished; total elapsed time ------>', diff_tot.total_seconds())
    s_time = datetime.now()
    
sft_map.close()
sft_nb.close()
sft_per_sft.close()
sft_av.close()
