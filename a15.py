#!/usr/bin/env ovitos

"""
Identify A15-related clusters in FCC atomic configurations.

Input:
    One or more OVITO-readable ``.cfg.gz`` atomic configurations passed as
    command-line arguments. Each filename must contain the iteration number.

Method:
    Apply Polyhedral Template Matching with an RMSD cutoff of 10.0, select
    atoms classified as icosahedral (ICO), and cluster the selected atoms
    using a cutoff of 3.0 angstrom.

Output:
    ``a15_map.tmp``: complete ICO-cluster size distribution for each
    iteration. The minimum cluster size used to identify A15 clusters is
    applied during subsequent post-processing.
"""

# ============================================================
# 1. Imports
# ============================================================
import re, sys
import os, glob

# OVITO modules
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *

# NumPy modules
import numpy as np
from numpy.linalg import eig, inv
import math
from math import *

import pprint
from collections import Counter

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
cutoff_poly = 10.
cutoff_cquinze = 3.0 

# ============================================================
# 5. Initialise the output file
# ============================================================
a15 = open("a15_map.tmp", "w")
a15.write("# ITERATION\tCLUSTER SIZE\tCLUSTER COUNT\n")

# ============================================================
# 6. Analyse A15-related clusters in each configuration
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
	print('Reading configuration')
	s_time = datetime.now()

	node = import_file(file)	

	# ============================================================
	# 6.2 Identify and cluster ICO atoms
	# ============================================================
	e_time = datetime.now()
	diff = (e_time-s_time)
	print('      elapsed ---------------------->', diff.total_seconds())
	print('Analysing A15 clusters')
	s_time = datetime.now()

	# Use a permissive RMSD cutoff to retain strongly distorted environments;
	# the structural classification is refined by the subsequent filters.
	poly = PolyhedralTemplateMatchingModifier(rmsd_cutoff = cutoff_poly)
	poly.structures[PolyhedralTemplateMatchingModifier.Type.OTHER].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
	poly.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = True
	node.modifiers.append(poly)
	node.compute()
	data = node.compute()
	n_ico = node.output.attributes['PolyhedralTemplateMatching.counts.ICO']
	n_bcc = node.output.attributes['PolyhedralTemplateMatching.counts.BCC']
	n_fcc = node.output.attributes['PolyhedralTemplateMatching.counts.FCC']
	n_hcp = node.output.attributes['PolyhedralTemplateMatching.counts.HCP']
	n_sc = node.output.attributes['PolyhedralTemplateMatching.counts.SC']
	n_other = node.output.attributes['PolyhedralTemplateMatching.counts.OTHER']

	# Select atoms with icosahedral coordination
	cquinze = SelectParticleTypeModifier(property = "Structure Type", types={PolyhedralTemplateMatchingModifier.Type.ICO})
	node.modifiers.append(cquinze)
	node.compute()
	data = node.compute()

	# Cluster the selected atoms and sort the clusters by size
	cquinze_cluster = ClusterAnalysisModifier(cutoff = cutoff_cquinze , only_selected = True, sort_by_size = True)
	node.modifiers.append(cquinze_cluster)
	node.compute()

	# ============================================================
	# 6.3 Save the ICO-cluster size distribution
	# ============================================================
	vect_line_decompte = np.bincount(np.bincount(node.output.particle_properties['Cluster'].array)[1:],minlength=250)
	
	for i in range(0,249): print(iteration,i,vect_line_decompte[i],end=" \n",file=g)
	print(" ",file=g)
	g.flush()

	# ============================================================
	# 6.4 Report analysis and total elapsed times
	# ============================================================
	e_time = datetime.now()
	diff = (e_time-s_time)
	diff_tot =(e_time-start_time)
	print('      elapsed ---------------------->', diff.total_seconds())
	print('Finished; total elapsed time ------>', diff_tot.total_seconds())
	s_time = datetime.now()
	
a15.close()
