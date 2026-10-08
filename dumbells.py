#!/usr/bin/env ovitos

"""
Identify dumbbell-type defects in FCC atomic configurations.

Input:
    One or more OVITO-readable ``.cfg.gz`` atomic configurations passed as
    command-line arguments. Each filename must contain the iteration number.

Method:
    Select atoms whose coordination number differs from 12 and cluster the
    selected atoms using a cutoff of 2.065 angstrom.

Outputs:
    ``dumbb_map.tmp``: complete cluster-size distribution for each iteration.
    ``dumbb_nb.tmp``: numbers of clusters containing two, three, and four atoms
    for each iteration.
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
cutoff_dumb = 3.0
cutoff_clus_dumb = 2.065

# ============================================================
# 5. Initialise output files
# ============================================================
dumb_map = open("dumbb_map.tmp", "w")
dumb_map.write("# ITERATION\tCLUSTER SIZE\tCLUSTER COUNT\n")

nb_dumb = open("dumbb_nb.tmp", "w")
nb_dumb.write("# ITERATION\tSIZE-2 CLUSTERS\tSIZE-3 CLUSTERS\tSIZE-4 CLUSTERS\n")

# ============================================================
# 6. Analyse dumbbell-type defects in each configuration
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
	# 6.2 Identify and cluster non-FCC atoms
	# ============================================================
	e_time = datetime.now()
	diff = (e_time-s_time)
	print('      elapsed ---------------------->', diff.total_seconds())
	print('Analysing dumbbells')
	s_time = datetime.now()
	
	# Select atoms outside a perfect FCC coordination environment
	node.modifiers.append(CoordinationNumberModifier(cutoff = cutoff_dumb, number_of_bins = 100))
	node.modifiers.append(SelectExpressionModifier(expression = "Coordination != 12"))
	
	# Cluster the selected atoms using the calibrated dumbbell cutoff
	dumb = ClusterAnalysisModifier(cutoff = cutoff_clus_dumb , only_selected = True, sort_by_size = True)
	node.modifiers.append(dumb)
	node.compute()
	data = node.compute()

	vect_line_decompte = np.bincount(np.bincount(node.output.particle_properties['Cluster'].array)[1:],minlength=250)

	# ============================================================
	# 6.3 Save cluster counts and size distributions
	# ============================================================
	for i in range(0,249): print(iteration,i,vect_line_decompte[i],end=" \n",file=dumb_map)
	print(" ",file=dumb_map)
	dumb_map.flush()
	
	# Save the numbers of clusters containing two, three, and four atoms
	print(iteration,vect_line_decompte[2],vect_line_decompte[3],vect_line_decompte[4],file=nb_dumb)
	nb_dumb.flush()	
	
	# ============================================================
	# 6.4 Report analysis and total elapsed times
	# ============================================================
	e_time = datetime.now()
	diff = (e_time-s_time)
	diff_tot =(e_time-start_time)
	print('      elapsed ---------------------->', diff.total_seconds())
	print('Finished; total elapsed time ------>', diff_tot.total_seconds())
	s_time = datetime.now()

dumb_map.close()
nb_dumb.close()
