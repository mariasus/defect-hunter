#!/usr/bin/env ovitos

"""
Analyse DXA dislocation networks and HCP planar structures in FCC atomic configurations.

Input:
    One or more OVITO-readable ``.cfg.gz`` atomic configurations passed as command-line arguments.
    Each filename must contain the iteration number.

Method:
    Extract and classify the DXA network, reconstruct Perfect dislocations represented by paired
    Shockley paths, correct the dislocation statistics, identify the associated HCP ribbons, and
    classify the remaining HCP grains as stacking faults, loops, three-dimensional objects, or
    residual planar structures.

Outputs:
    ``PD_workfile2.tmp``: corrected segment-level dislocation network.
    ``PD_nano.tmp``: corrected type-resolved dislocation lengths.
    ``PD_dens-nano.tmp``: corrected type-resolved dislocation densities.
    ``PD_n-nano.tmp``: native closed-loop size distribution and line densities.
    ``PD.tmp``: geometry and topology of reconstructed dissociated Perfect dislocations.
    ``PD_bubbles.tmp``: accessory lines assigned to accepted dissociation ribbons.
    ``SF_hcp_grain_details.tmp``: grain-level HCP classification and properties.
    ``SF_hcp_planar_structures.tmp``: structure-level planar-defect properties.
    ``SF_discarded_grains.tmp``: HCP grains excluded from geometrical classification.
"""

# ============================================================
# 1. Imports
# ============================================================
import re
import sys

from collections import defaultdict, deque
from math import ceil, cos, pi, sqrt, radians, degrees, acos

import numpy as np

from ovito.io import import_file
from ovito.modifiers import (
    DislocationAnalysisModifier,
    GrainSegmentationModifier,
    PolyhedralTemplateMatchingModifier,
)
from ovito.data import CutoffNeighborFinder

# ============================================================
# 2. Define analysis parameters
# ============================================================

# DXA: same values used in dislo.py and sftcleanversion6.py
dxa_trial_circuit_length = 14
dxa_circuit_stretchability = 9
dxa_defect_mesh_smoothing = 4
dxa_line_point_separation = 2.5

# Polyhedral Template Matching cutoff
cutoff_poly = 0.1

# Reconstruction of graph nodes and dissociated-perfect ribbons
ribbon_min_width = 0.5           # Angstrom
ribbon_max_mean_width = 30.0     # Angstrom
ribbon_max_local_width = 50.0    # Angstrom
burgers_closure_tol = 3.0e-3

# If both the backbone and the additional bubble line are Shockley
# dislocations, select the backbone from the local tangent continuity.
# Differences smaller than this value are considered ambiguous.
bubble_geometry_tie_tol = 0.05

# Grain segmentation and regular HCP-grain identification
grain_min_size = 20
layer_min_atoms = 3
layer_min_atom_fraction = 0.01
layer_overlap_min = 0.75
boundary_edge_min_fraction = 0.50
outline_grid_dilation = 1

# Association of DXA line pieces with the boundary of a planar HCP patch.
boundary_plane_tol = 3.0         # Angstrom
boundary_edge_tol = 3.0          # Angstrom
boundary_min_coverage = 0.50
require_connected_boundary = False
require_closed_boundary = False

# Pairing of individual HCP grains belonging to the same physical structure.
parallel_angle_tol_deg = 10.0
monolayer_spacing_tolerance_d111 = 0.25
monolayer_overlap_min = 0.80
same_boundary_edge_fraction_min = 0.75

# ============================================================
# 3. General geometry and dislocation-network functions
# ============================================================

def pca(X):
    n, m =X.shape
    cov = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]    # directions of maximum variance in decreasing order (columns)
    evals = evals[idx]
    singular= [np.sqrt((n-1)*evals[0]), np.sqrt((n-1)*evals[1]), np.sqrt((n-1)*evals[2])]   # extension of the cloud along the principal axes
    normal = evecs[:, 2]  # normal to the plane defined by the two directions of maximum variance
    return singular, evecs, normal

def burgers_type(b):
    """
    Classify dislocations using the same convention as in dislo.py.

    PERFECT  = 1/2 <110>
    FRANK    = 1/3 <111>
    SHOCKLEY = 1/6 <112>
    STAIRROD = 1/6 <110>
    HIRTH    = 1/3 <001>
    """

    abs_x = abs(float(b[0]))
    abs_y = abs(float(b[1]))
    abs_z = abs(float(b[2]))

    tol = 0.001

    def close(value, reference):
        return abs(value - reference) <= tol

    zero_x = abs_x <= tol
    zero_y = abs_y <= tol
    zero_z = abs_z <= tol

    half_x = close(abs_x, 0.5)
    half_y = close(abs_y, 0.5)
    half_z = close(abs_z, 0.5)

    third_x = close(abs_x, 0.33333)
    third_y = close(abs_y, 0.33333)
    third_z = close(abs_z, 0.33333)

    sixth_x = close(abs_x, 0.16666)
    sixth_y = close(abs_y, 0.16666)
    sixth_z = close(abs_z, 0.16666)

    # Perfect: 1/2 <110>
    if (
        (zero_x and half_y and half_z) or
        (half_x and zero_y and half_z) or
        (half_x and half_y and zero_z)
    ):
        return 'PERFECT'

    # Frank: 1/3 <111>
    if third_x and third_y and third_z:
        return 'FRANK'

    # Shockley: 1/6 <112>
    if (
        (third_x and sixth_y and sixth_z) or
        (sixth_x and third_y and sixth_z) or
        (sixth_x and sixth_y and third_z)
    ):
        return 'SHOCKLEY'

    # Stair-rod: 1/6 <110>
    if (
        (zero_x and sixth_y and sixth_z) or
        (sixth_x and zero_y and sixth_z) or
        (sixth_x and sixth_y and zero_z)
    ):
        return 'STAIRROD'

    # Hirth: 1/3 <001>
    if (
        (third_x and zero_y and zero_z) or
        (zero_x and third_y and zero_z) or
        (zero_x and zero_y and third_z)
    ):
        return 'HIRTH'

    return 'OTHER'

def segment_records(network):
    records = []
    for segment in network.segments:
        points = np.asarray(segment.points, dtype=float).copy()     # coordinates of the points forming the dislocation
        if len(points) == 0:
            continue
        records.append({
            'id': int(segment.id),
            'length': float(segment.length),
            'b': np.asarray(segment.true_burgers_vector, dtype=float).copy(),
            'type': burgers_type(segment.true_burgers_vector),
            'points': points,
            'mean': np.mean(points, axis=0),
            'is_infinite': int(segment.is_infinite_line),
            'is_loop': int(segment.is_loop),
            # These four values are completed after graph construction and ribbon identification.
            'start_node': -1,
            'end_node': -1,
            'role': 'RAW',
            'dissociation_id': -1,
        })
    return records

def build_dislocation_graph(records, cell):
    """Join coincident segment endpoints and save segment connectivity."""
    node_positions = []
    node_edges = defaultdict(list)
    node_merge_tol = 0.5

    for edge_index, record in enumerate(records):
        segment_nodes = []

        for endpoint in (record['points'][0], record['points'][-1]):
            matching_node = None

            for node_id, node_position in enumerate(node_positions):
                distance = np.linalg.norm(cell.delta_vector(node_position, endpoint))
                if distance <= node_merge_tol:
                    matching_node = node_id
                    break

            if matching_node is None:
                node_positions.append(np.asarray(endpoint, dtype=float).copy())
                matching_node = len(node_positions) - 1

            segment_nodes.append(matching_node)

        record['start_node'] = segment_nodes[0]
        record['end_node'] = segment_nodes[1]
        node_edges[record['start_node']].append(edge_index)
        node_edges[record['end_node']].append(edge_index)

    return np.asarray(node_positions), node_edges


def unique_node_edges(node_edges, node):
    """Incident edge indices without duplicates (self-edges are stored twice)."""
    return list(dict.fromkeys(node_edges.get(node, [])))


def other_edge_node(edge, node, records):
    """Return the node at the other end of edge, or node itself for a self-edge."""
    rec = records[edge]
    if rec['start_node'] == node:
        return rec['end_node']
    if rec['end_node'] == node:
        return rec['start_node']
    raise ValueError('Edge {} is not incident to node {}'.format(edge, node))


def node_burgers_residual(node, edges, records):
    """Burgers-vector closure residual using every unique incident edge.

    A segment whose two endpoints have been merged into the same graph node
    enters and leaves that node, and therefore has zero net contribution.
    """
    bsum = np.zeros(3, dtype=float)
    for edge in list(dict.fromkeys(edges)):
        rec = records[edge]
        if rec['start_node'] == node and rec['end_node'] == node:
            continue
        if rec['start_node'] == node:
            bsum += rec['b']
        elif rec['end_node'] == node:
            bsum -= rec['b']
        else:
            raise ValueError('Edge {} is not incident to node {}'.format(edge, node))
    return float(np.linalg.norm(bsum))


def dissociation_anchor_kind(records, edges):
    """Return the effective anchor kind from its incident-edge topology."""
    unique_edges = list(dict.fromkeys(edges))
    perfect = [edge for edge in unique_edges if records[edge]['type'] == 'PERFECT']
    shockley = [edge for edge in unique_edges if records[edge]['type'] == 'SHOCKLEY']

    if len(unique_edges) == 3 and len(perfect) == 1 and len(shockley) == 2:
        return 'PERFECT'
    # A common terminal/connection vertex may carry any number of additional
    # dislocations.  It is an anchor as soon as at least two Shockley segments
    # meet there.  Continuation through the vertex is decided later and is
    # allowed only when exactly two unused Shockley arms remain.
    if len(unique_edges) >= 3 and len(shockley) >= 2:
        return 'CONNECTION'
    return None


def edge_tangent_from_node(edge, node, records, cell):
    """Unit tangent leaving node along edge, reconstructed through the PBC."""
    rec = records[edge]
    if rec['start_node'] == node and rec['end_node'] != node:
        points = rec['points'].copy()
    elif rec['end_node'] == node and rec['start_node'] != node:
        points = rec['points'][::-1].copy()
    else:
        return None

    if len(points) < 2:
        return None

    reference = points[0]
    for i in range(1, len(points)):
        point = align_point(points[i], reference, cell)
        delta = point - reference
        norm = np.linalg.norm(delta)
        if norm > 1.0e-10:
            return delta / norm
        reference = point
    return None


def identify_intermediate_bubbles(records, node_edges, cell):
    """Find additional dislocation lines attached to a Shockley backbone.

    Accepted motifs are deliberately restricted to:

      1. one self-edge attached to a node that becomes a two-Shockley
         intermediate node when the self-edge is removed;
      2. two direct edges between nodes A and B, one selected as the Shockley
         backbone and the other as the additional bubble line.  After removal
         of the bubble edge, A and B may both be two-Shockley intermediate
         nodes, or one may be intermediate while the other is a valid terminal
         PERFECT/CONNECTION anchor of the lens.
      3. one cross-link between two nodes which both become ordinary
         two-Shockley intermediate nodes after its removal.  Once a complete
         ribbon has been traced, this link is classified as a same-side bubble
         or as a CROSS_RUNG joining the two opposite Shockley sides.
      4. one terminal bridge joining two distinct ends of the paired Shockley
         sides.  After its removal each endpoint has exactly one Shockley arm.

    The returned edges are only provisional.  They are removed from corrected
    statistics only if a complete dissociated-perfect candidate containing
    them is subsequently accepted.
    """
    bubble_edges = set()
    bubble_by_node = defaultdict(list)
    bubble_info_by_edge = {}

    # Case 1: an additional segment starts and ends at the same graph node.
    for node in sorted(node_edges):
        incident = unique_node_edges(node_edges, node)
        self_edges = [
            edge for edge in incident
            if records[edge]['start_node'] == node
            and records[edge]['end_node'] == node
        ]
        if len(self_edges) != 1:
            continue

        bubble_edge = self_edges[0]
        remaining = [edge for edge in incident if edge != bubble_edge]
        if len(remaining) != 2:
            continue
        if any(records[edge]['type'] != 'SHOCKLEY' for edge in remaining):
            continue
        if node_burgers_residual(node, incident, records) > burgers_closure_tol:
            continue

        info = {
            'kind': 'SELF_NODE',
            'edge': bubble_edge,
            'node1': node,
            'node2': node,
            'backbone_edge': -1,
            'geometry_score': np.nan,
        }
        bubble_edges.add(bubble_edge)
        bubble_by_node[node].append(info)
        bubble_info_by_edge[bubble_edge] = info

    # Collect direct multi-edges between each unordered pair of graph nodes.
    pair_edges = defaultdict(list)
    for edge, rec in enumerate(records):
        node1 = rec['start_node']
        node2 = rec['end_node']
        if node1 == node2:
            continue
        pair_edges[tuple(sorted((node1, node2)))].append(edge)

    # Case 2: one Shockley backbone edge and one additional A--B edge.
    for (node1, node2), between_edges in sorted(pair_edges.items()):
        if len(between_edges) != 2:
            continue
        if node1 in bubble_by_node or node2 in bubble_by_node:
            continue

        incident1 = unique_node_edges(node_edges, node1)
        incident2 = unique_node_edges(node_edges, node2)
        if len(incident1) != 3 or len(incident2) != 3:
            continue

        external1 = [edge for edge in incident1 if edge not in between_edges]
        external2 = [edge for edge in incident2 if edge not in between_edges]
        if len(external1) != 1 or len(external2) != 1:
            continue
        if records[external1[0]]['type'] != 'SHOCKLEY':
            continue
        if records[external2[0]]['type'] != 'SHOCKLEY':
            continue
        if node_burgers_residual(node1, incident1, records) > burgers_closure_tol:
            continue
        if node_burgers_residual(node2, incident2, records) > burgers_closure_tol:
            continue

        shockley_between = [
            edge for edge in between_edges
            if records[edge]['type'] == 'SHOCKLEY'
        ]
        if not shockley_between:
            continue

        geometry_score = np.nan
        if len(shockley_between) == 1:
            backbone_edge = shockley_between[0]
        else:
            tangent_external1 = edge_tangent_from_node(external1[0], node1, records, cell)
            tangent_external2 = edge_tangent_from_node(external2[0], node2, records, cell)
            if tangent_external1 is None or tangent_external2 is None:
                continue

            scored = []
            for candidate_edge in shockley_between:
                tangent1 = edge_tangent_from_node(candidate_edge, node1, records, cell)
                tangent2 = edge_tangent_from_node(candidate_edge, node2, records, cell)
                if tangent1 is None or tangent2 is None:
                    continue

                # The external tangents leave the attachment nodes.  A smooth
                # through-line therefore compares -external with the candidate.
                score = (
                    float(np.dot(-tangent_external1, tangent1))
                    + float(np.dot(-tangent2, tangent_external2))
                )
                scored.append((score, candidate_edge))

            if len(scored) != len(shockley_between):
                continue
            scored.sort(reverse=True)
            if len(scored) > 1 and scored[0][0] - scored[1][0] <= bubble_geometry_tie_tol:
                continue
            geometry_score, backbone_edge = scored[0]

        bubble_candidates = [edge for edge in between_edges if edge != backbone_edge]
        if len(bubble_candidates) != 1:
            continue
        bubble_edge = bubble_candidates[0]

        # Removing the selected bubble must leave exactly two Shockley arms at
        # both attachment nodes.
        effective1 = [edge for edge in incident1 if edge != bubble_edge]
        effective2 = [edge for edge in incident2 if edge != bubble_edge]
        if len(effective1) != 2 or len(effective2) != 2:
            continue
        if any(records[edge]['type'] != 'SHOCKLEY' for edge in effective1 + effective2):
            continue

        info = {
            'kind': 'TWO_NODE',
            'edge': bubble_edge,
            'node1': node1,
            'node2': node2,
            'backbone_edge': backbone_edge,
            'geometry_score': float(geometry_score),
        }
        bubble_edges.add(bubble_edge)
        bubble_by_node[node1].append(info)
        bubble_by_node[node2].append(info)
        bubble_info_by_edge[bubble_edge] = info

    # Case 3: the additional A--B line closes directly on a terminal vertex
    # of the lens.  One endpoint becomes a two-Shockley intermediate node
    # after removal of the bubble; the other becomes a valid
    # PERFECT/CONNECTION anchor.
    for (pair_node1, pair_node2), between_edges in sorted(pair_edges.items()):
        if len(between_edges) != 2:
            continue
        if pair_node1 in bubble_by_node or pair_node2 in bubble_by_node:
            continue

        possible = []

        # Try both orientations because the graph itself does not label which
        # endpoint is the intermediate node and which one is the lens vertex.
        for intermediate_node, vertex_node in (
            (pair_node1, pair_node2),
            (pair_node2, pair_node1),
        ):
            intermediate_incident = unique_node_edges(node_edges, intermediate_node)
            vertex_incident = unique_node_edges(node_edges, vertex_node)

            if node_burgers_residual(
                intermediate_node, intermediate_incident, records
            ) > burgers_closure_tol:
                continue
            if node_burgers_residual(
                vertex_node, vertex_incident, records
            ) > burgers_closure_tol:
                continue

            for backbone_edge in between_edges:
                if records[backbone_edge]['type'] != 'SHOCKLEY':
                    continue

                bubble_candidate = [
                    edge for edge in between_edges if edge != backbone_edge
                ]
                if len(bubble_candidate) != 1:
                    continue
                bubble_edge = bubble_candidate[0]

                effective_intermediate = [
                    edge for edge in intermediate_incident
                    if edge != bubble_edge
                ]
                effective_vertex = [
                    edge for edge in vertex_incident
                    if edge != bubble_edge
                ]

                # A must become an ordinary intermediate node of the selected
                # Shockley side.
                if len(effective_intermediate) != 2:
                    continue
                if any(
                    records[edge]['type'] != 'SHOCKLEY'
                    for edge in effective_intermediate
                ):
                    continue
                if backbone_edge not in effective_intermediate:
                    continue

                external_edge = [
                    edge for edge in effective_intermediate
                    if edge != backbone_edge
                ]
                if len(external_edge) != 1:
                    continue

                # B must remain the actual terminal vertex after the bubble is
                # removed, rather than being traversed as an intermediate node.
                vertex_kind = dissociation_anchor_kind(records, effective_vertex)
                if vertex_kind is None:
                    continue

                tangent_external = edge_tangent_from_node(
                    external_edge[0], intermediate_node, records, cell
                )
                tangent_backbone = edge_tangent_from_node(
                    backbone_edge, intermediate_node, records, cell
                )
                if tangent_external is None or tangent_backbone is None:
                    geometry_score = np.nan
                else:
                    geometry_score = float(
                        np.dot(-tangent_external, tangent_backbone)
                    )

                possible.append({
                    'kind': 'VERTEX_NODE',
                    'edge': bubble_edge,
                    'node1': intermediate_node,
                    'node2': vertex_node,
                    'backbone_edge': backbone_edge,
                    'geometry_score': geometry_score,
                    'vertex_kind': vertex_kind,
                })

        if not possible:
            continue

        if len(possible) == 1:
            info = possible[0]
        else:
            # More than one valid option occurs mainly when both A--B edges
            # are Shockley.  Select the smoother continuation at the
            # intermediate node; reject a missing or practically tied score.
            if any(not np.isfinite(item['geometry_score']) for item in possible):
                continue
            possible.sort(key=lambda item: item['geometry_score'], reverse=True)
            if (
                possible[0]['geometry_score']
                - possible[1]['geometry_score']
                <= bubble_geometry_tie_tol
            ):
                continue
            info = possible[0]

        bubble_edge = info['edge']
        bubble_edges.add(bubble_edge)
        bubble_by_node[info['node1']].append(info)
        bubble_by_node[info['node2']].append(info)
        bubble_info_by_edge[bubble_edge] = info

    # Case 4: a single link whose removal leaves two ordinary Shockley arms at
    # each endpoint.  The link may join two points of the same side or act as
    # a rung between the two sides.  This distinction can only be made after
    # the ribbon has been traced from its PERFECT anchor.
    for edge, rec in enumerate(records):
        if edge in bubble_edges:
            continue

        node1 = rec['start_node']
        node2 = rec['end_node']
        if node1 == node2:
            continue
        # Parallel direct edges are handled by Cases 2 and 3 above, where the
        # Shockley backbone can be selected explicitly.
        if len(pair_edges[tuple(sorted((node1, node2)))]) != 1:
            continue
        if node1 in bubble_by_node or node2 in bubble_by_node:
            continue

        incident1 = unique_node_edges(node_edges, node1)
        incident2 = unique_node_edges(node_edges, node2)
        effective1 = [item for item in incident1 if item != edge]
        effective2 = [item for item in incident2 if item != edge]

        if len(effective1) != 2 or len(effective2) != 2:
            continue
        if any(
            records[item]['type'] != 'SHOCKLEY'
            for item in effective1 + effective2
        ):
            continue
        if node_burgers_residual(node1, incident1, records) > burgers_closure_tol:
            continue
        if node_burgers_residual(node2, incident2, records) > burgers_closure_tol:
            continue

        # Store how smoothly the two remaining Shockley arms continue through
        # the two attachment nodes.  This is diagnostic only: final acceptance
        # still requires a complete PERFECT-anchored ribbon and the usual
        # width checks.
        geometry_score = 0.0
        geometry_valid = True
        for node, remaining in ((node1, effective1), (node2, effective2)):
            tangent1 = edge_tangent_from_node(remaining[0], node, records, cell)
            tangent2 = edge_tangent_from_node(remaining[1], node, records, cell)
            if tangent1 is None or tangent2 is None:
                geometry_valid = False
                break
            local_score = float(np.dot(-tangent1, tangent2))
            # The two retained Shockley arms must form a recognizable
            # continuation through the attachment node.  This prevents a
            # Shockley rung from causing a neighbouring backbone edge to be
            # removed instead.
            if local_score <= 0.0:
                geometry_valid = False
                break
            geometry_score += local_score

        if not geometry_valid:
            continue

        info = {
            'kind': 'CROSS_LINK',
            'edge': edge,
            'node1': node1,
            'node2': node2,
            'backbone_edge': -1,
            'geometry_score': geometry_score,
        }
        bubble_edges.add(edge)
        bubble_by_node[node1].append(info)
        bubble_by_node[node2].append(info)
        bubble_info_by_edge[edge] = info

    # Case 5: terminal bridge.  Internal bubbles/rungs have already been
    # provisionally removed.  Removing this final link must leave exactly one
    # Shockley arm at each endpoint; other non-Shockley junction arms are
    # allowed and remain part of the original dislocation network.
    for edge, rec in enumerate(records):
        if edge in bubble_edges:
            continue

        node1 = rec['start_node']
        node2 = rec['end_node']
        if node1 == node2:
            continue
        if len(pair_edges[tuple(sorted((node1, node2)))]) != 1:
            continue
        if node1 in bubble_by_node or node2 in bubble_by_node:
            continue

        incident1 = unique_node_edges(node_edges, node1)
        incident2 = unique_node_edges(node_edges, node2)
        effective1 = [
            item for item in incident1
            if item != edge and item not in bubble_edges
        ]
        effective2 = [
            item for item in incident2
            if item != edge and item not in bubble_edges
        ]
        shockley1 = [
            item for item in effective1
            if records[item]['type'] == 'SHOCKLEY'
        ]
        shockley2 = [
            item for item in effective2
            if records[item]['type'] == 'SHOCKLEY'
        ]

        if len(shockley1) != 1 or len(shockley2) != 1:
            continue
        if any(records[item]['type'] == 'PERFECT' for item in effective1 + effective2):
            continue
        if node_burgers_residual(node1, incident1, records) > burgers_closure_tol:
            continue
        if node_burgers_residual(node2, incident2, records) > burgers_closure_tol:
            continue

        tangent1 = edge_tangent_from_node(shockley1[0], node1, records, cell)
        tangent2 = edge_tangent_from_node(shockley2[0], node2, records, cell)
        if tangent1 is None or tangent2 is None:
            continue
        terminal_parallel_score = float(np.dot(tangent1, tangent2))
        # At a terminal bridge both remaining Shockley arms point back toward
        # the already traced ribbon and are therefore broadly parallel.  For a
        # normal Shockley backbone edge they point away in opposite directions.
        if terminal_parallel_score <= 0.0:
            continue

        info = {
            'kind': 'TERMINAL_BRIDGE',
            'edge': edge,
            'node1': node1,
            'node2': node2,
            'backbone_edge': -1,
            'geometry_score': terminal_parallel_score,
        }
        bubble_edges.add(edge)
        bubble_by_node[node1].append(info)
        bubble_by_node[node2].append(info)
        bubble_info_by_edge[edge] = info

    effective_node_edges = defaultdict(list)
    for node in node_edges:
        effective_node_edges[node] = [
            edge for edge in unique_node_edges(node_edges, node)
            if edge not in bubble_edges
        ]

    return bubble_edges, bubble_by_node, bubble_info_by_edge, effective_node_edges

# ============================================================
# 4. Dissociated-Perfect candidate functions
# ============================================================

def align_point(pos, ref, cell):
    """Move pos to the periodic image closest to ref."""
    return np.asarray(ref, dtype=float) + np.asarray(cell.delta_vector(ref, pos))

def path_polyline(path_edges, path_nodes, records, node_positions, cell):

    result = []
    reference = node_positions[path_nodes[0]]

    for k, edge in enumerate(path_edges):

        rec = records[edge]
        node_from = path_nodes[k]

        # Orient the edge points along the path traversal direction.
        if rec['start_node'] == node_from:
            pts = rec['points'].copy()
        else:
            pts = rec['points'][::-1].copy()

        # Move the first point to the periodic image closest to the last
        # point of the preceding segment.
        pts[0] = align_point(pts[0], reference, cell)

        # Reconstruct the segment point by point across the PBC.
        for i in range(1, len(pts)):
            pts[i] = align_point(pts[i], pts[i - 1], cell)

        # Append the segment to the complete path.
        if len(result):
            pts[0] = result[-1]
            result.extend(pts[1:])
        else:
            result.extend(pts)

        # Use the final point as the reference for the next edge.
        reference = result[-1]

    return np.asarray(result, dtype=float)

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))

def polyline_arclength_midpoint(points):
    """Point at half arclength of an already unwrapped polyline."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError('Cannot find the midpoint of an empty polyline')
    if len(points) == 1:
        return points[0].copy()

    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = cumulative[-1]
    if total <= 1.0e-12:
        return points[0].copy()

    target = 0.5 * total
    index = int(np.searchsorted(cumulative, target, side='right') - 1)
    index = min(max(index, 0), len(points) - 2)
    step = steps[index]
    if step <= 1.0e-12:
        return points[index].copy()
    fraction = (target - cumulative[index]) / step
    return points[index] + fraction * (points[index + 1] - points[index])

def resample_polyline(points, number):
    if len(points) < 2:
        return np.repeat(points[:1], number, axis=0)
    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = cumulative[-1]
    if total <= 1.0e-12:
        return np.repeat(points[:1], number, axis=0)
    target = np.linspace(0.0, total, number)
    out = np.empty((number, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(target, cumulative, points[:, dim])
    return out

def _candidate_dissociated_perfects_pass(
    records,
    node_positions,
    node_edges,
    cell,
    enable_accessories,
):
    """
    Find chains of one or more Shockley lenses starting from a PERFECT
    terminal.

    A chain may end at another PERFECT terminal, at a common connection node
    of arbitrary degree, or at two distinct nodes joined by one terminal
    bridge.  At a common connection node, the chain continues only if exactly
    two unused Shockley segments form another valid lens.
    """
    
    if enable_accessories:
        (
            _provisional_bubble_edges,
            bubble_by_node,
            bubble_info_by_edge,
            effective_node_edges,
        ) = identify_intermediate_bubbles(records, node_edges, cell)
    else:
        # Baseline pass: use the untouched DXA graph.  This pass intentionally
        # reproduces the topology accepted by the original script and cannot
        # be altered by a provisional bubble/rung/bridge classification.
        _provisional_bubble_edges = set()
        bubble_by_node = defaultdict(list)
        bubble_info_by_edge = {}
        effective_node_edges = defaultdict(list)
        for node in node_edges:
            effective_node_edges[node] = unique_node_edges(node_edges, node)

    # Look for nodes that could be junctions for dissociated perfect
    # dislocations.  Bubble edges are absent from effective_node_edges, so
    # their attachment nodes recover their effective role: intermediate node
    # or terminal PERFECT/CONNECTION anchor.
    def dissociation_anchors(records, graph_node_edges, original_node_edges):
        """
        Identify the anchor nodes used to trace dissociated-perfect chains.

        PERFECT:
            A terminal node with exactly three incident edges: one Perfect segment and two Shockley segments.

        CONNECTION:
            A node with at least three incident edges, at least two of which
            are Shockley segments.
            It may terminate the chain or, if exactly two unused Shockley segments remain, connect the current lens to another lens.

        BRIDGE_END:
            One endpoint of a provisional terminal bridge.  After removal of
            the bridge, exactly one Shockley arm remains at the node.

        All anchors must satisfy Burgers-vector closure.

        Nodes with exactly two incident Shockley segments are not anchors: they are treated as intermediate nodes while tracing a path.
        """
        anchors = {}

        for node, edges in graph_node_edges.items():
            unique_edges = list(dict.fromkeys(edges))

            perfect = [e for e in unique_edges if records[e]['type'] == 'PERFECT']  # Perfect segments incident on the node
            shockley = [e for e in unique_edges if records[e]['type'] == 'SHOCKLEY'] # Shockley segments incident on the node

            if enable_accessories:
                kind = dissociation_anchor_kind(records, unique_edges)
            elif (
                len(unique_edges) == 3
                and len(perfect) == 1
                and len(shockley) == 2
            ):
                kind = 'PERFECT'
            elif len(unique_edges) in (3, 4) and len(shockley) >= 2:
                kind = 'CONNECTION'
            else:
                kind = None
            if kind is None:
                terminal_infos = [
                    info for info in bubble_by_node.get(node, [])
                    if info['kind'] == 'TERMINAL_BRIDGE'
                ]
                if (
                    enable_accessories
                    and len(terminal_infos) == 1
                    and len(shockley) == 1
                ):
                    kind = 'BRIDGE_END'
                else:
                    continue

            # At a vertex carrying a bubble, conservation applies to the full
            # original junction, including the additional line.  Its effective
            # topology, however, is classified after removing that line.
            closure_edges = (
                unique_node_edges(original_node_edges, node)
                if enable_accessories and node in bubble_by_node
                else unique_edges
            )
            if node_burgers_residual(node, closure_edges, records) > burgers_closure_tol:
                continue

            anchors[node] = {
                'kind': kind,
                'perfect': perfect[0] if kind == 'PERFECT' else -1,
                'shockley': shockley,
            }
        return anchors

    anchors = dissociation_anchors(records, effective_node_edges, node_edges)
    anchor_nodes = set(anchors)     # indices of the anchor nodes
    visited_candidates = set()      # candidates and edges already analysed
    candidates = []                 # candidates that survive all checks

    for seed in sorted(anchors):
        if anchors[seed]['kind'] != 'PERFECT':
            continue
        # Start from the first PERFECT node.
        component_nodes = {seed}    # nodes in the connected Shockley component; initially only the seed
        component_edges = set()     # Shockley edges found
        queue = deque([seed])       # queue initialised with the seed for the breadth-first search

        while queue:
            node = queue.popleft() # take the first node from the queue

            for edge in effective_node_edges[node]:
                # Analyse all edges incident on the node, skipping non-Shockley
                # segments and edges that have already been processed.
                if records[edge]['type'] != 'SHOCKLEY':
                    continue
                if edge in component_edges:
                    continue

                component_edges.add(edge)

                # Find the node at the opposite end of the edge.
                if records[edge]['start_node'] == node:
                    next_node = records[edge]['end_node']
                else:
                    next_node = records[edge]['start_node']

                # Add each newly encountered node to the queue.
                if next_node not in component_nodes:
                    component_nodes.add(next_node)
                    queue.append(next_node)
        # Once the queue is empty, the complete connected Shockley component
        # has been found.

        start = seed
        used_edges = set()
        used_bubble_edges = set()
        resolved_accessory_kinds = {}
        terminal_bridge_info = None
        terminal_end_nodes = None
        current = start
        lenses = [] # each lens is (path1, path2), with the edges and nodes of its two sides

        # Identify one side of a lens from PERFECT to CONNECTION/PERFECT.
        # The path is traced on the effective graph, in
        # which provisional bubble edges have been removed.
        def trace_to_next_anchor(
            start,
            first_edge,
            component_edges,
            anchor_nodes,
            records,
            graph_node_edges,
            forbidden_edges,
        ):
            """
            Walk along Shockley segments from 'start' until another anchor node
            is reached. Every intermediate node must have exactly two Shockley
            arms; otherwise, the topology is not a simple ribbon and the walk fails.
            """
            path_edges = []
            path_nodes = [start]
            path_bubbles = set()
            used = set()
            current = start
            edge = first_edge

            while True:
                if edge in used or edge in forbidden_edges or edge not in component_edges:
                    return None
                used.add(edge)
                path_edges.append(edge)
                if records[edge]['start_node'] == current:
                    current = records[edge]['end_node']
                else:
                    current = records[edge]['start_node']
                path_nodes.append(current)

                # Record any additional line attached to the traversed side.
                # It is not part of path_edges and will not enter the ribbon
                # geometry.
                for bubble_info in bubble_by_node.get(current, []):
                    # Both sides of a lens reach the same terminal vertex.
                    # A vertex-attached bubble belongs only to the side that
                    # actually contains its selected backbone edge.
                    if (
                        bubble_info['kind'] == 'VERTEX_NODE'
                        and bubble_info['backbone_edge'] not in path_edges
                    ):
                        continue
                    path_bubbles.add(bubble_info['edge'])

                if current in anchor_nodes:
                    return path_edges, path_nodes, path_bubbles

                # A non-anchor must be an ordinary intermediate node with
                # exactly two Shockley edges.
                current_edges = list(dict.fromkeys(graph_node_edges[current]))

                current_shockley = [e for e in current_edges if records[e]['type'] == 'SHOCKLEY']

                if len(current_edges) != 2 or len(current_shockley) != 2: return None

                next_edges = [
                    e for e in current_shockley
                    if e in component_edges
                    and e not in used
                    and e not in forbidden_edges
                ]

                if len(next_edges) != 1: return None

                next_edge = next_edges[0]

                # In an ordinary intermediate node the two Shockley arms must
                # cancel.  At a bubble attachment node, closure was instead
                # checked on the complete original three-edge topology.
                if current not in bubble_by_node:
                    b_previous = records[edge]['b']
                    if records[edge]['end_node'] == current:
                        b_previous = -b_previous
                    b_next = records[next_edge]['b']
                    if records[next_edge]['end_node'] == current:
                        b_next = -b_next
                    if np.linalg.norm(b_previous + b_next) > burgers_closure_tol:
                        return None

                edge = next_edge

        while True:

            # Shockley segments incident on the current node and not yet used.
            current_edges = list(dict.fromkeys(effective_node_edges[current]))

            free_edges = [
                e for e in current_edges
                if e in component_edges
                and records[e]['type'] == 'SHOCKLEY'
                and e not in used_edges
            ]

            # If exactly two Shockley segments do not remain, terminate the
            # candidate at the current node.
            if len(free_edges) != 2: break

            forbidden_edges = used_edges | used_bubble_edges
            path1 = trace_to_next_anchor(
                current, free_edges[0], component_edges, anchor_nodes,
                records, effective_node_edges, forbidden_edges,
            )
            path2 = trace_to_next_anchor(
                current, free_edges[1], component_edges, anchor_nodes,
                records, effective_node_edges, forbidden_edges,
            )

            # If the two Shockley segments do not form a new lens, retain the
            # preceding lenses and terminate the chain here.
            if path1 is None or path2 is None: break
            # The two sides must not share edges.
            if set(path1[0]).intersection(path2[0]): break

            end1 = path1[1][-1]
            end2 = path2[1][-1]
            path1_nodes = set(path1[1])
            path2_nodes = set(path2[1])

            # Resolve provisional cross-links only after both Shockley sides
            # are known.  A link may have both endpoints on one side (a
            # same-side bubble) or one endpoint on each side (a rung).
            path1_bubbles = set(path1[2])
            path2_bubbles = set(path2[2])
            lens_bubbles = path1_bubbles | path2_bubbles
            shared_bubbles = path1_bubbles & path2_bubbles
            lens_resolved_kinds = {}
            lens_terminal_bridges = []
            valid_accessories = True

            for bubble_edge in lens_bubbles:
                info = bubble_info_by_edge[bubble_edge]
                kind = info['kind']
                node1 = info['node1']
                node2 = info['node2']

                if kind == 'CROSS_LINK':
                    node1_side1 = node1 in path1_nodes
                    node1_side2 = node1 in path2_nodes
                    node2_side1 = node2 in path1_nodes
                    node2_side2 = node2 in path2_nodes

                    if (
                        node1_side1 and node2_side1
                        and not node1_side2 and not node2_side2
                    ) or (
                        node1_side2 and node2_side2
                        and not node1_side1 and not node2_side1
                    ):
                        kind = 'TWO_NODE_PATH'
                    elif (
                        node1_side1 and not node1_side2
                        and node2_side2 and not node2_side1
                    ) or (
                        node2_side1 and not node2_side2
                        and node1_side2 and not node1_side1
                    ):
                        kind = 'CROSS_RUNG'
                    else:
                        valid_accessories = False
                        break

                elif kind == 'TERMINAL_BRIDGE':
                    if {node1, node2} != {end1, end2} or end1 == end2:
                        valid_accessories = False
                        break
                    lens_terminal_bridges.append(info)

                lens_resolved_kinds[bubble_edge] = kind

            if not valid_accessories:
                break

            # Shared additional lines are allowed only when they genuinely
            # connect the two opposite sides: an internal rung or the unique
            # terminal bridge.
            if any(
                lens_resolved_kinds.get(edge)
                not in ('CROSS_RUNG', 'TERMINAL_BRIDGE')
                for edge in shared_bubbles
            ):
                break

            if end1 == end2:
                # Ordinary closure in one common node, irrespective of the
                # total degree of that junction.
                if lens_terminal_bridges:
                    break
                closes_with_bridge = False
            else:
                # Distinct Shockley endpoints are accepted only when one and
                # only one additional segment connects those exact nodes.
                if len(lens_terminal_bridges) != 1:
                    break
                bridge_edge = lens_terminal_bridges[0]['edge']
                if bridge_edge not in shared_bubbles:
                    break
                closes_with_bridge = True

            # Additional lines must be disjoint from the Shockley backbone and
            # from all previously accepted parts of the same chain.
            if lens_bubbles.intersection(path1[0]): break
            if lens_bubbles.intersection(path2[0]): break
            if lens_bubbles.intersection(used_edges): break
            if lens_bubbles.intersection(used_bubble_edges): break

            # The new lens is valid.
            used_edges.update(path1[0])
            used_edges.update(path2[0])
            used_bubble_edges.update(lens_bubbles)
            resolved_accessory_kinds.update(lens_resolved_kinds)

            lenses.append((path1, path2))

            if closes_with_bridge:
                terminal_bridge_info = lens_terminal_bridges[0]
                terminal_end_nodes = (end1, end2)
                break

            # Use the common node as the starting point of a possible next lens.
            current = end1

        if not lenses:
            continue
        candidate_key = tuple(sorted(used_edges | used_bubble_edges))
        if candidate_key in visited_candidates:
            continue
        
        side1_parts = []
        side2_parts = []
        width_parts = []

        side1_total_length = 0.0
        side2_total_length = 0.0
        midline_total_length = 0.0

        bad_lens = False

        for path1, path2 in lenses:
            side1_line = path_polyline(path1[0], path1[1], records, node_positions, cell) # retrieve all DXA points of path1 in traversal order, correcting orientation and PBC
            side2_line = path_polyline(path2[0], path2[1], records, node_positions, cell)
            side1_length = polyline_length(side1_line)
            side2_length = polyline_length(side2_line)

            # The two sides may contain different numbers of points and must
            # therefore be resampled.
            longest_side = max(side1_length, side2_length)

            number_of_points = max(3, int(ceil(longest_side / 2)) + 1)    # approximate resampling interval: 2 Angstrom
            side1_points = resample_polyline(side1_line, number_of_points)  # resample by interpolation while retaining the endpoints
            side2_points = resample_polyline(side2_line, number_of_points)

            # Move each point of the second side to the periodic image closest
            # to the corresponding point of the first side.
            for i in range(number_of_points): side2_points[i] = align_point(side2_points[i], side1_points[i], cell)

            # Distance between the two sides at each position.
            local_widths = np.linalg.norm(side2_points - side1_points, axis=1)
            mean_width = np.mean(local_widths)
            maximum_width = np.max(local_widths)

            if (mean_width < ribbon_min_width or mean_width > ribbon_max_mean_width or maximum_width > ribbon_max_local_width): # apply the ribbon-width limits
                bad_lens = True
                break
           
            # Average the two sides point by point to obtain the midline.
            lens_midline = 0.5 * (side1_points + side2_points)

            side1_parts.append(side1_points)
            side2_parts.append(side2_points)
            width_parts.append(local_widths)

            # Total length of all concatenated lenses.
            side1_total_length += side1_length
            side2_total_length += side2_length

        if bad_lens: continue

        visited_candidates.add(candidate_key)

        # Move each lens to the same periodic image as the preceding one.
        for i in range(1, len(side1_parts)):

            aligned_start = align_point(
                side1_parts[i][0],       # start of the current lens
                side1_parts[i - 1][-1],  # end of the preceding lens
                cell
            )

            shift = aligned_start - side1_parts[i][0]

            # Shift both sides of the lens together.
            side1_parts[i] = side1_parts[i] + shift
            side2_parts[i] = side2_parts[i] + shift

        # Store the midline and maximum width of each lens separately.
        midline_parts = []
        width_max_parts = []

        for side1_part, side2_part, local_widths in zip(side1_parts, side2_parts, width_parts):
            midline_parts.append(0.5 * (side1_part + side2_part))
            width_max_parts.append(float(np.max(local_widths)))

        # When the two Shockley sides end at distinct nodes connected by a
        # terminal bridge, the reconstructed perfect ends at the half-arclength
        # point of that bridge.  The bridge is unwrapped through the PBC and
        # moved into the same periodic image as the final ribbon cross-section.
        if terminal_bridge_info is not None:
            bridge_edge = terminal_bridge_info['edge']
            bridge_node1 = terminal_bridge_info['node1']
            bridge_node2 = terminal_bridge_info['node2']
            bridge_line = path_polyline(
                [bridge_edge],
                [bridge_node1, bridge_node2],
                records,
                node_positions,
                cell,
            )
            terminal_anchor = polyline_arclength_midpoint(bridge_line)
            final_cross_section_midpoint = 0.5 * (
                side1_parts[-1][-1] + side2_parts[-1][-1]
            )
            terminal_anchor = align_point(
                terminal_anchor, final_cross_section_midpoint, cell
            )
            midline_parts[-1][-1] = terminal_anchor

        # Concatenate the lenses belonging to the same chain.
        side1 = np.concatenate(side1_parts, axis=0)
        side2 = np.concatenate(side2_parts, axis=0)
        width = np.concatenate(width_parts)

        midline = np.concatenate(midline_parts, axis=0)
        midline_total_length = sum(
            polyline_length(part) for part in midline_parts
        )

        # Burgers vector of the Perfect segment incident on the initial node.
        perfect_record = records[anchors[start]['perfect']]

        # Orient the Perfect Burgers vector away from the initial node.
        if perfect_record['start_node'] == start:
            perfect_b_out = perfect_record['b']
        else:
            perfect_b_out = -perfect_record['b']

        # The two Shockley segments, oriented from start to end, sum to the
        # Burgers vector of the equivalent Perfect dislocation.
        equivalent_b = -perfect_b_out

        if terminal_bridge_info is None:
            candidate_end_node = current
            candidate_end_kind = anchors[current]['kind']
            candidate_perfect_end = anchors[current]['perfect']
        else:
            candidate_end_node = -1
            candidate_end_kind = 'TERMINAL_BRIDGE'
            candidate_perfect_end = -1

        candidates.append({
            'start_node': start,
            'end_node': candidate_end_node,
            'terminal_end_nodes': terminal_end_nodes,
            'start_kind': anchors[start]['kind'],
            'end_kind': candidate_end_kind,
            'n_lenses': len(lenses),
            'perfect_start': anchors[start]['perfect'],
            'perfect_end': candidate_perfect_end,
            'equivalent_b': equivalent_b,
            'path1_edges': [edge
                for path1, path2 in lenses
                for edge in path1[0]
            ],
            'path2_edges': [
                edge
                for path1, path2 in lenses
                for edge in path2[0]
            ],
            'bubble_edges': sorted(used_bubble_edges),
            'bubble_info': [
                dict(
                    bubble_info_by_edge[edge],
                    kind=resolved_accessory_kinds.get(
                        edge, bubble_info_by_edge[edge]['kind']
                    ),
                )
                for edge in sorted(used_bubble_edges)
            ],
            'path1': side1,
            'path2': side2,
            'midline': midline,
            'midline_parts': midline_parts,
            'width_max_parts': width_max_parts,
            'length1': side1_total_length,
            'length2': side2_total_length,
            'midline_length': midline_total_length,
            'width_mean': float(np.mean(width)),
            'width_max': float(np.max(width)),
        })
    return candidates


def candidate_dissociated_perfects(records, node_positions, node_edges, cell):
    """Return the invariant baseline plus locally compatible extensions.

    The first pass runs on the untouched DXA graph and therefore preserves
    every candidate recognized by the original algorithm.  The second pass
    may provisionally hide bubble/rung/bridge edges in order to recover more
    complex ribbons.  Its result is allowed to replace a baseline candidate
    only when it contains that candidate's complete Shockley backbone.

    This separation is essential: an accessory edge is only a hypothesis
    during topology discovery and must never delete a disjoint baseline lens.
    Actual exclusion from dislocation statistics still happens later, after
    the final candidate has been accepted.
    """
    baseline = _candidate_dissociated_perfects_pass(
        records,
        node_positions,
        node_edges,
        cell,
        enable_accessories=False,
    )
    extended = _candidate_dissociated_perfects_pass(
        records,
        node_positions,
        node_edges,
        cell,
        enable_accessories=True,
    )

    selected = list(baseline)

    def backbone(candidate):
        return set(candidate['path1_edges']).union(candidate['path2_edges'])

    for candidate in extended:
        candidate_backbone = backbone(candidate)
        if not candidate_backbone:
            continue

        exact_matches = [
            index for index, original in enumerate(selected)
            if backbone(original) == candidate_backbone
        ]
        if exact_matches:
            # Keep the byte-for-byte baseline geometry for an ordinary lens.
            # Replace it only when the extended pass has associated a real
            # accessory or a terminal bridge with that same backbone.
            if (
                candidate['bubble_edges']
                or candidate['end_kind'] == 'TERMINAL_BRIDGE'
            ):
                selected[exact_matches[0]] = candidate
            continue

        overlapping = [
            index for index, original in enumerate(selected)
            if candidate_backbone.intersection(backbone(original))
        ]

        if not overlapping:
            # This is the important invariant: disjoint original and extended
            # lenses coexist.  A provisional accessory elsewhere in the graph
            # cannot suppress an original candidate.
            selected.append(candidate)
            continue

        overlapping_backbones = [backbone(selected[index]) for index in overlapping]
        strictly_extends_all = all(
            original_backbone.issubset(candidate_backbone)
            for original_backbone in overlapping_backbones
        ) and any(
            original_backbone != candidate_backbone
            for original_backbone in overlapping_backbones
        )

        if strictly_extends_all:
            # A longer multi-lens chain may legitimately absorb one or more
            # baseline pieces, but a merely partial/crossing overlap is never
            # allowed to erase an original result.
            insertion_index = min(overlapping)
            for index in sorted(overlapping, reverse=True):
                del selected[index]
            selected.insert(insertion_index, candidate)

    return selected

# ============================================================
# 5. HCP-grain analysis functions
# ============================================================

def unwrap_grain(indices, positions, cell, neighbor_finder):
    """Reconstruct a finite grain across the PBC by following neighbouring atoms."""

    indices = np.asarray(indices, dtype=int)
    global_to_local = {int(atom): local_index for local_index, atom in enumerate(indices)}
    unwrapped_positions = np.full((len(indices), 3), np.nan, dtype=float)
    visited = np.zeros(len(indices), dtype=bool)
    number_of_components = 0

    for root_local in range(len(indices)):

        if visited[root_local]: continue

        number_of_components += 1
        root_global = int(indices[root_local])

        if number_of_components == 1:
            unwrapped_positions[root_local] = positions[root_global]
        else:
            first_global = int(indices[0])
            unwrapped_positions[root_local] = unwrapped_positions[0] + np.asarray(cell.delta_vector(positions[first_global], positions[root_global]))

        visited[root_local] = True
        queue = deque([root_local])

        while queue:

            current_local = queue.popleft()
            current_global = int(indices[current_local])

            for neighbor in neighbor_finder.find(current_global):

                other_global = int(neighbor.index)

                if other_global not in global_to_local: continue

                other_local = global_to_local[other_global]

                if visited[other_local]: continue

                unwrapped_positions[other_local] = unwrapped_positions[current_local] + np.asarray(neighbor.delta)
                visited[other_local] = True
                queue.append(other_local)

    return unwrapped_positions, number_of_components

def convex_hull_2d(points):
    """Return the smallest convex polygon enclosing all points."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) < 3:
        arr = np.asarray(pts, dtype=float)
        if len(arr) == 2: perimeter = 2.0 * np.linalg.norm(arr[1] - arr[0])
        else: perimeter = 0.0
        return arr, 0.0, perimeter

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) # cross product of two vectors in the plane

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()  # discard collinear or clockwise points while moving from left to right
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    shifted = np.roll(hull, -1, axis=0)
    area = 0.5 * abs(np.sum(hull[:, 0] * shifted[:, 1] - hull[:, 1] * shifted[:, 0]))   # shoelace formula
    perimeter = float(np.sum(np.linalg.norm(shifted - hull, axis=1)))
    
    return hull, float(area), perimeter

def grid_outline_2d(layer_cells, grid_step, dilation):
    """Build the external grid outline of all HCP layers."""

    occupied_cells = set()

    # Combine the cells occupied by all layers.
    for cells in layer_cells:
        occupied_cells.update(cells)

    if not occupied_cells:
        return np.empty((0, 2, 2), dtype=float), 0.0, 0.0

    # Expand the region to connect neighbouring atomic sites.
    expanded_cells = set()

    for x, y in occupied_cells:
        for dx in range(-dilation, dilation + 1):
            for dy in range(-dilation, dilation + 1):
                expanded_cells.add((x + dx, y + dy))

    occupied_cells = expanded_cells

    # Bounding rectangle of the grid, including a one-cell empty margin.
    x_values = [cell[0] for cell in occupied_cells]
    y_values = [cell[1] for cell in occupied_cells]

    x_min = min(x_values) - 1
    x_max = max(x_values) + 1
    y_min = min(y_values) - 1
    y_max = max(y_values) + 1

    # Identify empty cells connected to the exterior.
    outside_cells = {(x_min, y_min)}
    queue = deque([(x_min, y_min)])

    while queue:

        x, y = queue.popleft()

        for next_cell in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        ):

            next_x, next_y = next_cell

            if next_x < x_min or next_x > x_max:
                continue

            if next_y < y_min or next_y > y_max:
                continue

            if next_cell in occupied_cells:
                continue

            if next_cell in outside_cells:
                continue

            outside_cells.add(next_cell)
            queue.append(next_cell)

    # Fill small internal holes that do not represent the physical grain
    # boundary.
    filled_cells = set(occupied_cells)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if (x, y) not in outside_cells:
                filled_cells.add((x, y))

    outline_segments = []

    # A side belongs to the outline only if it separates the grain from a cell
    # that is connected to the exterior.
    for x, y in filled_cells:

        lower_left = grid_step * np.asarray([x - 0.5, y - 0.5])
        lower_right = grid_step * np.asarray([x + 0.5, y - 0.5])
        upper_right = grid_step * np.asarray([x + 0.5, y + 0.5])
        upper_left = grid_step * np.asarray([x - 0.5, y + 0.5])

        if (x, y - 1) in outside_cells:
            outline_segments.append([lower_left, lower_right])

        if (x + 1, y) in outside_cells:
            outline_segments.append([lower_right, upper_right])

        if (x, y + 1) in outside_cells:
            outline_segments.append([upper_right, upper_left])

        if (x - 1, y) in outside_cells:
            outline_segments.append([upper_left, lower_left])

    outline_segments = np.asarray(outline_segments, dtype=float)
    area = len(filled_cells) * grid_step**2
    perimeter = len(outline_segments) * grid_step

    return outline_segments, float(area), float(perimeter)

def point_outline_distance(point, outline_segments):
    """Distance from a 2D point to the closest outline segment."""

    best_distance = np.inf

    for segment in outline_segments:

        point1 = segment[0]
        point2 = segment[1]
        vector = point2 - point1
        squared_length = np.dot(vector, vector)

        if squared_length <= 1.0e-16:
            distance = np.linalg.norm(point - point1)
        else:
            fraction = np.dot(point - point1, vector) / squared_length
            fraction = min(1.0, max(0.0, fraction))
            closest_point = point1 + fraction * vector
            distance = np.linalg.norm(point - closest_point)

        if distance < best_distance:
            best_distance = distance

    return float(best_distance)

def point_hull_distance(point, hull):
    """Distance from a 2D point to the closest edge of the hull polygon."""
    if len(hull) == 0:
        return np.inf
    if len(hull) == 1:
        return float(np.linalg.norm(point - hull[0]))

    best = np.inf
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        ab = b - a
        denom = np.dot(ab, ab)
        if denom <= 1.0e-16:
            distance = float(np.linalg.norm(point - a))
        else:
            t = np.dot(point - a, ab) / denom
            t = min(1.0, max(0.0, t))
            distance = float(np.linalg.norm(point - (a + t * ab)))
        if distance < best:
            best = distance
    return best

def atom_composition(indices, particle_types, type_names):
    """Counts and atomic fractions of Fe, Cr, Ni """
    indices = np.unique(np.asarray(indices, dtype=int))
    counts = dict((name, 0) for name in ('FE', 'CR', 'NI'))
    for atom in indices:
        species = type_names.get(int(particle_types[atom]))
        counts[species] += 1

    total = len(indices)
    fractions = dict((name, counts[name] / float(total) if total else 0.0) for name in counts)
    return {'total': total, 'counts': counts, 'fractions': fractions}

def dislocation_association(plane, records, node_edges, cell, periodic_shifts):
    """Associate portions of DXA polylines with a projected plane boundary."""
    center = plane['center']
    normal = plane['normal']
    axis1 = plane['axes'][:, 0]
    axis2 = plane['axes'][:, 1]
    outline_segments = plane['outline_segments']

    by_edge = {}
    by_type = defaultdict(float)
    edge_fractions = {}

    for edge_index, rec in enumerate(records):
        if rec['role'] in ('DISSOCIATED_PERFECT', 'DISSOCIATION_BUBBLE'):
            continue
        pts = rec['points']
        associated = 0.0
        for k in range(len(pts) - 1):
            # Reconstruct the DXA section continuously across the PBC.
            point1 = np.asarray(pts[k], dtype=float)
            point2 = align_point(pts[k + 1], point1, cell)
            length = np.linalg.norm(point2 - point1)

            if length <= 1.0e-12: continue

            midpoint = 0.5 * (point1 + point2)

            # First move the midpoint to the minimum image of the centre.
            base_midpoint = align_point(midpoint, center, cell)
            segment_associated = False

            # Find the periodic copy closest to the unwrapped grain boundary.
            for periodic_shift in periodic_shifts:
                image_midpoint = base_midpoint + periodic_shift
                delta = image_midpoint - center

                if abs(np.dot(delta, normal)) > boundary_plane_tol:
                    continue

                point2d = np.array([np.dot(delta, axis1), np.dot(delta, axis2)])

                if point_outline_distance(point2d, outline_segments) <= boundary_edge_tol:
                    segment_associated = True
                    break

            if segment_associated:
                associated += length

        full_edge_length = float(rec['length'])
        if full_edge_length > 1.0e-12:  associated_fraction = associated / full_edge_length
        else: associated_fraction = 0.0
        # Retain the DXA line only if at least the required fraction of its
        # length lies near the grain boundary.
        if associated_fraction >= boundary_edge_min_fraction:
            by_edge[edge_index] = associated
            edge_fractions[edge_index] = associated_fraction
            by_type[rec['type']] += associated

    edge_set = set(by_edge)
    connected = False
    closed = False
    if edge_set:
        start = next(iter(edge_set))
        seen_edges = set([start])
        queue = deque([start])
        while queue:
            edge = queue.popleft()
            rec = records[edge]
            for node in (rec['start_node'], rec['end_node']):
                for other in node_edges[node]:
                    if other in edge_set and other not in seen_edges:
                        seen_edges.add(other)
                        queue.append(other)
        connected = len(seen_edges) == len(edge_set)

        network_nodes = set()
        for edge in edge_set:
            network_nodes.add(records[edge]['start_node'])
            network_nodes.add(records[edge]['end_node'])
        if connected:
            number_of_cycles = len(edge_set) - len(network_nodes) + 1
        else:
            number_of_cycles = 0

        closed = number_of_cycles > 0

    total = float(sum(by_edge.values()))
    coverage = total / max(plane['perimeter_hull'], 1.0e-12)
    accepted = bool(edge_set) and coverage >= boundary_min_coverage
    if require_connected_boundary: accepted = accepted and connected
    if require_closed_boundary: accepted = accepted and closed

    dxa_ids = [int(records[edge]['id']) for edge in sorted(edge_set)]

    return {
        'edge_lengths': by_edge,
        'edge_fractions': edge_fractions,
        'edge_set': edge_set,
        'dxa_ids': dxa_ids,
        'type_lengths': by_type,
        'total': total,
        'coverage': coverage,
        'connected': connected,
        'closed': closed,
        'accepted': accepted,
    }

def analyze_hcp_grain_geometry(current_grain_id, indices, pos, center, singular, pca_axes, pca_normal, d111, local_dnn, particle_types, type_names, orientation_by_grain):
    """Identify the HCP layers of one grain and return its planar geometry."""

    relative = pos - center
    projection = np.dot(relative, pca_normal)

    # Build the histogram of projections along the PCA normal.
    histogram_bin_width = 0.1 * d111
    number_of_bins = max(10, int(ceil(np.ptp(projection) / histogram_bin_width)))
    histogram, bin_edges = np.histogram(projection, bins=number_of_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Find sufficiently high local maxima.
    peak_candidates = []
    minimum_peak_height = 0.05 * np.max(histogram)

    for i in range(len(histogram)):
        left_value = histogram[i - 1] if i > 0 else -1
        right_value = histogram[i + 1] if i < len(histogram) - 1 else -1
        is_local_maximum = histogram[i] >= left_value and histogram[i] > right_value
        is_large_enough = histogram[i] >= minimum_peak_height

        if is_local_maximum and is_large_enough:
            peak_candidates.append(i)

    # If two maxima are too close, retain only the higher one.
    selected_peaks = []
    minimum_peak_separation = 0.5 * d111

    for candidate_peak in sorted(peak_candidates, key=lambda peak: histogram[peak], reverse=True):
        sufficiently_far = all(abs(bin_centers[candidate_peak] - bin_centers[accepted_peak]) >= minimum_peak_separation for accepted_peak in selected_peaks)

        if sufficiently_far:
            selected_peaks.append(candidate_peak)

    selected_peaks.sort()
    peak_centers = bin_centers[selected_peaks]

    # No recognisable HCP layer.
    if len(peak_centers) == 0:
        irregular_result = {
            'n_layers': 0,
            'layer_pattern': '-',
            'layer_sizes_text': '-',
            'layer_gaps_text': '-',
            'minimum_layer_overlap': 0.0,
            'reasons': ['NO_HCP_LAYER_PEAKS']
        }
        return None, irregular_result

    # Assign each atom to the nearest peak.
    distances_to_peaks = np.abs(projection[:, None] - peak_centers[None, :])
    atom_peak_labels = np.argmin(distances_to_peaks, axis=1)

    # Discard peaks containing too few atoms.
    minimum_layer_size = max(layer_min_atoms, int(ceil(layer_min_atom_fraction * len(indices))))
    groups = []

    for peak_index in range(len(peak_centers)):
        group = np.flatnonzero(atom_peak_labels == peak_index)

        if len(group) >= minimum_layer_size:
            groups.append(group)

    if not groups:
        irregular_result = {
            'n_layers': 0,
            'layer_pattern': '-',
            'layer_sizes_text': '-',
            'layer_gaps_text': '-',
            'minimum_layer_overlap': 0.0,
            'reasons': ['NO_VALID_HCP_LAYERS']
        }
        return None, irregular_result

    # Recalculate each layer centre using the assigned atoms.
    layer_centers = np.asarray([np.mean(projection[group]) for group in groups])
    layer_order = np.argsort(layer_centers)
    groups = [groups[i] for i in layer_order]
    layer_centers = layer_centers[layer_order]

    n_layers = len(groups)
    layer_atoms = np.asarray([len(group) for group in groups], dtype=int)
    layer_gaps = np.diff(layer_centers) / d111 if n_layers > 1 else np.array([], dtype=float)

    # Classify the layer sequence.
    layer_pattern = 'HCP'
    regular_spacing = True

    for gap in layer_gaps:
        if abs(gap - 1.0) <= 0.25:
            layer_pattern += '-HCP'
        elif abs(gap - 2.0) <= 0.25:
            layer_pattern += '-FCC-HCP'
        else:
            layer_pattern += '-?-HCP'
            regular_spacing = False

    # Project all atoms onto the first two PCA axes.
    points2d = np.column_stack((np.dot(relative, pca_axes[:, 0]), np.dot(relative, pca_axes[:, 1])))

    # Divide the plane into cells and record those occupied by each layer.
    grid_step = 0.6 * local_dnn
    layer_cells = []

    for group in groups:
        cells = np.rint(points2d[group] / grid_step).astype(int)
        occupied_cells = set((int(cell_index[0]), int(cell_index[1])) for cell_index in cells)
        layer_cells.append(occupied_cells)

    # Find the lowest overlap among all pairs of layers.
    minimum_layer_overlap = 1.0

    for a in range(n_layers):
        for b in range(a + 1, n_layers):
            cells_a = layer_cells[a]
            cells_b = layer_cells[b]
            expanded_a = set()
            expanded_b = set()

            for x, y in cells_a:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        expanded_a.add((x + dx, y + dy))

            for x, y in cells_b:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        expanded_b.add((x + dx, y + dy))

            fraction_a = sum(cell_index in expanded_b for cell_index in cells_a) / float(len(cells_a))
            fraction_b = sum(cell_index in expanded_a for cell_index in cells_b) / float(len(cells_b))
            pair_overlap = min(fraction_a, fraction_b)
            minimum_layer_overlap = min(minimum_layer_overlap, pair_overlap)

    regular_layers = regular_spacing and minimum_layer_overlap >= layer_overlap_min

    # Return the information required to record an irregular grain.
    if not regular_layers:
        reasons = []

        if not regular_spacing:
            reasons.append('IRREGULAR_LAYER_SPACING')

        if minimum_layer_overlap < layer_overlap_min:
            reasons.append('PARTIAL_LAYER_OVERLAP')

        irregular_result = {
            'n_layers': n_layers,
            'layer_pattern': layer_pattern,
            'layer_sizes_text': ','.join(str(value) for value in layer_atoms),
            'layer_gaps_text': ','.join('{:.4g}'.format(value) for value in layer_gaps) if len(layer_gaps) else '-',
            'minimum_layer_overlap': minimum_layer_overlap,
            'reasons': reasons
        }
        return None, irregular_result

    # Geometry of the regular grain.
    hull, area_hull, perimeter_hull = convex_hull_2d(points2d)

    # Grid-based concave outline, which follows an irregular boundary more closely.
    outline_segments, area_outline, perimeter_outline = grid_outline_2d(layer_cells, grid_step, outline_grid_dilation)


    thickness = float(np.percentile(projection, 95) - np.percentile(projection, 5))
    planarity = singular[2] / max(singular[1], 1.0e-12)
    linearity = singular[1] / max(singular[0], 1.0e-12)

    plane = {
        'grain_id': current_grain_id,
        'indices': indices,
        'positions': pos,
        'center': center,
        'normal': pca_normal,
        'axes': pca_axes,
        'singular': singular,
        'planarity': planarity,
        'linearity': linearity,
        'thickness': thickness,
        'n_layers': n_layers,
        'layer_centers': layer_centers,
        'layer_atoms': layer_atoms,
        'layer_gaps_d111': layer_gaps,
        'layer_pattern': layer_pattern,
        'minimum_layer_overlap': minimum_layer_overlap,
        'd111': d111,
        'hull': hull,
        'area_hull': area_hull,
        'perimeter_hull': perimeter_hull,
        'outline_segments': outline_segments,
        'area_outline': area_outline,
        'perimeter_outline': perimeter_outline,
        'composition': atom_composition(indices, particle_types, type_names),
        'orientation': orientation_by_grain[current_grain_id],
        'review_with': [],
        'max_failed_edge_fraction': 0.0,
        'structure_id': 0,
        'status': 'UNCLASSIFIED'
    }

    return plane, None

# ============================================================
# 6. Initialise timing
# ============================================================
from datetime import datetime
print("DATE AND TIME: ", datetime.now())

s_time = datetime.now()
start_time = datetime.now()

# ============================================================
# 7. Build and sort the input file list
# ============================================================
newlist = list()
readlist = list()

for a in sys.argv[1:]: newlist.append(a.rsplit('.cfg.gz', 1)[0])
newlist.sort(key=lambda x: int(re.search(r'\d+', x).group()))

for a in newlist: readlist.append(a + '.cfg.gz')

if len(readlist) == 0:
    print('Usage: stacks.py file1.cfg.gz [...]')
    sys.exit(1)
   
# ============================================================
# 8. Initialise output files
# ============================================================
pd_workfile = open('PD_workfile2.tmp', 'w')
pd_workfile.write('#ITER_NB\tSEGMENT_ID\tTYPE\tLENGTH\tLINEAR_DENSITY\tBURGERS\tINFINITE?\tLOOP?\n')

pd_nano = open('PD_nano.tmp', 'w')
pd_nano.write('#ITER_NB\tPERFECT_TOT\tFRANK_TOT\tSHOCKLEY_TOT\tOTHER_TOT\tSTAIR_TOT\tHIRTH_TOT\tTOT\tTOT_NO_OTHER\tTOT_NO_NANO\tTOT_NO_OTHER_NO_NANO\n')

pd_density = open('PD_dens-nano.tmp', 'w')
pd_density.write('#ITER_NB\tPERFECT_DEN\tFRANK_DEN\tSHOCKLEY_DEN\tOTHER_DEN\tSTAIR_DEN\tHIRTH_DEN\tTOT_DEN\tDEN_NO_OTHER\tDEN_NO_NANO\tDEN_NO_OTHER_NO_NANO\n')

pd_loops = open('PD_n-nano.tmp', 'w')
pd_loops.write('#ITER_NB\tLENGTH_1\tLENGTH_2\tLENGTH_3\tLENGTH_4\tLENGTH_5\tLENGTH_6\tLENGTH_7\tLENGTH_8\tLENGTH_9\tLENGTH_10\tLENGTH_11\tTOT_LOOP_NB\tDENSITY_TOT\tDENSITY_NON_NANO_TOT\n')

pd_file = open('PD.tmp', 'w')
pd_file.write('#ITER_NB\tDISSOCIATION_ID\tSTART_NODE\tEND_NODE\tEND_KIND\tN_LENSES\tPATH1_DISLO_IDS\tPATH2_DISLO_IDS\tPATH1_LENGTH\tPATH2_LENGTH\tEQPERFECT_LENGTH\tEQPERFECT_BURGERS\tWIDTH_MEAN\tWIDTH_MAX\tPERFECT_START_ID\tPERFECT_END_ID\tHCP_GRAIN_IDS\tFOUND_HCP_ATOMS\tEQPERFECT_COORDINATES\n')

pd_bubbles = open('PD_bubbles.tmp', 'w')
pd_bubbles.write('#ITER_NB\tDISSOCIATION_ID\tBUBBLE_ID\tBUBBLE_KIND\tNODE1\tNODE2\tSEGMENT_ID\tTYPE\tLENGTH\tBURGERS\tBACKBONE_SEGMENT_ID\tGEOMETRY_SCORE\n')

grains_file = open('SF_hcp_grain_details.tmp', 'w')
grains_file.write('#ITER_NB\tSTRUCTURE_ID\tCLASS\tSUBTYPE\tGRAIN_ID\tNPART\tN_HCP_LAYERS\tLAYER_PATTERN\tCENTER_X\tCENTER_Y\tCENTER_Z\tNORMAL_X\tNORMAL_Y\tNORMAL_Z\tTHICKNESS\tHULL_AREA\tHULL_PERIMETER\tBOUNDARY_DXA_IDS\tBOUNDARY_LENGTH\tBOUNDARY_COVERAGE\tBOUNDARY_CONNECTED\tBOUNDARY_CLOSED\tFE\tCR\tNI\n')

geometry_file = open('SF_hcp_planar_structures.tmp', 'w')
geometry_file.write('#ITER_NB\tSTRUCTURE_ID\tSTRUCTURE_TYPE\tN_GRAINS\tGRAIN_IDS\tNPART\tN_HCP_LAYERS\tTHICKNESS\tHULL_AREA\tHULL_PERIMETER\tHULL_MAX_DIST\tHULL_PCA_S0_DIST\tHULL_PCA_S1_DIST\tDXA_ASSOCIATED_LENGTH\tBOUNDARY_COVERAGE\tPERFECT_LENGTH\tFRANK_LENGTH\tSHOCKLEY_LENGTH\tOTHER_LENGTH\tSTAIRROD_LENGTH\tHIRTH_LENGTH\tFE\tCR\tNI\n')

discarded_file = open('SF_discarded_grains.tmp', 'w')
discarded_file.write('#ITER_NB\tGRAIN_ID\tHCP_ATOMS\tN_HCP_LAYERS\tLAYER_PATTERN\tLAYER_SIZES\tLAYER_GAPS_D111\tMINIMUM_LAYER_OVERLAP\tREASON\n')

# ============================================================
# 9. Analyse dislocations and planar defects in each configuration
# ============================================================
for file in readlist:
    iteration = re.search(r'\d+', file).group()
    print(file)

    # ============================================================
    # 9.1 Read the configuration and run DXA
    # ============================================================
    e_time = datetime.now()
    diff = e_time - s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Lecture and DXA')
    s_time = datetime.now()

    # Import the file and cell parameters.
    node = import_file(file)
    # Extract the native DXA dislocation network and segment properties
    dislo = DislocationAnalysisModifier()
    dislo.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
    dislo.trial_circuit_length = dxa_trial_circuit_length
    dislo.circuit_stretchability = dxa_circuit_stretchability
    dislo.defect_mesh_smoothing_level = dxa_defect_mesh_smoothing
    dislo.line_coarsening_enabled = True
    dislo.line_point_separation = dxa_line_point_separation
    dislo.line_smoothing_enabled = True
    node.modifiers.append(dislo)
    data_dxa = node.compute()
    cell = data_dxa.cell
    cell_volume = float(cell.volume)

    # Translations to neighbouring periodic images for DXA-grain association.
    cell_matrix = np.asarray(cell)[:, :3]
    cell_vector_a = cell_matrix[:, 0]
    cell_vector_b = cell_matrix[:, 1]
    cell_vector_c = cell_matrix[:, 2]

    range_a = (-1, 0, 1) if cell.pbc[0] else (0,)
    range_b = (-1, 0, 1) if cell.pbc[1] else (0,)
    range_c = (-1, 0, 1) if cell.pbc[2] else (0,)

    periodic_shifts = []

    for shift_a in range_a:
        for shift_b in range_b:
            for shift_c in range_c:
                shift = shift_a * cell_vector_a + shift_b * cell_vector_b + shift_c * cell_vector_c
                periodic_shifts.append(shift)

    # Use the same network access as in the dislo.py and SFT scripts.
    network = data_dxa.dislocations
    records = segment_records(network)  # list of dictionaries containing information about all network dislocations

    node_positions, node_edges = build_dislocation_graph(records, cell) # node coordinates and dislocations associated with each node
    ribbon_candidates = candidate_dissociated_perfects(records, node_positions, node_edges, cell)

    # ============================================================
    # 9.2 Run PTM and segment HCP grains
    # ============================================================
    e_time = datetime.now()
    diff = e_time - s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('PTM and Grain Segmentation')
    s_time = datetime.now()

    # Identify local structures and segment connected HCP grains
    ptm = PolyhedralTemplateMatchingModifier(rmsd_cutoff = cutoff_poly)
    ptm.output_orientation = True
    ptm.output_interatomic_distance = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.OTHER].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = True
    node.modifiers.append(ptm)

    grain = GrainSegmentationModifier()
    grain.algorithm = GrainSegmentationModifier.Algorithm.GraphClusteringAuto
    grain.color_particles = False
    grain.handle_stacking_faults = False
    grain.min_grain_size = grain_min_size
    grain.orphan_adoption = False
    node.modifiers.append(grain)

    data = node.compute()

    # Particle-property arrays.
    positions = np.asarray(data.particles['Position']) # position array
    structure_type = np.asarray(data.particles['Structure Type']).astype(int)   # structure-type array
    particle_types = np.asarray(data.particles['Particle Type']).astype(int)    # particle-type array
    grain_id = np.asarray(data.particles['Cluster']).astype(int)   # grain-membership array

    # Map atom type numbers to names.
    ptype_prop = data.particles.particle_types
    type_names = { t.id: t.name.strip().upper() for t in ptype_prop.types } 

    # Identify HCP grains.
    grain_table = data.tables['grains']
    all_grain_ids = np.asarray(grain_table['Grain Identifier']).astype(int)
    grain_structure = np.asarray(grain_table['Structure Type']).astype(int)
    hcp_grain_ids = all_grain_ids[grain_structure == PolyhedralTemplateMatchingModifier.Type.HCP ]
  
    hcp_mask = (structure_type == PolyhedralTemplateMatchingModifier.Type.HCP) & np.isin(grain_id, hcp_grain_ids)
    hcp_indices = np.flatnonzero(hcp_mask)      # indices of HCP atoms belonging to an HCP grain

    # ============================================================
    # 9.3 Reconstruct dissociated Perfect dislocations
    # ============================================================
    e_time = datetime.now()
    diff = e_time - s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Search Perfect Dissociation and associated grains')
    s_time = datetime.now()
    
    ribbon_grain_ids = set()
    valid_ribbons = []
    used_shockley_edges = set()
    used_bubble_edges = set()
    used_dissociation_edges = set()
    bubble_rows_written = 0

    for candidate in ribbon_candidates:
        candidate_edges = set(candidate['path1_edges']).union(candidate['path2_edges']) # Shockley backbone of the lens
        candidate_bubble_edges = set(candidate['bubble_edges'])
        candidate_all_edges = candidate_edges | candidate_bubble_edges
        # Complete duplicates were removed during the search. Here, reject
        # distinct candidates that share backbone or bubble edges.
        if candidate_all_edges.intersection(used_dissociation_edges): continue

        # Search each lens separately for HCP atoms belonging to one grain.
        associated_grains = []
        associated_atoms = set()

        offset = 0
        for lens_midline, lens_Mwidth in zip(candidate['midline_parts'], candidate['width_max_parts']):
            number_of_lens_points = len(lens_midline)

            # Extract the two sides of the individual lens.
            lens_side1 = candidate['path1'][offset:offset + number_of_lens_points]
            lens_side2 = candidate['path2'][offset:offset + number_of_lens_points]
            offset += number_of_lens_points

            # Construct the elliptical cylinder used to search for HCP grains.
            # Use the central 80% of the midline as axis 1.
            central_midline = resample_polyline(lens_midline, 11)
            axis_start = central_midline[1]     # point at 10%
            axis_end = central_midline[-2]      # point at 90%
            A1_vector = axis_end - axis_start
            A1_length = np.linalg.norm(A1_vector)

            if A1_length <= 1.0e-8 or lens_Mwidth <= 1.0e-8: continue

            A1_direction = A1_vector / A1_length

            # Find the transverse direction at the point of maximum width (axis 2).
            local_widths = np.linalg.norm(lens_side2 - lens_side1, axis=1)
            widest_point = int(np.argmax(local_widths))
            A2_vector = lens_side2[widest_point] - lens_side1[widest_point]
            A2_vector -= (np.dot(A2_vector, A1_direction) * A1_direction) # make the width direction perpendicular to the length direction
            A2_length = np.linalg.norm(A2_vector)

            if A2_length <= 1.0e-8: continue

            A2_direction = A2_vector / A2_length

            # Unit vector normal to the lens plane (axis 3).
            A3_direction = np.cross(A1_direction, A2_direction)

            # Search for atoms within the volume.
            cylinder_center = 0.5 * (axis_start + axis_end)
            semi_length = 0.5 * A1_length
            semi_width = 0.5 *  A2_length
            half_height = 2.5   # half-height of 2.5 Angstrom gives a total height of 5 Angstrom

            lens_atoms = []
            for atom in hcp_indices:
                delta = np.asarray(cell.delta_vector(cylinder_center, positions[atom]))
                along_length = np.dot(delta, A1_direction)
                along_width = np.dot(delta, A2_direction)
                outside_plane = np.dot(delta, A3_direction)
                inside_ellipse = ((along_length / semi_length)**2 + (along_width / semi_width)**2 <= 1.0)
                inside_height = (abs(outside_plane) <= half_height)

                if inside_ellipse and inside_height: lens_atoms.append(int(atom))

            # Associate the lens only if it contains enough HCP atoms and all
            # atoms found belong to the same grain.
            if len(lens_atoms) >= grain_min_size / 2: 
                grains_inside = np.unique(grain_id[np.asarray(lens_atoms, dtype=int)])

                if len(grains_inside) == 1:
                    associated_grains.append(int(grains_inside[0]))
                    associated_atoms.update(lens_atoms)

        associated_grains = sorted(set(associated_grains))
        atoms = np.asarray(sorted(associated_atoms), dtype=int)

        ribbon_id = len(valid_ribbons) + 1
        ribbon_atoms = atoms 
        candidate['id'] = ribbon_id
        candidate['hcp_indices'] = ribbon_atoms
        candidate['hcp_grain_ids'] = associated_grains
        valid_ribbons.append(candidate)
        used_shockley_edges.update(candidate_edges)
        used_bubble_edges.update(candidate_bubble_edges)
        used_dissociation_edges.update(candidate_all_edges)
        ribbon_grain_ids.update(associated_grains)

        for edge in candidate_edges:
            records[edge]['role'] = 'DISSOCIATED_PERFECT'
            records[edge]['dissociation_id'] = ribbon_id

        for edge in candidate_bubble_edges:
            records[edge]['role'] = 'DISSOCIATION_BUBBLE'
            records[edge]['dissociation_id'] = ribbon_id

        perfect_start_id = records[candidate['perfect_start']]['id']
        perfect_end_id = records[candidate['perfect_end']]['id'] if candidate['perfect_end'] >= 0 else -1
        start_position = candidate['midline'][0]
        end_position = candidate['midline'][-1]
        start_node_text = '({:.8g}, {:.8g}, {:.8g})'.format(start_position[0], start_position[1], start_position[2])
        end_node_text = '({:.8g}, {:.8g}, {:.8g})'.format(end_position[0], end_position[1], end_position[2])
        path1_ids = ','.join(str(records[e]['id']) for e in candidate['path1_edges'])
        path2_ids = ','.join(str(records[e]['id']) for e in candidate['path2_edges'])
        grain_ids_text = ','.join(str(g) for g in associated_grains) if associated_grains else '-'
        burgers_text = ','.join('{:.8g}'.format(x) for x in candidate['equivalent_b'])
        coordinates = ';'.join('{:.8g}, {:.8g}, {:.8g}'.format(p[0], p[1], p[2]) for p in candidate['midline'])
        print(iteration, ribbon_id, start_node_text, end_node_text, candidate['end_kind'], candidate['n_lenses'], path1_ids, path2_ids, candidate['length1'], candidate['length2'], candidate['midline_length'], burgers_text, candidate['width_mean'], candidate['width_max'], perfect_start_id, perfect_end_id, grain_ids_text, len(ribbon_atoms), coordinates, sep='\t', file=pd_file)

        for bubble_id, bubble_info in enumerate(candidate['bubble_info'], start=1):
            bubble_edge = bubble_info['edge']
            bubble_record = records[bubble_edge]
            bubble_burgers = ','.join('{:.8g}'.format(x) for x in bubble_record['b'])
            backbone_edge = bubble_info['backbone_edge']
            backbone_segment_id = records[backbone_edge]['id'] if backbone_edge >= 0 else -1
            geometry_score = bubble_info['geometry_score']
            geometry_score_text = '-' if not np.isfinite(geometry_score) else '{:.8g}'.format(geometry_score)
            print(
                iteration,
                ribbon_id,
                bubble_id,
                bubble_info['kind'],
                bubble_info['node1'],
                bubble_info['node2'],
                bubble_record['id'],
                bubble_record['type'],
                bubble_record['length'],
                bubble_burgers,
                backbone_segment_id,
                geometry_score_text,
                sep='\t',
                file=pd_bubbles,
            )
            bubble_rows_written += 1
        
    # Record the frame even if no dissociation was found.
    if not valid_ribbons: print(iteration, *([0] * 18), sep="\t", file=pd_file)
    if bubble_rows_written == 0: print(iteration, *([0] * 11), sep="\t", file=pd_bubbles)

    excluded_dissociation_edges = used_shockley_edges | used_bubble_edges

    # Corrected segment-level dislocation network.
    for edge, rec in enumerate(records):
        # Exclude both the Shockley backbone and the accessory lines associated
        # with accepted dissociations.
        if edge in excluded_dissociation_edges: continue
        print(iteration, rec['id'], rec['type'], rec['length'], rec['length'] / cell_volume, ','.join('{:.8g}'.format(x)for x in rec['b']), rec['is_infinite'], rec['is_loop'], sep='\t', file=pd_workfile)
    # Add the equivalent Perfect segments for the removed Shockley-path pairs.
    for candidate in valid_ribbons:
        print(iteration, 'PD_{}'.format(candidate['id']), 'PERFECT', candidate['midline_length'], candidate['midline_length'] / cell_volume, ','.join('{:.8g}'.format(x) for x in candidate['equivalent_b']), 0, 0, sep='\t', file=pd_workfile)
    print(file=pd_workfile)

    equivalent_lengths = defaultdict(float)
    nonnano_lengths = defaultdict(float)
    loop_limits = [31.4, 62.8, 94.2, 125.6, 157.0, 188.5, 219.9, 251.3, 282.7, 314.2]
    loop_counts = [0] * 11
    n_loops_tot = 0
    l_loops_tot = 0.0
    l_loops_tot_nonano = 0.0

    # Shockley backbone edges and additional bubble lines belonging to accepted
    # dissociations are skipped.  Bubble lines generate no equivalent segment.
    for edge, rec in enumerate(records):
        if edge in excluded_dissociation_edges: continue
        equivalent_lengths[rec['type']] += rec['length']
        nano_loop = rec['is_infinite'] == 0 and rec['is_loop'] == 1 and rec['length'] <= 31.4
        if not nano_loop: nonnano_lengths[rec['type']] += rec['length']

        if rec['is_infinite'] == 0 and rec['is_loop'] == 1:
            placed = False
            for k, limit in enumerate(loop_limits):
                if rec['length'] <= limit:
                    loop_counts[k] += 1
                    placed = True
                    break
            if not placed: loop_counts[10] += 1
            n_loops_tot += 1
            l_loops_tot += rec['length']
            if rec['length'] > 31.4: l_loops_tot_nonano += rec['length']

    # Each midline is an open, finite Perfect segment and is therefore non-nano.
    for candidate in valid_ribbons:
        equivalent_lengths['PERFECT'] += candidate['midline_length']
        nonnano_lengths['PERFECT'] += candidate['midline_length']

    type_order = ['PERFECT', 'FRANK', 'SHOCKLEY', 'OTHER', 'STAIRROD', 'HIRTH']
    total = sum(equivalent_lengths[t] for t in type_order)
    total_no_other = total - equivalent_lengths['OTHER']
    total_no_nano = sum(nonnano_lengths[t] for t in type_order)
    total_no_other_no_nano = total_no_nano - nonnano_lengths['OTHER']
    corrected_values = [equivalent_lengths[t] for t in type_order] + [total, total_no_other, total_no_nano, total_no_other_no_nano]
    print(iteration, *corrected_values, sep='\t', file=pd_nano)
    print(iteration, *[value/cell_volume for value in corrected_values], sep='\t', file=pd_density)
    print(iteration, *loop_counts, n_loops_tot, l_loops_tot/cell_volume, l_loops_tot_nonano/cell_volume, sep='\t', file=pd_loops)

    # ============================================================
    # 9.4 Identify and classify planar HCP structures
    # ============================================================
    e_time = datetime.now()
    diff = e_time - s_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Identify SF candidates')
    s_time = datetime.now()

    grain_orientations = np.asarray(grain_table['Orientation'])
    orientation_by_grain = {int(grain): grain_orientations[i] for i, grain in enumerate(all_grain_ids)}
    remaining_hcpgrain_ids = [int(grain) for grain in hcp_grain_ids if int(grain) not in ribbon_grain_ids]

    # Interatomic Distance returned by PTM.
    interatomic_distance = np.asarray(data.particles['Interatomic Distance'])
    global_dnn = float(np.median(interatomic_distance))

    planes = []
    irregular_grain_ids = set()

    # Neighbour finder for grain unwrapping.
    grain_unwrap_cutoff = 1.9 * global_dnn # 1.9, approximately sqrt(3)*dnn, connects HCP-FCC-HCP configurations correctly
    grain_neighbor_finder = CutoffNeighborFinder(grain_unwrap_cutoff, data)

    # Identify irregular grains and classify the rest by number of layers.
    for current_grain_id in remaining_hcpgrain_ids:
        indices = np.flatnonzero((grain_id == current_grain_id) & (structure_type == PolyhedralTemplateMatchingModifier.Type.HCP))
        if len(indices) == 0: continue

        pos, number_of_components = unwrap_grain(indices, positions, cell, grain_neighbor_finder)

        # Do not classify a Grain ID composed of disconnected parts.
        if number_of_components != 1:
            irregular_grain_ids.add(current_grain_id)
            print(iteration, current_grain_id, len(indices), 0, '-', '-', '-', 0.0, 'DISCONNECTED_HCP_GRAIN', sep='\t', file=discarded_file)
            continue
    
        # Local {111} interplanar spacing.
        local_distances = interatomic_distance[indices]
        local_dnn = float(np.median(local_distances)) 
        d111 = sqrt(2.0 / 3.0) * local_dnn

        center = np.mean(pos, axis=0)
        singular, pca_axes, pca_normal = pca(pos - center)

        plane, irregular_result = analyze_hcp_grain_geometry(current_grain_id, indices, pos, center, singular, pca_axes, pca_normal, d111, local_dnn, particle_types, type_names, orientation_by_grain)

        if plane is None:
            print(iteration, current_grain_id, len(indices), irregular_result['n_layers'], irregular_result['layer_pattern'], irregular_result['layer_sizes_text'], irregular_result['layer_gaps_text'], irregular_result['minimum_layer_overlap'], ','.join(irregular_result['reasons']), sep='\t', file=discarded_file)
            irregular_grain_ids.add(current_grain_id)
            continue

        plane['id'] = len(planes) + 1
        plane['boundary'] = dislocation_association(plane, records, node_edges, cell, periodic_shifts)
        planes.append(plane)
    
    accepted_indices = [i for i, plane in enumerate(planes) if plane['boundary']['accepted']]

    # Pair parallel monolayers belonging to the same physical structure.
    monolayer_indices = [i for i in accepted_indices if planes[i]['n_layers'] == 1]
    monolayer_neighbors = {i: set() for i in monolayer_indices}
    cos_parallel = cos(radians(parallel_angle_tol_deg))
    
    for local_a in range(len(monolayer_indices)):
        a = monolayer_indices[local_a]
        pa = planes[a]

        for local_b in range(local_a + 1, len(monolayer_indices)):
            b = monolayer_indices[local_b]
            pb = planes[b]

            # The normals must be parallel; the absolute value admits opposite directions.
            if abs(np.dot(pa['normal'], pb['normal'])) < cos_parallel: continue

            # Move the centre and atoms of B to the periodic image containing A.
            aligned_center_b = pa['center'] + np.asarray(cell.delta_vector(pa['center'], pb['center']))
            shift_b = aligned_center_b - pb['center']
            positions_a = pa['positions']
            positions_b = pb['positions'] + shift_b

            # The two monolayers must be separated by approximately 2*d111 along the normal.
            center_delta = aligned_center_b - pa['center']
            normal_separation = abs(np.dot(center_delta, pa['normal']))
            mean_d111 = 0.5 * (pa['d111'] + pb['d111'])
            separation_d111 = normal_separation / mean_d111

            if abs(separation_d111 - 2.0) > monolayer_spacing_tolerance_d111: continue

            # Project both monolayers onto the same plane using the axes of A.
            relative_a = positions_a - pa['center']
            relative_b = positions_b - pa['center']

            points2d_a = np.column_stack((np.dot(relative_a, pa['axes'][:, 0]), np.dot(relative_a, pa['axes'][:, 1])))
            points2d_b = np.column_stack((np.dot(relative_b, pa['axes'][:, 0]), np.dot(relative_b, pa['axes'][:, 1])))

            # Compare the regions occupied by the two monolayers.
            mean_dnn = mean_d111 / sqrt(2.0 / 3.0)
            grid_step = 0.6 * mean_dnn

            cells_a = set((int(cell_index[0]), int(cell_index[1])) for cell_index in np.rint(points2d_a / grid_step).astype(int))
            cells_b = set((int(cell_index[0]), int(cell_index[1])) for cell_index in np.rint(points2d_b / grid_step).astype(int))

            expanded_a = set()
            expanded_b = set()

            for x, y in cells_a:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        expanded_a.add((x + dx, y + dy))

            for x, y in cells_b:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        expanded_b.add((x + dx, y + dy))

            fraction_a = sum(cell_index in expanded_b for cell_index in cells_a) / float(len(cells_a))
            fraction_b = sum(cell_index in expanded_a for cell_index in cells_b) / float(len(cells_b))
            monolayer_overlap = min(fraction_a, fraction_b)

            if monolayer_overlap < monolayer_overlap_min: continue

            # The two surfaces must be associated with substantially the same DXA lines.
            edges_a = set(pa['boundary']['dxa_ids'])
            edges_b = set(pb['boundary']['dxa_ids'])

            if not edges_a or not edges_b:continue

            common_edges = edges_a.intersection(edges_b)

            # Fraction of the smaller boundary contained in the larger one.
            smaller_boundary_size = min(len(edges_a), len(edges_b))
            shared_edge_fraction = (len(common_edges) / float(smaller_boundary_size))

            if shared_edge_fraction >= same_boundary_edge_fraction_min:
                monolayer_neighbors[a].add(b)
                monolayer_neighbors[b].add(a)
            else:
                pa['review_with'].append(pb['grain_id'])
                pb['review_with'].append(pa['grain_id'])
                pa['max_failed_edge_fraction'] = max(pa['max_failed_edge_fraction'], shared_edge_fraction)
                pb['max_failed_edge_fraction'] = max(pb['max_failed_edge_fraction'], shared_edge_fraction)

    # Build structures by transitive grouping: if A is associated with B and B
    # with C, the resulting structure is [A, B, C].
    structure_groups = []
    grouped_indices = set()

    for i in accepted_indices:
        if planes[i]['n_layers'] != 1:
            structure_groups.append([i])
            grouped_indices.add(i)

    visited_monolayers = set()

    for root in monolayer_indices:
        if root in visited_monolayers: continue

        group = []
        stack = [root]
        visited_monolayers.add(root)

        while stack:
            current = stack.pop()
            group.append(current)

            for other in monolayer_neighbors[current]:
                if other not in visited_monolayers:
                    visited_monolayers.add(other)
                    stack.append(other)

        group = sorted(group)
        structure_groups.append(group)
        grouped_indices.update(group)

    # Geometrically regular grains with a rejected DXA boundary remain
    # individual planar structures.
    for i in range(len(planes)):
        if i not in grouped_indices:
            structure_groups.append([i])
            grouped_indices.add(i)

    # Identify loops using only the native DXA is_loop flag.
    loop_structure_ids = set()
    loop_plane_indices = set()

    for temporary_structure_id, group in enumerate(structure_groups, 1):
        # Combine the dislocations associated with all grains in the structure.
        edge_set = set()

        for i in group: edge_set.update(planes[i]['boundary']['edge_set'])

        # Select only dislocations classified directly by DXA as closed loops.
        native_loop_edges = {edge for edge in edge_set if records[edge]['is_loop'] == 1}

        # Store the provisional structure ID in its component grains.
        for i in group: planes[i]['structure_id'] = temporary_structure_id

        # A structure is a loop only if it contains at least one DXA line with
        # the native is_loop flag set.
        if native_loop_edges:
            loop_structure_ids.add(temporary_structure_id)
            loop_plane_indices.update(group)

    # Identify three-dimensional structures.
    possible_3d = [i for i in accepted_indices if i not in loop_plane_indices and len(planes[i]['indices']) <= 300]
    stair_neighbors = {i: set() for i in possible_3d}
    shared_stairrods = {}

    for ia in range(len(possible_3d)):
        a = possible_3d[ia]
        stair_a = {e for e in planes[a]['boundary']['edge_set'] if records[e]['type'] == 'STAIRROD'}
        for ib in range(ia + 1, len(possible_3d)):
            b = possible_3d[ib]
            stair_b = {e for e in planes[b]['boundary']['edge_set'] if records[e]['type'] == 'STAIRROD'}
            shared = stair_a.intersection(stair_b)
            if shared:
                stair_neighbors[a].add(b)
                stair_neighbors[b].add(a)
                shared_stairrods[(min(a, b), max(a, b))] = shared

    three_d_groups = []
    visited_3d = set()
    for root in possible_3d:
        if root in visited_3d or len(stair_neighbors[root]) < 2: continue
        stack = [root]
        group = []
        visited_3d.add(root)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in stair_neighbors[current]:
                if other not in visited_3d:
                    visited_3d.add(other)
                    stack.append(other)
        # Keep the 2-core: every retained face must share Stair-rods with at least two other faces of the same candidate object.
        core = set(group)
        changed = True
        while changed:
            changed = False
            remove = [i for i in core if len(stair_neighbors[i].intersection(core)) < 2]
            if remove:
                core.difference_update(remove)
                changed = True
        group = sorted(core)
        if len(group) < 3: continue
        enough_shared_stairrods = True
        for i in group:
            shared_by_plane = set()
            for other in stair_neighbors[i].intersection(group): shared_by_plane.update(shared_stairrods[(min(i, other), max(i, other))])
            if len(shared_by_plane) < 1:
                enough_shared_stairrods = False
                break
        if not enough_shared_stairrods: continue

        # Best common point of the normal lines through the grain centers.
        reference_center = planes[group[0]]['center']
        aligned_centers = {i: reference_center + np.asarray(cell.delta_vector(reference_center, planes[i]['center'])) for i in group}
        matrix = np.zeros((3, 3))
        vector = np.zeros(3)
        for i in group:
            normal = planes[i]['normal'] / np.linalg.norm(planes[i]['normal'])
            projector = np.eye(3) - np.outer(normal, normal)
            center = aligned_centers[i]
            matrix += projector
            vector += np.dot(projector, center)
        apex = np.dot(np.linalg.pinv(matrix), vector)
        distances = []
        for i in group:
            normal = planes[i]['normal'] / np.linalg.norm(planes[i]['normal'])
            distances.append(np.linalg.norm(np.cross(apex - aligned_centers[i], normal)))
        nonparallel = any(abs(np.dot(planes[group[a]]['normal'], planes[group[b]]['normal'])) < cos_parallel for a in range(len(group)) for b in range(a + 1, len(group)))
        if nonparallel and max(distances) <= 5: three_d_groups.append(group)
        
    three_d_plane_indices = set(i for group in three_d_groups for i in group)

    # ============================================================
    # 9.5 Final classification and output
    # ============================================================
    structure_id = 0
    used_plane_indices = set()

    # Record 3D structures first because they take precedence over SF and PL.
    for group in three_d_groups:
        structure_id += 1
        group = sorted(set(group))
        used_plane_indices.update(group)

        for i in group:
            plane = planes[i]
            boundary = plane['boundary']
            counts = plane['composition']['counts']
            dxa_ids_text = ','.join(str(value) for value in boundary['dxa_ids']) if boundary['dxa_ids'] else '-'

            print(iteration, structure_id, '3D', 'SHARED_STAIRRODS', plane['grain_id'],len(plane['indices']), plane['n_layers'], plane['layer_pattern'], plane['center'][0], plane['center'][1], plane['center'][2], plane['normal'][0], plane['normal'][1], plane['normal'][2], plane['thickness'], plane['area_hull'], plane['perimeter_hull'], dxa_ids_text, boundary['total'], boundary['coverage'], int(boundary['connected']), int(boundary['closed']), counts['FE'], counts['CR'], counts['NI'], sep='\t', file=grains_file)
        print(file=grains_file)

    # Then analyse planar structures not already assigned to a 3D structure.
    geometry_rows_written = 0
    for original_group in structure_groups:
        group = [i for i in original_group if i not in used_plane_indices]

        if not group:  continue

        structure_id += 1
        used_plane_indices.update(group)

        # Total number of HCP layers in the structure.
        effective_layers = sum(planes[i]['n_layers'] for i in group)

        # Accept the structure boundary only if it is accepted for every
        # component grain.
        group_has_accepted_boundary = all(planes[i]['boundary']['accepted'] for i in group)
        group_is_loop = any(i in loop_plane_indices for i in group)

        # Combine the DXA portions, using the maximum length to avoid counting
        # a dislocation shared by multiple monolayers more than once.
        structure_edge_lengths = {}

        for i in group:
            for edge, length in planes[i]['boundary']['edge_lengths'].items():
                structure_edge_lengths[edge] = max(structure_edge_lengths.get(edge, 0.0), length)

        structure_type_lengths = defaultdict(float)

        for edge, length in structure_edge_lengths.items():
            structure_type_lengths[records[edge]['type']] += length

        structure_dxa_length = float(sum(structure_edge_lengths.values()))
        shockley_stairrod_fraction = (structure_type_lengths['SHOCKLEY'] + structure_type_lengths['STAIRROD']) / max(structure_dxa_length, 1.0e-12)

        # SF/PL classification.
        is_sf = (
            group_has_accepted_boundary
            and effective_layers >= 2
            and not group_is_loop
            and shockley_stairrod_fraction >= 0.7
        )

        if is_sf:
            structure_class = 'SF'
            if len(group) > 1 and all(planes[i]['n_layers'] == 1 for i in group): subtype = 'PAIRED_MONOLAYERS'
            else: subtype = 'MULTILAYER'
        else:
            structure_class = 'PL'
            if group_is_loop: subtype = 'LO'
            elif not group_has_accepted_boundary: subtype = 'INCOMPLETE_DXA_BOUNDARY'
            elif effective_layers == 1: subtype = 'MONOLAYER'
            else: subtype = 'NON_SF_DXA_COMPOSITION'
        # Write one row per component grain to SF_hcp_grain_details.tmp.
        for i in group:
            plane = planes[i]
            boundary = plane['boundary']
            counts = plane['composition']['counts']
            dxa_ids_text = ','.join(str(value) for value in boundary['dxa_ids']) if boundary['dxa_ids'] else '-'

            print(iteration, structure_id, structure_class, subtype, plane['grain_id'], len(plane['indices']), plane['n_layers'], plane['layer_pattern'], plane['center'][0], plane['center'][1], plane['center'][2], plane['normal'][0], plane['normal'][1], plane['normal'][2], plane['thickness'], plane['area_hull'], plane['perimeter_hull'], dxa_ids_text, boundary['total'], boundary['coverage'], int(boundary['connected']), int(boundary['closed']), counts['FE'], counts['CR'], counts['NI'], sep='\t', file=grains_file)

        print(file=grains_file)

        # In-plane geometry using the measurements already calculated for each grain.
        hull_areas = []
        hull_perimeters = []
        hull_max_dists = []
        pca_s0_dists = []
        pca_s1_dists = []

        for i in group:
            plane = planes[i]
            hull = plane['hull']

            hull_areas.append(plane['area_hull'])
            hull_perimeters.append(plane['perimeter_hull'])

            # Extent of the hull projections along S0 and S1.
            pca_s0_dist = float(np.ptp(hull[:, 0]))
            pca_s1_dist = float(np.ptp(hull[:, 1]))
            pca_s0_dists.append(pca_s0_dist)
            pca_s1_dists.append(pca_s1_dist)

            hull_max_dist = 0.0

            for a in range(len(hull)):
                for b in range(a + 1, len(hull)):
                    hull_max_dist = max(hull_max_dist, float(np.linalg.norm(hull[a] - hull[b])))

            hull_max_dists.append(hull_max_dist)

        # Associated grains represent different layers of the same structure;
        # use the mean of their in-plane dimensions.
        hull_area = float(np.mean(hull_areas))
        hull_perimeter = float(np.mean(hull_perimeters))
        hull_max_dist = float(np.mean(hull_max_dists))
        structure_pca_s0_dist = float(np.mean(pca_s0_dists))
        structure_pca_s1_dist = float(np.mean(pca_s1_dists))

        # Overall structure thickness.
        if len(group) == 1: structure_thickness = planes[group[0]]['thickness'] # the grain already contains all its HCP layers

        else:
            # For separate monolayers, include the distance between their centres.
            reference_plane = planes[group[0]]
            reference_center = reference_plane['center']
            normal = reference_plane['normal']

            lower_limits = []
            upper_limits = []

            for i in group:
                center_delta = np.asarray(cell.delta_vector(reference_center, planes[i]['center']))

                center_height = np.dot(center_delta, normal)
                half_thickness = 0.5 * planes[i]['thickness']

                lower_limits.append(center_height - half_thickness)
                upper_limits.append(center_height + half_thickness)

            structure_thickness = float(max(upper_limits) - min(lower_limits))

        # Ratio of associated DXA length to HCP perimeter.
        boundary_coverage = ( structure_dxa_length / max(hull_perimeter, 1.0e-12))

        # Sum the atom counts and compositions of the component grains.
        total_atoms = sum(len(planes[i]['indices']) for i in group)
        structure_counts = dict((name, 0) for name in ('FE', 'CR', 'NI'))

        for i in group:
            counts = planes[i]['composition']['counts']
            for name in structure_counts: structure_counts[name] += counts[name]

        if group_is_loop: structure_type = 'LO'
        else: structure_type = structure_class

        grain_ids_text = ','.join(str(grain_id) for grain_id in sorted(int(planes[i]['grain_id']) for i in group))

        # Write one row for the complete physical 2D structure.
        print(iteration, structure_id, structure_type, len(group), grain_ids_text, total_atoms, effective_layers, structure_thickness, hull_area, hull_perimeter, hull_max_dist, structure_pca_s0_dist, structure_pca_s1_dist, structure_dxa_length, boundary_coverage, structure_type_lengths['PERFECT'], structure_type_lengths['FRANK'], structure_type_lengths['SHOCKLEY'], structure_type_lengths['OTHER'], structure_type_lengths['STAIRROD'], structure_type_lengths['HIRTH'], structure_counts['FE'], structure_counts['CR'], structure_counts['NI'], sep='\t', file=geometry_file)
        geometry_rows_written += 1
    
    if geometry_rows_written == 0:
        print(iteration, *([0] * 23), sep='\t', file=geometry_file)

    for handle in (pd_workfile, pd_nano, pd_density, pd_loops, pd_file, pd_bubbles, grains_file, geometry_file, discarded_file): handle.flush()

    # ============================================================
    # 9.6 Report analysis and total elapsed times
    # ============================================================
    e_time = datetime.now()
    diff = e_time - s_time
    diff_tot = e_time - start_time
    print('      elapsed ---------------------->', diff.total_seconds())
    print('Finished; total elapsed time ------>', diff_tot.total_seconds())
    print()
    s_time = datetime.now()


pd_workfile.close()
pd_nano.close()
pd_density.close()
pd_loops.close()
pd_file.close()
pd_bubbles.close()
grains_file.close()
geometry_file.close()
discarded_file.close()
