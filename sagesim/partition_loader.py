"""
Partition file preprocessing for distributed model loading.

This module provides utilities to enrich raw partition files (from external
partitioners like METIS or DGL) with ghost edge info and remote agent ranks,
so each MPI rank can load its partition file and build the model independently.

Usage:
    # Single-node preprocessing (run once before distributed simulation)
    from sagesim.partition_loader import preprocess_partitions
    preprocess_partitions(
        raw_partition_dir="raw_partitions/",
        output_dir="enriched_partitions/",
        node_partition_file="node_partition.npy",
        num_partitions=8,
    )

    # Then in distributed simulation (each rank):
    model.load_partition(f"enriched_partitions/partition_{rank}.pkl")
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def preprocess_partitions(
    raw_partition_dir: str,
    output_dir: str,
    node_partition_file: str,
    num_partitions: int,
) -> None:
    """Enrich raw partition files with ghost edges and remote agent ranks.

    Streams partition files one at a time — does NOT load the complete network
    into memory. Memory usage: node_partition array (memory-mapped for large
    networks) + one partition file at a time + ghost edge lists.

    Raw partition files must contain:
    - 'agents': list of agent dicts with at least 'id' and 'breed'
    - 'edges': list of edge dicts with 'source' and 'target'
      (source is always a local agent, target may be remote)

    Enriched output adds:
    - 'remote_agent_ranks': dict mapping remote agent_id -> rank
    - Additional ghost edges: for each cross-partition edge (src on rank A,
      tgt on rank B), rank B gets a reverse edge (tgt -> src) so its local
      agent can read from the remote agent.

    :param raw_partition_dir: Directory containing raw_partition_{rank}.pkl files
    :param output_dir: Directory to write enriched partition_{rank}.pkl files
    :param node_partition_file: Path to numpy array mapping node_id -> rank.
        For very large networks, this file is loaded with memory mapping.
    :param num_partitions: Number of partition files (must match MPI world size)
    """
    raw_dir = Path(raw_partition_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load node partition mapping (memory-mapped for large networks)
    node_partition = np.load(node_partition_file, mmap_mode='r')

    # Pass 1: Scan all raw partitions to find cross-partition edges
    # For each cross-partition edge (src on rank r, tgt on rank t):
    #   rank t needs a reverse edge (tgt -> src) so tgt can read src's data
    extra_edges = defaultdict(list)        # target_rank -> list of extra edges
    extra_remote_refs = defaultdict(dict)  # target_rank -> {agent_id: rank}

    print(f"[preprocess] Pass 1: Scanning {num_partitions} partition files for cross-partition edges...")
    for r in range(num_partitions):
        raw_file = raw_dir / f"partition_{r}.pkl"
        with open(raw_file, 'rb') as f:
            partition = pickle.load(f)

        for edge in partition['edges']:
            src = edge['source']
            tgt = edge['target']
            if tgt < 0:
                continue  # skip external input sentinel
            tgt_rank = int(node_partition[tgt])
            if tgt_rank != r:
                # Target is on a different rank — that rank needs a reverse edge
                extra_edges[tgt_rank].append({'source': tgt, 'target': src})
                extra_remote_refs[tgt_rank][src] = r

        del partition  # free memory

    # Pass 2: Write enriched partition files
    print(f"[preprocess] Pass 2: Writing enriched partition files to {out_dir}...")
    for r in range(num_partitions):
        raw_file = raw_dir / f"partition_{r}.pkl"
        with open(raw_file, 'rb') as f:
            partition = pickle.load(f)

        # Build remote_agent_ranks for edges already in the file
        remote_ranks = {}
        for edge in partition['edges']:
            tgt = edge['target']
            if tgt < 0:
                continue
            tgt_rank = int(node_partition[tgt])
            if tgt_rank != r:
                remote_ranks[tgt] = tgt_rank

        # Add extra (reverse) edges from other partitions
        partition['edges'] = partition['edges'] + extra_edges.get(r, [])

        # Add remote refs for extra edges
        remote_ranks.update(extra_remote_refs.get(r, {}))
        partition['remote_agent_ranks'] = remote_ranks

        # Write enriched file
        out_file = out_dir / f"partition_{r}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(partition, f)

        del partition

    total_ghost_edges = sum(len(v) for v in extra_edges.values())
    total_remote_refs = sum(len(v) for v in extra_remote_refs.values())
    print(f"[preprocess] Done. Added {total_ghost_edges} ghost edges, "
          f"{total_remote_refs} remote agent refs across {num_partitions} partitions.")
