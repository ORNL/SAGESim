#!/usr/bin/env python3
"""
METIS-based network partitioning utilities for SAGESim.

This module provides functions to generate network partitions using METIS
to optimize multi-worker performance by minimizing cross-worker communication.

Usage:
    from sagesim.partition_utils import generate_metis_partition

    # Generate partition for a NetworkX graph
    partition = generate_metis_partition(graph, num_workers=4)

    # Save partition
    save_partition(partition, "partition.pkl")

    # Load in your model
    model.load_partition("partition.pkl")
"""

from typing import Dict, Optional
import pickle
import json
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    metis = None


def generate_metis_partition(
    graph,
    num_workers: int,
    node_id_attr: str = None,
    recursive: bool = True,
) -> Dict[int, int]:
    """Generate a network partition using METIS.

    Uses METIS to partition a graph into balanced parts that minimize edge cuts,
    which reduces cross-worker communication overhead in distributed simulations.

    Args:
        graph: NetworkX graph to partition
        num_workers: Number of partitions (MPI workers)
        node_id_attr: Node attribute containing agent IDs. If None, uses node labels.
        recursive: Use recursive bisection (True) or k-way partitioning (False)

    Returns:
        Dictionary mapping agent_id -> rank (worker assignment)

    Raises:
        ImportError: If metis or networkx are not installed
        ValueError: If graph is empty or num_workers < 2

    Example:
        >>> import networkx as nx
        >>> G = nx.karate_club_graph()
        >>> partition = generate_metis_partition(G, num_workers=4)
        >>> print(f"Assigned {len(partition)} nodes to 4 workers")
    """
    if not METIS_AVAILABLE:
        raise ImportError(
            "METIS is not installed. Install with: pip install metis\n"
            "On Frontier, also run: module load metis/5.1.0\n"
            "And set: export METIS_DLL=$OLCF_METIS_ROOT/lib/libmetis.so"
        )

    if nx is None:
        raise ImportError("NetworkX is required. Install with: pip install networkx")

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty")

    if num_workers < 2:
        raise ValueError(f"num_workers must be >= 2, got {num_workers}")

    # Convert to undirected for partitioning if needed
    if graph.is_directed():
        graph_undirected = graph.to_undirected()
    else:
        graph_undirected = graph

    # Partition the graph
    (edgecuts, parts) = metis.part_graph(graph_undirected, num_workers, recursive=recursive)

    # Create mapping from node ID to rank
    partition = {}
    for node, rank in zip(graph.nodes(), parts):
        if node_id_attr is not None:
            agent_id = graph.nodes[node][node_id_attr]
        else:
            agent_id = node
        partition[int(agent_id)] = int(rank)

    return partition


def save_partition(partition: Dict[int, int], filepath: str, format: str = "auto") -> None:
    """Save partition to file.

    Args:
        partition: Dictionary mapping agent_id -> rank
        filepath: Output file path
        format: File format ('pickle', 'json', 'numpy', 'text', or 'auto')
    """
    from pathlib import Path

    path = Path(filepath)

    # Auto-detect format from extension
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix in [".pkl", ".pickle"]:
            format = "pickle"
        elif suffix == ".json":
            format = "json"
        elif suffix == ".npy":
            format = "numpy"
        elif suffix == ".txt":
            format = "text"
        else:
            format = "pickle"  # default

    # Save based on format
    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(partition, f)
    elif format == "json":
        with open(filepath, "w") as f:
            json.dump(partition, f, indent=2)
    elif format == "numpy":
        # Convert dict to array (assumes sequential agent IDs starting from 0)
        max_id = max(partition.keys())
        arr = np.full(max_id + 1, -1, dtype=np.int32)
        for agent_id, rank in partition.items():
            arr[agent_id] = rank
        np.save(filepath, arr)
    elif format == "text":
        with open(filepath, "w") as f:
            f.write("# agent_id rank\n")
            for agent_id, rank in sorted(partition.items()):
                f.write(f"{agent_id} {rank}\n")
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved partition to {filepath} (format: {format})")


def analyze_partition(partition: Dict[int, int], graph=None) -> None:
    """Print partition quality metrics.

    Args:
        partition: Dictionary mapping agent_id -> rank
        graph: Optional NetworkX graph to compute edge cut statistics
    """
    from collections import Counter

    rank_counts = Counter(partition.values())
    num_workers = len(rank_counts)
    total_agents = len(partition)

    print(f"Partition Analysis:")
    print(f"  Total agents: {total_agents}")
    print(f"  Number of workers: {num_workers}")
    print(f"  Agents per worker:")
    for rank in sorted(rank_counts.keys()):
        count = rank_counts[rank]
        pct = 100 * count / total_agents
        print(f"    Rank {rank}: {count:6d} agents ({pct:5.1f}%)")

    # Balance metrics
    max_agents = max(rank_counts.values())
    min_agents = min(rank_counts.values())
    avg_agents = total_agents / num_workers
    imbalance = (max_agents - min_agents) / avg_agents if avg_agents > 0 else 0

    print(f"  Balance:")
    print(f"    Min: {min_agents}, Max: {max_agents}, Avg: {avg_agents:.1f}")
    print(f"    Imbalance: {imbalance:.2%}")

    # Edge cut analysis (if graph provided)
    if graph is not None and nx is not None:
        edge_cuts = 0
        total_edges = graph.number_of_edges()

        for u, v in graph.edges():
            u_rank = partition.get(u, -1)
            v_rank = partition.get(v, -1)
            if u_rank != v_rank:
                edge_cuts += 1

        if total_edges > 0:
            cut_ratio = edge_cuts / total_edges
            print(f"  Edge cuts: {edge_cuts}/{total_edges} ({cut_ratio:.1%})")


# Example usage function
def example():
    """Example of how to use METIS partitioning with SAGESim."""
    if not METIS_AVAILABLE or nx is None:
        print("This example requires metis and networkx")
        return

    # Create a test graph
    G = nx.karate_club_graph()
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Generate partition
    num_workers = 4
    partition = generate_metis_partition(G, num_workers)

    # Analyze partition quality
    analyze_partition(partition, G)

    # Save partition
    save_partition(partition, "example_partition.pkl")

    print("\nTo use this partition in SAGESim:")
    print("  model = SAGESim(...)")
    print("  model.load_partition('example_partition.pkl')")
    print("  # ... create agents ...")
    print("  model.setup()")


if __name__ == "__main__":
    example()
