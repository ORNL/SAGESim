#!/usr/bin/env python3
"""
Example script for generating network partitions for SAGESim.

This script demonstrates how to create partitions using different methods:
1. METIS - Graph-based partitioning (recommended for best performance)
2. NetworkX Communities - Community detection based partitioning
3. Random - Random assignment (for baseline comparison)

Usage:
    python generate_partition.py --input network.graphml --output partition.pkl --method metis --workers 4
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np


def partition_with_metis(graph: nx.Graph, num_workers: int) -> Dict[int, int]:
    """Partition graph using METIS for minimal edge cuts.

    Args:
        graph: NetworkX graph (will be converted to undirected)
        num_workers: Number of MPI workers

    Returns:
        Dictionary mapping node_id -> rank
    """
    try:
        import metis
    except ImportError:
        raise ImportError(
            "METIS not installed. Install with: pip install metis\n"
            "Note: Requires METIS library (http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)"
        )

    print(f"Running METIS partitioning with {num_workers} partitions...")

    # Convert to undirected graph for METIS
    if graph.is_directed():
        G_undirected = graph.to_undirected()
    else:
        G_undirected = graph

    # Create adjacency list (METIS format)
    node_list = list(G_undirected.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    adjacency = []
    for node in node_list:
        neighbors = [node_to_idx[neighbor] for neighbor in G_undirected.neighbors(node)]
        adjacency.append(neighbors)

    # Run METIS
    _, partition_array = metis.part_graph(
        adjacency,
        nparts=num_workers,
        recursive=True
    )

    # Convert back to original node IDs
    partition = {node_list[idx]: int(rank) for idx, rank in enumerate(partition_array)}

    return partition


def partition_with_communities(graph: nx.Graph, num_workers: int) -> Dict[int, int]:
    """Partition graph using community detection.

    Args:
        graph: NetworkX graph
        num_workers: Number of MPI workers

    Returns:
        Dictionary mapping node_id -> rank
    """
    from networkx.algorithms import community

    print(f"Running community detection...")

    # Convert to undirected for community detection
    if graph.is_directed():
        G_undirected = graph.to_undirected()
    else:
        G_undirected = graph

    # Detect communities
    communities = list(community.greedy_modularity_communities(G_undirected))

    print(f"Found {len(communities)} communities")

    # Assign communities to workers
    # Sort communities by size (largest first) for better load balancing
    communities_sorted = sorted(communities, key=len, reverse=True)

    # Round-robin assignment of communities to workers
    partition = {}
    worker_loads = [0] * num_workers  # Track number of nodes per worker

    for comm in communities_sorted:
        # Assign this community to the least loaded worker
        target_worker = worker_loads.index(min(worker_loads))

        for node in comm:
            partition[node] = target_worker

        worker_loads[target_worker] += len(comm)

    return partition


def partition_random(graph: nx.Graph, num_workers: int, seed: int = 42) -> Dict[int, int]:
    """Random partition (for baseline comparison).

    Args:
        graph: NetworkX graph
        num_workers: Number of MPI workers
        seed: Random seed

    Returns:
        Dictionary mapping node_id -> rank
    """
    print(f"Generating random partition with seed={seed}...")

    np.random.seed(seed)
    partition = {}

    for node in graph.nodes():
        partition[node] = np.random.randint(0, num_workers)

    return partition


def partition_round_robin(graph: nx.Graph, num_workers: int) -> Dict[int, int]:
    """Round-robin partition (SAGESim default).

    Args:
        graph: NetworkX graph
        num_workers: Number of MPI workers

    Returns:
        Dictionary mapping node_id -> rank
    """
    print(f"Generating round-robin partition...")

    partition = {}
    nodes = list(graph.nodes())

    for idx, node in enumerate(nodes):
        partition[node] = idx % num_workers

    return partition


def analyze_partition(graph: nx.Graph, partition: Dict[int, int], num_workers: int):
    """Analyze partition quality."""
    print("\n" + "=" * 60)
    print("PARTITION QUALITY ANALYSIS")
    print("=" * 60)

    # Load balance
    agents_per_rank = {}
    for rank in partition.values():
        agents_per_rank[rank] = agents_per_rank.get(rank, 0) + 1

    num_agents = len(partition)
    max_agents = max(agents_per_rank.values()) if agents_per_rank else 0
    min_agents = min(agents_per_rank.values()) if agents_per_rank else 0
    avg_agents = num_agents / num_workers if num_workers > 0 else 0
    imbalance = (max_agents - min_agents) / avg_agents if avg_agents > 0 else 0

    print(f"\nLoad Balance:")
    print(f"  Agents per rank: {dict(sorted(agents_per_rank.items()))}")
    print(f"  Max: {max_agents}, Min: {min_agents}, Avg: {avg_agents:.1f}")
    print(f"  Imbalance: {imbalance:.2%}")

    # Edge cut analysis
    total_edges = 0
    cross_worker_edges = 0

    for u, v in graph.edges():
        if u in partition and v in partition:
            total_edges += 1
            if partition[u] != partition[v]:
                cross_worker_edges += 1

    edge_cut_ratio = cross_worker_edges / total_edges if total_edges > 0 else 0

    print(f"\nEdge Cut Analysis:")
    print(f"  Total edges: {total_edges}")
    print(f"  Cross-worker edges: {cross_worker_edges}")
    print(f"  Edge cut ratio (P_cross): {edge_cut_ratio:.4f}")

    # Performance estimation (from SAGESIM_OVERHEAD_ANALYSIS.md)
    avg_neighbors = total_edges / num_agents if num_agents > 0 else 0

    print(f"\nExpected Performance:")
    print(f"  Average neighbors per agent: {avg_neighbors:.1f}")

    if edge_cut_ratio < 0.10:
        print(f"  ✓ EXCELLENT partition (P_cross < 0.10)")
        print(f"    Expected MPI overhead: ~30-60 ms/contextualization")
        print(f"    Expected weak scaling: 90-95% efficiency")
    elif edge_cut_ratio < 0.20:
        print(f"  ✓ GOOD partition (P_cross < 0.20)")
        print(f"    Expected MPI overhead: ~60-120 ms/contextualization")
        print(f"    Expected weak scaling: 75-90% efficiency")
    elif edge_cut_ratio < 0.40:
        print(f"  ⚠ MODERATE partition (P_cross < 0.40)")
        print(f"    Expected MPI overhead: ~120-250 ms/contextualization")
        print(f"    Expected weak scaling: 50-75% efficiency")
    else:
        print(f"  ✗ POOR partition (P_cross >= 0.40)")
        print(f"    Expected MPI overhead: >250 ms/contextualization")
        print(f"    Expected weak scaling: <50% efficiency")
        print(f"    Consider using METIS or community-based partitioning")

    print("=" * 60 + "\n")


def save_partition(partition: Dict[int, int], output_file: str, format: str = "auto"):
    """Save partition to file.

    Args:
        partition: Dictionary mapping node_id -> rank
        output_file: Output file path
        format: Output format ('pickle', 'json', 'numpy', 'text', or 'auto')
    """
    output_path = Path(output_file)

    # Auto-detect format
    if format == "auto":
        suffix = output_path.suffix.lower()
        if suffix in [".pkl", ".pickle"]:
            format = "pickle"
        elif suffix == ".json":
            format = "json"
        elif suffix == ".npy":
            format = "numpy"
        elif suffix in [".txt", ".dat"]:
            format = "text"
        else:
            format = "pickle"  # Default
            print(f"Warning: Unknown extension '{suffix}', using pickle format")

    # Save based on format
    if format == "pickle":
        with open(output_path, "wb") as f:
            pickle.dump(partition, f)
        print(f"Saved partition to {output_path} (pickle format)")

    elif format == "json":
        # Convert keys to strings for JSON
        json_partition = {str(k): v for k, v in partition.items()}
        with open(output_path, "w") as f:
            json.dump(json_partition, f, indent=2)
        print(f"Saved partition to {output_path} (JSON format)")

    elif format == "numpy":
        # Convert to array (assumes sequential node IDs starting from 0)
        max_id = max(partition.keys())
        partition_array = np.zeros(max_id + 1, dtype=np.int32)
        for node_id, rank in partition.items():
            partition_array[node_id] = rank
        np.save(output_path, partition_array)
        print(f"Saved partition to {output_path} (NumPy format)")

    elif format == "text":
        with open(output_path, "w") as f:
            f.write("# agent_id rank\n")
            for node_id in sorted(partition.keys()):
                f.write(f"{node_id} {partition[node_id]}\n")
        print(f"Saved partition to {output_path} (text format)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate network partitions for SAGESim multi-worker optimization"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input network file (GraphML, GML, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output partition file"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["metis", "community", "random", "roundrobin"],
        default="metis",
        help="Partitioning method (default: metis)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        required=True,
        help="Number of MPI workers"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["pickle", "json", "numpy", "text", "auto"],
        default="auto",
        help="Output format (default: auto-detect from extension)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random partitioning (default: 42)"
    )

    args = parser.parse_args()

    # Load network
    print(f"Loading network from {args.input}...")
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Detect format and load
    suffix = input_path.suffix.lower()
    if suffix == ".graphml":
        graph = nx.read_graphml(input_path)
    elif suffix == ".gml":
        graph = nx.read_gml(input_path)
    elif suffix in [".edgelist", ".edges"]:
        graph = nx.read_edgelist(input_path)
    else:
        print(f"Error: Unsupported network format: {suffix}")
        print("Supported: .graphml, .gml, .edgelist")
        return 1

    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Generate partition
    if args.method == "metis":
        partition = partition_with_metis(graph, args.workers)
    elif args.method == "community":
        partition = partition_with_communities(graph, args.workers)
    elif args.method == "random":
        partition = partition_random(graph, args.workers, args.seed)
    elif args.method == "roundrobin":
        partition = partition_round_robin(graph, args.workers)

    # Analyze partition
    analyze_partition(graph, partition, args.workers)

    # Save partition
    save_partition(partition, args.output, args.format)

    print(f"\n✓ Partition generation complete!")
    print(f"\nTo use this partition in SAGESim:")
    print(f"  model.load_partition('{args.output}')")

    return 0


if __name__ == "__main__":
    exit(main())
