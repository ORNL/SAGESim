#!/usr/bin/env python3
"""
Quick test script for METIS on Frontier.

Run with:
    module load metis
    python test_metis_frontier.py
"""

import sys

# Test 1: Check if metis module can be imported
print("=" * 60)
print("Test 1: Checking if METIS Python module is available...")
print("=" * 60)

try:
    import metis
    print("✓ METIS module imported successfully")
    print(f"  METIS version: {metis.__version__}")
except ImportError as e:
    print(f"✗ Failed to import metis: {e}")
    print("\nTry: module load metis")
    sys.exit(1)

# Test 2: Simple graph partitioning test
print("\n" + "=" * 60)
print("Test 2: Testing METIS graph partitioning...")
print("=" * 60)

import networkx as nx

# Create a simple test graph
G = nx.karate_club_graph()
print(f"Created test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

try:
    # Partition into 2 parts
    edgecuts, parts = metis.part_graph(G, 2)
    print(f"✓ METIS partitioning successful!")
    print(f"  Edge cuts: {edgecuts}")
    print(f"  Partition sizes: {parts.count(0)} nodes in part 0, {parts.count(1)} nodes in part 1")

    # Partition into 4 parts
    edgecuts, parts = metis.part_graph(G, 4)
    print(f"✓ METIS 4-way partitioning successful!")
    print(f"  Edge cuts: {edgecuts}")
    for i in range(4):
        print(f"  Part {i}: {parts.count(i)} nodes")

except Exception as e:
    print(f"✗ METIS partitioning failed: {e}")
    sys.exit(1)

# Test 3: Test with SuperNeuroABM integration
print("\n" + "=" * 60)
print("Test 3: Testing SuperNeuroABM METIS integration...")
print("=" * 60)

try:
    from superneuroabm.io.nx import generate_metis_partition

    # Create a small test graph
    test_graph = nx.DiGraph()
    for i in range(100):
        test_graph.add_node(i, soma_breed="lif_soma", config="config_0")

    # Add some edges
    for i in range(99):
        test_graph.add_edge(i, i+1, synapse_breed="single_exp_synapse", config="no_learning_config_0")

    # Generate partition
    partition = generate_metis_partition(test_graph, num_workers=4)

    print(f"✓ SuperNeuroABM METIS partition generated successfully!")
    print(f"  Graph: {test_graph.number_of_nodes()} nodes, {test_graph.number_of_edges()} edges")
    print(f"  Partition size: {len(partition)} node assignments")

    # Count nodes per worker
    nodes_per_worker = {}
    for node_id, rank in partition.items():
        nodes_per_worker[rank] = nodes_per_worker.get(rank, 0) + 1

    print(f"  Nodes per worker: {nodes_per_worker}")

except Exception as e:
    print(f"✗ SuperNeuroABM integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Full MPI test (optional, run with mpirun)
print("\n" + "=" * 60)
print("Test 4: MPI environment check...")
print("=" * 60)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"✓ MPI initialized: rank {rank}/{size}")

    if size > 1:
        print(f"  Running with {size} MPI workers - partition will be broadcast")
    else:
        print(f"  Running with 1 worker - to test MPI, run with:")
        print(f"    mpirun -n 4 python {__file__}")

except Exception as e:
    print(f"⚠ MPI check skipped: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nMETIS is working correctly on Frontier.")
print("\nTo test with your actual network:")
print("  1. cd /home/xxz/superneuroabm_sgnn")
print("  2. Edit run_sgnn_sna.py to ensure partition_method='metis'")
print("  3. Run: srun -n 4 python run_sgnn_sna.py")
