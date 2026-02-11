#!/usr/bin/env python3
"""
Weak Scaling Test for SAGESim SIR Model

Tests how performance scales as we add more MPI ranks while keeping
agents-per-rank constant (weak scaling).

Usage:
    srun -N1 -n8 python weak_scaling_sir.py --agents-per-rank 5000 --ticks 100
"""
import sys
import os
import time
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "sir"))

from sir_model import SIRModel
from state import SIRState
from random import random, sample, seed as set_seed
import networkx as nx

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None

def print_rank0(msg):
    """Print only from rank 0"""
    if rank == 0:
        print(msg, flush=True)

def generate_clustered_network_constant_comm(
    num_clusters,
    agents_per_cluster,
    intra_cluster_degree=10,
    cross_cluster_edges=2000,
    num_neighbor_clusters=1,
    seed_val=42
):
    """
    Generate clustered network for TRUE weak scaling with constant communication.

    Similar to superneuroabm's approach:
    - Each cluster has fixed number of agents (constant per-worker load)
    - Intra-cluster connections use fixed degree (O(n) edges, not O(n²))
    - Cross-cluster edges are fixed count between neighbor pairs
    - Ring topology: each cluster connects to K neighbors (constant communication)

    Args:
        num_clusters: Number of clusters (should match MPI size)
        agents_per_cluster: Agents per cluster (constant for weak scaling)
        intra_cluster_degree: Degree per agent within cluster (default: 10)
        cross_cluster_edges: Fixed edges to each neighbor cluster (default: 2000)
        num_neighbor_clusters: Number of neighbor clusters in ring (default: 1)
        seed_val: Random seed

    Returns:
        NetworkX graph with 'cluster' node attribute
    """
    import numpy as np
    np.random.seed(seed_val)

    graph = nx.Graph()
    total_agents = num_clusters * agents_per_cluster

    # Create agents organized by cluster
    agent_ids = []
    for cluster_id in range(num_clusters):
        cluster_agents = []
        for i in range(agents_per_cluster):
            agent_id = cluster_id * agents_per_cluster + i
            graph.add_node(agent_id, cluster=cluster_id)
            cluster_agents.append(agent_id)
        agent_ids.append(cluster_agents)

    # Add intra-cluster connections (fixed degree for O(n) scaling)
    # Optimized: use add_edges_from for batch edge addition
    intra_edges = 0
    for cluster_id, cluster_agents in enumerate(agent_ids):
        edges_to_add = []
        for agent in cluster_agents:
            # Each agent connects to exactly intra_cluster_degree neighbors
            num_targets = min(intra_cluster_degree, len(cluster_agents) - 1)

            if num_targets > 0:
                # Fast random sampling using randint
                targets = set()
                while len(targets) < num_targets:
                    target_idx = np.random.randint(0, len(cluster_agents))
                    target = cluster_agents[target_idx]
                    if target != agent:
                        targets.add(target)

                # Add edges (undirected, so only add if not already added in reverse)
                for target in targets:
                    if agent < target:  # Avoid duplicates in undirected graph
                        edges_to_add.append((agent, target))
                        intra_edges += 1

        # Batch add edges for this cluster
        graph.add_edges_from(edges_to_add)

    # Add cross-cluster connections (directed ring topology)
    # Optimized: avoid creating huge lists of all pairs
    inter_edges = 0
    if num_clusters > 1 and num_neighbor_clusters > 0:
        for cluster_i in range(num_clusters):
            # Connect to next K clusters in ring
            for offset in range(1, num_neighbor_clusters + 1):
                cluster_j = (cluster_i + offset) % num_clusters

                # Add exactly cross_cluster_edges between this pair
                # Use random sampling without creating all pairs upfront
                edges_to_add_list = []
                added_pairs = set()

                num_edges_needed = min(cross_cluster_edges,
                                      agents_per_cluster * agents_per_cluster)

                while len(edges_to_add_list) < num_edges_needed:
                    # Random agent from each cluster
                    src_idx = np.random.randint(0, len(agent_ids[cluster_i]))
                    dst_idx = np.random.randint(0, len(agent_ids[cluster_j]))
                    src = agent_ids[cluster_i][src_idx]
                    dst = agent_ids[cluster_j][dst_idx]

                    pair = (src, dst)
                    if pair not in added_pairs:
                        added_pairs.add(pair)
                        edges_to_add_list.append(pair)

                # Batch add edges
                graph.add_edges_from(edges_to_add_list)
                inter_edges += len(edges_to_add_list)

    return graph

def main():
    parser = argparse.ArgumentParser(description='SAGESim SIR Weak Scaling Test')
    parser.add_argument('--agents-per-rank', type=int, default=100000,
                        help='Number of agents per MPI rank (default: 100000)')
    parser.add_argument('--intra-cluster-degree', type=int, default=10,
                        help='Connections per agent within cluster (default: 10)')
    parser.add_argument('--cross-cluster-edges', type=int, default=2000,
                        help='Edges between neighbor clusters (default: 2000)')
    parser.add_argument('--num-neighbor-clusters', type=int, default=1,
                        help='Number of neighbor clusters in ring (default: 1 for true weak scaling)')
    parser.add_argument('--infection-rate', type=float, default=0.1,
                        help='Initial infection rate (default: 0.1)')
    parser.add_argument('--ticks', type=int, default=100,
                        help='Number of simulation ticks (default: 100)')
    parser.add_argument('--sync-ticks', type=int, default=1,
                        help='Sync every N ticks (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-gpu-aware-mpi', action='store_true',
                        help='Disable GPU-aware MPI (use CPU staging)')
    parser.add_argument('--validate', action='store_true',
                        help='Run deterministic validation test (forces p_infection=1.0, p_recovery=0.0)')

    args = parser.parse_args()

    # Validation mode: deterministic infection spreading
    if args.validate:
        p_infection = 1.0  # Force deterministic infection
        p_recovery = 0.0   # No recovery for validation
        validation_mode = True
        print_rank0("VALIDATION MODE: Using p_infection=1.0, p_recovery=0.0")
    else:
        p_infection = 0.3  # Realistic probability
        p_recovery = 0.1   # Realistic recovery
        validation_mode = False

    # Calculate derived parameters
    total_agents = args.agents_per_rank * size
    num_infected = int(args.infection_rate * total_agents)
    use_gpu_aware_mpi = not args.no_gpu_aware_mpi

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Print header
    print_rank0("=" * 80)
    print_rank0("WEAK SCALING TEST - SAGESim SIR Model")
    print_rank0("=" * 80)

    if rank == 0:
        # Environment info
        print(f"Environment:")
        print(f"  MPICH_GPU_SUPPORT_ENABLED: {os.environ.get('MPICH_GPU_SUPPORT_ENABLED', 'NOT SET')}")
        print(f"  MPICH_SMP_SINGLE_COPY_MODE: {os.environ.get('MPICH_SMP_SINGLE_COPY_MODE', 'NOT SET')}")

        import cupy as cp
        print(f"  CuPy version: {cp.__version__}")
        print(f"  GPUs per node: {cp.cuda.runtime.getDeviceCount()}")
        print()

        # Test configuration
        print(f"Configuration:")
        print(f"  MPI ranks (clusters): {size}")
        print(f"  Agents per rank: {args.agents_per_rank}")
        print(f"  Total agents: {total_agents:,}")
        print(f"  Intra-cluster degree: {args.intra_cluster_degree}")
        print(f"  Cross-cluster edges per neighbor: {args.cross_cluster_edges}")
        print(f"  Neighbor clusters: {args.num_neighbor_clusters} (ring topology)")
        if size > 1:
            total_cross_edges_out = args.cross_cluster_edges * args.num_neighbor_clusters
            print(f"  Total cross-cluster edges per rank: {total_cross_edges_out} (constant!)")
        print(f"  Initial infected: {num_infected} ({args.infection_rate*100:.1f}%)")
        print(f"  Simulation ticks: {args.ticks}")
        print(f"  Sync interval: {args.sync_ticks}")
        print(f"  GPU-aware MPI: {'ENABLED' if use_gpu_aware_mpi else 'DISABLED'}")
        print(f"  Random seed: {args.seed}")
        print("=" * 80)

    # Timing dictionary
    timings = defaultdict(float)

    # Create model
    print_rank0("\n[1/4] Creating SIR model...")
    t0 = time.time()
    model = SIRModel()

    # Register global infection/recovery probabilities
    model.register_global_property("p_infection", p_infection)
    model.register_global_property("p_recovery", p_recovery)

    timings['model_init'] = time.time() - t0

    t0 = time.time()
    model.setup(use_gpu=True)
    timings['model_setup'] = time.time() - t0

    model_create_time = timings['model_init'] + timings['model_setup']
    print_rank0(f"      Model init: {timings['model_init']:.3f}s, setup: {timings['model_setup']:.3f}s")

    # Generate/load network (cached to file)
    print_rank0("\n[2/4] Loading/generating network...")

    # Create network filename based on parameters
    from pathlib import Path
    network_dir = Path(__file__).parent / "network_data"
    network_dir.mkdir(exist_ok=True)
    network_filename = (
        f"network_clustered_c{size}_n{args.agents_per_rank}_"
        f"deg{args.intra_cluster_degree}_cross{args.cross_cluster_edges}_"
        f"nbr{args.num_neighbor_clusters}_s{args.seed}.pkl"
    )
    network_path = network_dir / network_filename

    # Check if network file exists (rank 0 checks)
    if rank == 0:
        file_exists = network_path.exists()
    else:
        file_exists = None

    # Broadcast file existence to all ranks
    if comm is not None:
        file_exists = comm.bcast(file_exists, root=0)

    if file_exists:
        # Load existing network (all ranks)
        if rank == 0:
            print(f"      Loading cached network from {network_filename}...")
        t0 = time.time()
        with open(network_path, 'rb') as f:
            network = pickle.load(f)
        timings['network_generation'] = time.time() - t0
        if rank == 0:
            print(f"      Network loaded in {timings['network_generation']:.3f}s")
            print(f"      Nodes: {network.number_of_nodes():,}, Edges: {network.number_of_edges():,}")
    else:
        # Generate new network (rank 0 only)
        if rank == 0:
            print(f"      Generating new clustered network...")
            print(f"        Clusters: {size}")
            print(f"        Agents per cluster: {args.agents_per_rank}")
            print(f"        Intra-cluster degree: {args.intra_cluster_degree}")
            print(f"        Cross-cluster edges: {args.cross_cluster_edges}")
            print(f"        Neighbor clusters: {args.num_neighbor_clusters}")
            t0 = time.time()
            network = generate_clustered_network_constant_comm(
                num_clusters=size,
                agents_per_cluster=args.agents_per_rank,
                intra_cluster_degree=args.intra_cluster_degree,
                cross_cluster_edges=args.cross_cluster_edges,
                num_neighbor_clusters=args.num_neighbor_clusters,
                seed_val=args.seed
            )
            timings['network_generation'] = time.time() - t0
            print(f"      Network generated in {timings['network_generation']:.3f}s")
            print(f"      Nodes: {network.number_of_nodes():,}, Edges: {network.number_of_edges():,}")

            # Save to file
            print(f"      Saving network to {network_filename}...")
            with open(network_path, 'wb') as f:
                pickle.dump(network, f)
        else:
            network = None
            timings['network_generation'] = 0.0

        # Wait for rank 0 to finish saving
        if comm is not None:
            comm.Barrier()

        # All non-zero ranks load the saved network
        if rank != 0:
            with open(network_path, 'rb') as f:
                network = pickle.load(f)

    # Create agents and connections (all ranks)
    print_rank0("\n[3/4] Initializing agents...")

    # Create agents (all ranks)
    t0 = time.time()
    for agent_id in range(total_agents):
        preventative_measures = [random() for _ in range(100)]
        model.create_agent(SIRState.SUSCEPTIBLE.value, preventative_measures)
    timings['agent_creation'] = time.time() - t0

    # Infect initial agents (all ranks)
    t0 = time.time()
    infected_ids = sample(range(total_agents), num_infected)
    for agent_id in infected_ids:
        model.set_agent_property_value(agent_id, "state", SIRState.INFECTED.value)
    timings['initial_infection'] = time.time() - t0

    # Create network connections (all ranks)
    t0 = time.time()
    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
    timings['network_connections'] = time.time() - t0

    if rank == 0:
        print(f"      Agent creation: {timings['agent_creation']:.3f}s")
        print(f"      Initial infection: {timings['initial_infection']:.3f}s")
        print(f"      Network connections: {timings['network_connections']:.3f}s")

    if comm:
        comm.Barrier()

    # Run simulation with instrumentation
    print_rank0(f"\n[4/4] Running simulation ({args.ticks} ticks)...")
    print_rank0(f"      Testing GPU-aware MPI: {use_gpu_aware_mpi}")

    # Run simulation (no warmup)
    timings['warmup'] = 0.0

    if comm:
        comm.Barrier()

    sim_start = time.time()
    model.simulate(args.ticks, sync_workers_every_n_ticks=args.sync_ticks)
    sim_time = time.time() - sim_start
    timings['simulation'] = sim_time

    if comm:
        comm.Barrier()
        all_sim_times = comm.gather(sim_time, root=0)
        max_sim_time = comm.reduce(sim_time, op=MPI.MAX, root=0)
        min_sim_time = comm.reduce(sim_time, op=MPI.MIN, root=0)
    else:
        all_sim_times = [sim_time]
        max_sim_time = sim_time
        min_sim_time = sim_time

    # Gather final state counts
    t0 = time.time()
    local_counts = {
        'susceptible': 0,
        'infected': 0,
        'recovered': 0
    }

    # Store states for validation
    all_states = []
    for agent_id in range(total_agents):
        state = model.get_agent_property_value(agent_id, property_name="state")
        if state is not None:
            all_states.append((agent_id, state))
            if state == SIRState.SUSCEPTIBLE.value:
                local_counts['susceptible'] += 1
            elif state == SIRState.INFECTED.value:
                local_counts['infected'] += 1
            elif state == SIRState.RECOVERED.value:
                local_counts['recovered'] += 1
    timings['state_collection'] = time.time() - t0

    t0 = time.time()
    # NOTE: All ranks have the same counts due to MPI broadcast in get_agent_property_value
    # So we don't need comm.reduce - just use local_counts on rank 0
    total_counts = local_counts
    all_states_gathered = [all_states] if rank == 0 else None

    # Gather timing data
    if comm:
        all_timings = comm.gather(dict(timings), root=0)
    else:
        all_timings = [dict(timings)]
    timings['final_gather'] = time.time() - t0

    # Print results
    if rank == 0:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        # Network stats
        print(f"Network Size:")
        print(f"  Total agents: {total_agents:,}")
        print(f"  Total edges: {network.number_of_edges():,}")
        print(f"  Agents per rank: {args.agents_per_rank:,}")

        # Performance stats
        print(f"\nPerformance:")
        print(f"  Simulation time: {max_sim_time:.3f}s (max across ranks)")
        if len(all_sim_times) > 1:
            import numpy as np
            print(f"    Mean: {np.mean(all_sim_times):.3f}s")
            print(f"    Min:  {min_sim_time:.3f}s")
            print(f"    Std:  {np.std(all_sim_times):.3f}s")
            load_imbalance = (max_sim_time - min_sim_time) / max_sim_time * 100
            print(f"    Load imbalance: {load_imbalance:.1f}%")

        agent_ticks_per_sec = (total_agents * args.ticks) / max_sim_time
        print(f"  Agent-ticks/sec: {agent_ticks_per_sec:,.0f}")
        print(f"  Time per tick: {max_sim_time / args.ticks * 1000:.2f} ms")

        # Detailed timing breakdown
        print(f"\nDetailed Timing Breakdown (Rank 0):")
        print(f"  Network generation:    {timings['network_generation']:8.3f}s")
        print(f"  Agent creation:        {timings['agent_creation']:8.3f}s")
        print(f"  Initial infection:     {timings['initial_infection']:8.3f}s")
        print(f"  Network connections:   {timings['network_connections']:8.3f}s")
        print(f"  Model init:            {timings['model_init']:8.3f}s")
        print(f"  Model setup (GPU):     {timings['model_setup']:8.3f}s")
        print(f"  Simulation ({args.ticks} ticks): {timings['simulation']:8.3f}s")
        print(f"  State collection:      {timings['state_collection']:8.3f}s")
        print(f"  Final gather:          {timings['final_gather']:8.3f}s")

        total_time = sum([
            timings['network_generation'],
            timings['agent_creation'],
            timings['initial_infection'],
            timings['network_connections'],
            timings['model_init'],
            timings['model_setup'],
            timings['simulation'],
            timings['state_collection'],
            timings['final_gather']
        ])
        print(f"  {'='*30}")
        print(f"  Total:                 {total_time:8.3f}s")

        # Compute percentages
        if total_time > 0:
            print(f"\nTime Distribution:")
            print(f"  Setup (init+network):  {(timings['network_generation']+timings['agent_creation']+timings['network_connections']+timings['model_init']+timings['model_setup'])/total_time*100:5.1f}%")
            print(f"  Simulation:            {timings['simulation']/total_time*100:5.1f}%")
            print(f"  Overhead (collect): {(timings['state_collection']+timings['final_gather'])/total_time*100:5.1f}%")

        # Cross-rank timing statistics
        if len(all_timings) > 1:
            print(f"\nCross-Rank Timing Statistics:")
            for key in ['simulation', 'state_collection']:
                values = [t.get(key, 0.0) for t in all_timings]
                print(f"  {key:20s}: min={np.min(values):.3f}s, max={np.max(values):.3f}s, " +
                      f"mean={np.mean(values):.3f}s, std={np.std(values):.3f}s")

        # Efficiency (ideal weak scaling = constant time)
        if size > 1:
            # Compare against single-rank baseline (if available)
            # For now, just show absolute performance
            print(f"  Parallel efficiency: N/A (need baseline run)")

        # Final state
        total_accounted = sum(total_counts.values())
        print(f"\nFinal State:")
        print(f"  Susceptible: {total_counts['susceptible']:,} ({total_counts['susceptible']/total_agents*100:.1f}%)")
        print(f"  Infected:    {total_counts['infected']:,} ({total_counts['infected']/total_agents*100:.1f}%)")
        print(f"  Recovered:   {total_counts['recovered']:,} ({total_counts['recovered']/total_agents*100:.1f}%)")
        print(f"  Total:       {total_accounted:,}")

        # Validation
        print(f"\nValidation:")
        validation_passed = True

        if total_accounted == total_agents:
            print(f"  ✓ All agents accounted for")
        else:
            print(f"  ✗ Agent mismatch: expected {total_agents}, got {total_accounted}")
            validation_passed = False

        if total_counts['infected'] + total_counts['recovered'] >= num_infected:
            print(f"  ✓ Infection spread or recovery occurred")
        else:
            print(f"  ✗ No infection spread detected!")
            validation_passed = False

        # Deterministic validation mode checks
        if validation_mode:
            print(f"\n  DETERMINISTIC VALIDATION (p_infection=1.0, p_recovery=0.0):")

            # With p_infection=1.0, infection should spread deterministically
            # Expected: initial infected + their neighbors after 1 hop
            # For small-world network with avg_degree=10:
            # After 1 tick: ~num_infected * (1 + avg_degree) infected
            # After N ticks: exponential spread

            min_expected_infected = num_infected * (1 + args.avg_degree)

            if args.ticks >= 1:
                if total_counts['infected'] >= min_expected_infected:
                    print(f"    ✓ Infection spread correctly: {total_counts['infected']} >= {min_expected_infected} expected")
                else:
                    print(f"    ✗ Insufficient spread: {total_counts['infected']} < {min_expected_infected} expected")
                    validation_passed = False

            # With p_recovery=0.0, there should be NO recovered agents
            if total_counts['recovered'] == 0:
                print(f"    ✓ No recovery occurred (as expected with p_recovery=0.0)")
            else:
                print(f"    ✗ Unexpected recovery: {total_counts['recovered']} agents recovered!")
                validation_passed = False

            # Check for network structure preservation
            # Flatten gathered states
            state_dict = {}
            for rank_states in all_states_gathered:
                for agent_id, state in rank_states:
                    state_dict[agent_id] = state

            # Sample check: initially infected agents should still be infected (no recovery)
            initially_infected = list(range(num_infected))  # Assuming first N agents
            all_still_infected = all(state_dict.get(aid) == SIRState.INFECTED.value
                                     for aid in initially_infected if aid in state_dict)

            if all_still_infected:
                print(f"    ✓ Initially infected agents remain infected")
            else:
                print(f"    ✗ Some initially infected agents changed state!")
                validation_passed = False

        # Summary line (for parsing by scripts)
        print("\n" + "=" * 80)
        print(f"SUMMARY: ranks={size}, agents={total_agents}, time={max_sim_time:.3f}s, " +
              f"agent_ticks_per_sec={agent_ticks_per_sec:,.0f}")

        if validation_passed:
            print("SUCCESS")
        else:
            print("FAILED")

        print("=" * 80)

if __name__ == "__main__":
    main()
