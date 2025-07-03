"""
Test script to demonstrate the dataloader functionality.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import CascadeNetworkDataLoader
import numpy as np


def main():
    """Test the dataloader with the sample data files."""

    # File paths
    data_dir = "../data"
    cascades_file = os.path.join(data_dir, "cascades_filtered_sample.json")
    network_file = os.path.join(data_dir, "network_preprocessed_sample.graphml")
    embeddings_file = os.path.join(data_dir, "source_embeddings_sample.json")

    print("=== Cascade Network DataLoader Test ===\n")

    # Initialize dataloader
    dataloader = CascadeNetworkDataLoader(cascades_file, network_file, embeddings_file)

    # Test 1: Load and examine network
    print("1. Loading network with embeddings...")
    network = dataloader.load_network()
    print(
        f"   ✓ Network loaded: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges"
    )

    # Show sample node with embedding
    sample_node = list(network.nodes())[0]
    node_data = network.nodes[sample_node]
    print(f"   ✓ Sample node '{sample_node}':")
    print(f"     - Embedding dimension: {len(node_data['embedding'])}")
    print(f"     - Quote count: {node_data['quote_count']}")
    print(f"     - Avg quote length: {node_data['avg_quote_length']:.2f}")

    # Test 2: Network statistics
    print("\n2. Network statistics:")
    stats = dataloader.get_network_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")

    # Test 3: Cascade iteration
    print("\n3. Testing cascade iterator:")
    cascade_count = 0
    for cascade_name, cascade_events in dataloader.cascade_iterator():
        cascade_count += 1
        print(f"\n   Cascade {cascade_count}: '{cascade_name}'")
        print(f"   - {len(cascade_events)} events")
        print(f"   - Duration: {cascade_events[0][0]} to {cascade_events[-1][0]}")

        # Show temporal distribution
        timestamps = [event[0] for event in cascade_events]
        durations = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
            for i in range(len(timestamps) - 1)
        ]
        if durations:
            print(f"   - Avg time between events: {np.mean(durations):.2f} hours")

        # Show sample events
        print("   - First 3 events:")
        for i, (timestamp, source) in enumerate(cascade_events[:3]):
            embedding = dataloader.get_node_embedding(source)
            in_network = source in network.nodes()
            print(
                f"     {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M')} - {source[:30]}... "
                f"(in network: {in_network}, embedding: {len(embedding)}D)"
            )

    # Test 4: Specific cascade access
    print("\n4. Testing specific cascade access:")
    cascade_names = list(dataloader.load_cascades().keys())
    if cascade_names:
        test_cascade = cascade_names[0]
        cascade_data = dataloader.get_cascade_by_name(test_cascade)
        print(
            f"   ✓ Retrieved cascade '{test_cascade}' with {len(cascade_data)} events"
        )

        # Analyze cascade network coverage
        cascade_sources = {source for _, source in cascade_data}
        network_sources = set(network.nodes())
        coverage = len(cascade_sources.intersection(network_sources)) / len(
            cascade_sources
        )
        print(
            f"   ✓ Network coverage: {coverage:.1%} of cascade sources are in the network"
        )

    # Test 5: Embedding access
    print("\n5. Testing embedding access:")
    embeddings = dataloader.load_embeddings()
    sample_sources = list(embeddings.keys())[:3]

    for source in sample_sources:
        embedding = dataloader.get_node_embedding(source)
        print(f"   ✓ {source[:30]}... - Embedding shape: {embedding.shape}")
        print(
            f"     Embedding stats: min={embedding.min():.3f}, max={embedding.max():.3f}, "
            f"mean={embedding.mean():.3f}"
        )

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()
