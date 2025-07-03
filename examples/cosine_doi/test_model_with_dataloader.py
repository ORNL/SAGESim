#!/usr/bin/env python3
"""
Test script for the updated cosine_doi model with CascadeNetworkDataLoader
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cosine_doi_model import SIRModel
from cosine_doi_breed import CosineDOIBreed
from dataloader import CascadeNetworkDataLoader
from state import SIRState


def test_model_with_dataloader():
    """Test the model creation with dataloader"""

    # File paths
    data_dir = "../data"
    cascades_file = os.path.join(data_dir, "cascades_filtered_sample.json")
    network_file = os.path.join(data_dir, "network_preprocessed_sample.graphml")
    embeddings_file = os.path.join(data_dir, "source_embeddings_sample.json")

    print("=== Testing CosineDOI Model with CascadeNetworkDataLoader ===\n")

    # Test 1: Initialize dataloader
    print("1. Initializing CascadeNetworkDataLoader...")
    dataloader = CascadeNetworkDataLoader(cascades_file, network_file, embeddings_file)

    # Test 2: Load network
    print("2. Loading network with embeddings...")
    network = dataloader.load_network()
    print(
        f"   ✓ Loaded {network.number_of_nodes()} nodes and {network.number_of_edges()} edges"
    )

    # Test 3: Check node embeddings
    print("3. Checking node embeddings...")
    sample_node = list(network.nodes())[0]
    node_data = network.nodes[sample_node]
    embedding = node_data.get("embedding", [])
    print(
        f"   ✓ Sample node '{sample_node}' has embedding of dimension {len(embedding)}"
    )

    # Test 4: Test breed creation
    print("4. Testing CosineDOIBreed...")
    breed = CosineDOIBreed()
    print(f"   ✓ Breed created with name: {breed.get_name()}")

    # Test 5: Test model creation
    print("5. Testing SIRModel...")
    model = SIRModel()
    print("   ✓ Model created successfully")

    # Test 6: Test agent creation with embedding
    print("6. Testing agent creation with embeddings...")
    embedding_list = (
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )
    preventative_measures = [0.5] * 10  # Simplified for testing

    agent_id = model.create_agent(
        state=SIRState.SUSCEPTIBLE.value,
        preventative_measures=preventative_measures,
        embedding=embedding_list,
    )
    print(f"   ✓ Created agent {agent_id} with embedding")

    # Test 7: Check agent properties
    print("7. Checking agent properties...")
    agent_state = model.get_agent_property_value(agent_id, "state")
    agent_embedding = model.get_agent_property_value(agent_id, "embedding")
    print(f"   ✓ Agent state: {agent_state}")
    print(
        f"   ✓ Agent embedding dimension: {len(agent_embedding) if agent_embedding else 0}"
    )

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_model_with_dataloader()
