"""
Dataloader script for processing cascade, network, and embedding data.

This module provides functions to:
1. Load and regenerate the network in an efficient format
2. Provide an iterator for cascades
3. Combine all data sources for easy access
"""

import json
import networkx as nx
from typing import Dict, List, Tuple, Iterator, Any
from datetime import datetime
import numpy as np


class CascadeNetworkDataLoader:
    """
    DataLoader for cascade propagation data combining network structure,
    node embeddings, and cascade sequences.
    """

    def __init__(self, cascades_file: str, network_file: str, embeddings_file: str):
        """
        Initialize the dataloader with paths to the three data files.

        Args:
            cascades_file: Path to cascades JSON file
            network_file: Path to network GraphML file
            embeddings_file: Path to source embeddings JSON file
        """
        self.cascades_file = cascades_file
        self.network_file = network_file
        self.embeddings_file = embeddings_file

        self._network = None
        self._cascades = None
        self._embeddings = None

    def load_network(self) -> nx.Graph:
        """
        Load the network from GraphML file and add embedding attributes to nodes.

        Returns:
            NetworkX Graph with nodes having embedding attributes
        """
        if self._network is None:
            # Load network structure
            self._network = nx.read_graphml(self.network_file)

            # Load embeddings
            embeddings = self.load_embeddings()

            # Add embedding attributes to nodes
            for node_id in self._network.nodes():
                if node_id in embeddings:
                    embedding_data = embeddings[node_id]
                    self._network.nodes[node_id]["embedding"] = np.array(
                        embedding_data["embedding"]
                    )
                    self._network.nodes[node_id]["quote_count"] = embedding_data.get(
                        "quote_count", 0
                    )
                    self._network.nodes[node_id]["total_text_length"] = (
                        embedding_data.get("total_text_length", 0)
                    )
                    self._network.nodes[node_id]["avg_quote_length"] = (
                        embedding_data.get("avg_quote_length", 0.0)
                    )
                else:
                    # Handle missing embeddings with zero vector
                    # Assuming embedding dimension from first available embedding
                    if embeddings:
                        first_embedding = next(iter(embeddings.values()))["embedding"]
                        embedding_dim = len(first_embedding)
                        self._network.nodes[node_id]["embedding"] = np.zeros(
                            embedding_dim
                        )
                    else:
                        self._network.nodes[node_id]["embedding"] = np.array([])
                    self._network.nodes[node_id]["quote_count"] = 0
                    self._network.nodes[node_id]["total_text_length"] = 0
                    self._network.nodes[node_id]["avg_quote_length"] = 0.0

        return self._network

    def load_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """
        Load source embeddings from JSON file.

        Returns:
            Dictionary mapping source names to embedding data
        """
        if self._embeddings is None:
            with open(self.embeddings_file, "r") as f:
                self._embeddings = json.load(f)
        return self._embeddings

    def load_cascades(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load cascades from JSON file.

        Returns:
            Dictionary mapping cascade names to lists of (timestamp, source) tuples
        """
        if self._cascades is None:
            with open(self.cascades_file, "r") as f:
                self._cascades = json.load(f)
        return self._cascades

    def cascade_iterator(self) -> Iterator[Tuple[str, List[Tuple[datetime, str, int]]]]:
        """
        Provide an iterator over cascades with parsed timestamps and ticks.

        Yields:
            Tuple of (cascade_name, list of (datetime, source, tick) tuples)
            where tick is the number of minutes since the first event
        """
        cascades = self.load_cascades()

        for cascade_name, cascade_data in cascades.items():
            # Parse timestamps and create structured cascade data
            parsed_cascade = []
            for timestamp_str, source in cascade_data:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    parsed_cascade.append((timestamp, source))
                except ValueError as e:
                    print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
                    continue

            # Sort by timestamp to ensure chronological order
            parsed_cascade.sort(key=lambda x: x[0])
            
            # Calculate ticks (minutes since first event)
            if parsed_cascade:
                first_timestamp = parsed_cascade[0][0]
                cascade_with_ticks = []
                for timestamp, source in parsed_cascade:
                    tick = int((timestamp - first_timestamp).total_seconds() / 60)
                    cascade_with_ticks.append((timestamp, source, tick))
                yield cascade_name, cascade_with_ticks
            else:
                yield cascade_name, []

    def get_cascade_by_name(self, cascade_name: str) -> List[Tuple[datetime, str, int]]:
        """
        Get a specific cascade by name with parsed timestamps and ticks.

        Args:
            cascade_name: Name of the cascade to retrieve

        Returns:
            List of (datetime, source, tick) tuples sorted chronologically
            where tick is the number of minutes since the first event
        """
        cascades = self.load_cascades()
        if cascade_name not in cascades:
            raise KeyError(f"Cascade '{cascade_name}' not found")

        cascade_data = cascades[cascade_name]
        parsed_cascade = []

        for timestamp_str, source in cascade_data:
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                parsed_cascade.append((timestamp, source))
            except ValueError as e:
                print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
                continue

        # Sort by timestamp to ensure chronological order
        parsed_cascade.sort(key=lambda x: x[0])
        
        # Calculate ticks (minutes since first event)
        if parsed_cascade:
            first_timestamp = parsed_cascade[0][0]
            cascade_with_ticks = []
            for timestamp, source in parsed_cascade:
                tick = int((timestamp - first_timestamp).total_seconds() / 60)
                cascade_with_ticks.append((timestamp, source, tick))
            return cascade_with_ticks
        else:
            return []

    def get_node_embedding(self, source: str) -> np.ndarray:
        """
        Get embedding for a specific source node.

        Args:
            source: Source name

        Returns:
            NumPy array containing the embedding vector
        """
        embeddings = self.load_embeddings()
        if source in embeddings:
            return np.array(embeddings[source]["embedding"])
        else:
            # Return zero vector if embedding not found
            if embeddings:
                first_embedding = next(iter(embeddings.values()))["embedding"]
                embedding_dim = len(first_embedding)
                return np.zeros(embedding_dim)
            return np.array([])

    def get_network_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the network.

        Returns:
            Dictionary with network statistics
        """
        network = self.load_network()
        embeddings = self.load_embeddings()
        cascades = self.load_cascades()

        # Get embedding dimension
        embedding_dim = 0
        if embeddings:
            first_embedding = next(iter(embeddings.values()))["embedding"]
            embedding_dim = len(first_embedding)

        stats = {
            "num_nodes": network.number_of_nodes(),
            "num_edges": network.number_of_edges(),
            "num_cascades": len(cascades),
            "embedding_dimension": embedding_dim,
            "nodes_with_embeddings": len(
                [n for n in network.nodes() if n in embeddings]
            ),
            "cascade_names": list(cascades.keys()),
            "avg_cascade_length": np.mean(
                [len(cascade) for cascade in cascades.values()]
            ),
            "is_connected": nx.is_connected(network),
            "avg_degree": np.mean([d for n, d in network.degree()]),
        }

        return stats


def load_data(
    cascades_file: str, network_file: str, embeddings_file: str
) -> CascadeNetworkDataLoader:
    """
    Convenience function to create a dataloader instance.

    Args:
        cascades_file: Path to cascades JSON file
        network_file: Path to network GraphML file
        embeddings_file: Path to source embeddings JSON file

    Returns:
        CascadeNetworkDataLoader instance
    """
    return CascadeNetworkDataLoader(cascades_file, network_file, embeddings_file)


# Example usage and testing functions
if __name__ == "__main__":
    # Example file paths (adjust as needed)
    cascades_file = "../data/cascades_filtered_sample.json"
    network_file = "../data/network_preprocessed_sample.graphml"
    embeddings_file = "../data/source_embeddings_sample.json"

    # Create dataloader
    dataloader = load_data(cascades_file, network_file, embeddings_file)

    # Load network with embeddings
    print("Loading network...")
    network = dataloader.load_network()
    print(
        f"Network loaded with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges"
    )

    # Show network statistics
    print("\nNetwork Statistics:")
    stats = dataloader.get_network_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Iterate through cascades
    print("\nCascades:")
    for cascade_name, cascade_events in dataloader.cascade_iterator():
        print(f"\nCascade: {cascade_name}")
        print(f"  Length: {len(cascade_events)} events")
        if cascade_events:
            print(f"  First event: {cascade_events[0][0]} at {cascade_events[0][1]} (tick {cascade_events[0][2]})")
            print(f"  Last event: {cascade_events[-1][0]} at {cascade_events[-1][1]} (tick {cascade_events[-1][2]})")

            # Show first few events
            print("  First 5 events:")
            for i, (timestamp, source, tick) in enumerate(cascade_events[:5]):
                embedding = dataloader.get_node_embedding(source)
                print(
                    f"    {i+1}. {timestamp} - {source} (tick: {tick} min, embedding dim: {len(embedding)})"
                )
