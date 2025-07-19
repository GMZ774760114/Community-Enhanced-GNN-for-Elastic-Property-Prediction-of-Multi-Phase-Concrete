import pickle
import torch
from torch_geometric.data import Data
import networkx as nx
import community as community_louvain
from collections import defaultdict
from networkx.algorithms.centrality import betweenness_centrality
import random

# Load the original graph data
file_path = 'your input file path (graph without community structure)'
with open(file_path, 'rb') as f:
    data_list = pickle.load(f)

# Create a new list for updated graphs
updated_data_list = []

# Iterate through each subgraph and compute community structure
for idx, data in enumerate(data_list):
    print(f"Processing Graph {idx + 1}...")

    # Ensure node indices in edge_index are valid
    valid_edges = data.edge_index[:, (data.edge_index < data.x.shape[0]).all(dim=0)]
    data.edge_index = valid_edges

    # Convert to NetworkX graph
    G = nx.Graph()
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Compute community partition
    partition = community_louvain.best_partition(G)

    # Initialize community labels and assign values
    community_labels = torch.full((data.x.shape[0], 1), fill_value=-1, dtype=torch.float)  # Default community is -1
    for node, community in partition.items():
        community_labels[node, 0] = community

    # Check dimension consistency
    if data.x.shape[0] != community_labels.shape[0]:
        raise ValueError(f"Mismatch: data.x has {data.x.shape[0]} nodes, but community_labels has {community_labels.shape[0]}.")

    # Append community labels to node features
    data.x = torch.cat([data.x, community_labels], dim=1)

    # === Methods to reconstruct the graph based on communities ===
    # 1. Enhance connections between communities
    centrality = betweenness_centrality(G)  # Compute betweenness centrality for nodes
    communities = list(set(partition.values()))
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            nodes_in_comm1 = [node for node, comm in partition.items() if comm == communities[i]]
            nodes_in_comm2 = [node for node, comm in partition.items() if comm == communities[j]]
            if len(nodes_in_comm1) > 0 and len(nodes_in_comm2) > 0:
                # Select nodes with highest centrality in each community
                node1 = max(nodes_in_comm1, key=lambda x: centrality[x])
                node2 = max(nodes_in_comm2, key=lambda x: centrality[x])
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=1.0)  # Add edge

    # 2. Optimize connections within communities
    for community in set(partition.values()):
        nodes_in_community = [node for node, comm in partition.items() if comm == community]
        for i in range(len(nodes_in_community)):
            for j in range(i + 1, len(nodes_in_community)):
                if not G.has_edge(nodes_in_community[i], nodes_in_community[j]):
                    G.add_edge(nodes_in_community[i], nodes_in_community[j], weight=2.0)  # Add edge with high weight

    # 3. Merge small communities
    community_sizes = defaultdict(int)
    for node, community in partition.items():
        community_sizes[community] += 1
    min_community_size = 5  # Define minimum community size
    small_communities = [comm for comm, size in community_sizes.items() if size < min_community_size]
    for comm in small_communities:
        nodes_in_small_comm = [node for node, c in partition.items() if c == comm]
        if len(nodes_in_small_comm) > 0:
            # Check for other communities to merge with
            other_communities = set(partition.values()) - {comm}
            if len(other_communities) > 0:
                # Find the larger community with the most connections to the small community
                target_community = max(
                    other_communities,
                    key=lambda x: sum(1 for node in nodes_in_small_comm if G.has_edge(node, [n for n, c in partition.items() if c == x][0]))
                )
                for node in nodes_in_small_comm:
                    partition[node] = target_community  # Merge nodes from small community to target community
            else:
                # If no other communities exist, skip merging
                print(f"Warning: No other communities to merge for community {comm}. Skipping merge.")

    # Update edge data for the graph
    edges = list(G.edges())
    data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Add updated graph to the list
    updated_data_list.append(data)

# Save the processed dataset
output_file = 'your_output_file_path.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(updated_data_list, f)

print(f"Updated and improved graph dataset saved to {output_file}.")