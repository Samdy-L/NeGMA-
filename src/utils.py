import networkx as nx
import random
import pandas as pd

def generate_mock_data(num_snapshots=10, num_nodes=200, seed=42):
    """
    Generates mock dynamic graphs, community partitions, and user labels.

    Args:
        seed (int | None): Fixes the RNG so a run reproduces exactly.
            Pass None to draw a fresh random network each time.

    Returns:
        snapshots (list): List of networkx graphs.
        communities_over_time (list): List of dicts {comm_id: [nodes]}.
        known_labels (dict): Partial supervision {node_id: label}.
        user_ground_truth (dict): Full ground truth {node_id: label}.
    """
    print("Generating mock data...")

    if seed is not None:
        random.seed(seed)

    # 1. User Labels (Ground Truth)
    # 0: Normal, 1: Fake
    # Assume 30% users are fake spreaders to ensure positive samples
    user_ground_truth = {i: 1 if random.random() < 0.3 else 0 for i in range(num_nodes)}
    
    # Partial Supervision: Mask some labels (e.g., we only know 40% of labels)
    known_labels = {}
    for node, label in user_ground_truth.items():
        if random.random() < 0.4:
            known_labels[node] = label
            
    # 2. Dynamic Graphs and Communities
    snapshots = []
    communities_over_time = []
    
    for t in range(num_snapshots):
        # Create a random graph
        G = nx.erdos_renyi_graph(n=num_nodes, p=0.05)
        snapshots.append(G)
        
        # Mock Communities (Partition of nodes)
        # Randomly assign nodes to 10 communities
        num_comms = 10
        partition = {}
        for i in range(num_nodes):
            partition[i] = random.randint(0, num_comms - 1)
        
        # Group by community id
        comms = {}
        for node, comm_id in partition.items():
            if comm_id not in comms:
                comms[comm_id] = []
            comms[comm_id].append(node)
        
        communities_over_time.append(comms)
        
    return snapshots, communities_over_time, known_labels, user_ground_truth

def load_data():
    """
    Loads graph snapshots and community partitions.
    To be implemented.
    """
    pass
