import networkx as nx
import pandas as pd
import numpy as np

def calculate_jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def extract_community_features(G, community_nodes, community_id, 
                             prev_nodes=None, first_seen_t=0, current_t=0, 
                             size_history=None,
                             max_edge_weight=1.0):
    """
    Extracts features for a single community at a specific snapshot.
    
    Features:
    1. community_size
    2. internal_weight_sum
    3. internal_density
    4. formation_speed (Snapshots to reach stability)
    5. isolation_degree
    6. member_stability
    7. average_internal_degree
    8. internal_edge_weight_variance
    """
    nodes_list = list(community_nodes)
    k_c = len(nodes_list)
    
    # 1. Community Size
    feature_size = k_c
    
    if k_c == 0:
        return {
            'community_size': 0, 'internal_weight_sum': 0, 'internal_density': 0,
            'formation_speed': 0, 'isolation_degree': 0, 'member_stability': 0,
            'avg_internal_degree': 0, 'internal_edge_weight_var': 0
        }

    # Subgraph for internal calculations
    subgraph = G.subgraph(nodes_list)
    
    # 2. Internal Weight Sum & 8. Variance
    weights = []
    for u, v, data in subgraph.edges(data=True):
        w = data.get('weight', 1.0)
        weights.append(w)
    
    w_in_c = sum(weights)
    internal_edge_weight_var = np.var(weights) if weights else 0.0
    
    # 3. Internal Density
    # D = w_in_c / ((k_c * (k_c - 1) / 2) * max_edge_weight)
    possible_edges = k_c * (k_c - 1) / 2
    if possible_edges > 0:
        internal_density = w_in_c / (possible_edges * max_edge_weight)
    else:
        internal_density = 0.0
        
    # 4. Formation Speed (Snapshots to reach stable size)
    # Definition: Time taken until size variation < 10% for 2 consecutive snapshots
    formation_speed = -1 # -1 indicates not yet stable
    if size_history and len(size_history) >= 2:
        # Check history for stability point
        # We look for the first time t where |size(t) - size(t-1)| / size(t-1) < 0.1
        # Since we only have the history list, we iterate to find the first stability event
        
        # Current size is k_c, append to history temporarily for calculation
        full_history = size_history + [k_c]
        
        stable_t = -1
        for i in range(1, len(full_history)):
            prev_size = full_history[i-1]
            curr_size = full_history[i]
            if prev_size > 0:
                change = abs(curr_size - prev_size) / prev_size
                if change < 0.1:
                    # Found stability point relative to start
                    # i is the index in history. 
                    # If history starts at first_seen_t, then time is i.
                    stable_t = i
                    break
        
        if stable_t != -1:
            formation_speed = stable_t
        else:
            # If never stable, use current age as proxy or keep -1
            formation_speed = current_t - first_seen_t
    else:
        formation_speed = 0 # Too young to tell
    
    # 5. Isolation Degree
    # Ratio of external connections to total connections
    ext_edges_count = 0
    nodes_set = set(nodes_list)
    for n in nodes_list:
        for nbr in G.neighbors(n):
            if nbr not in nodes_set:
                ext_edges_count += 1
                
    int_edges_count = len(weights)
    total_edges = int_edges_count + ext_edges_count
    
    isolation_degree = ext_edges_count / total_edges if total_edges > 0 else 0.0
    
    # 6. Member Stability
    if prev_nodes is not None:
        member_stability = calculate_jaccard(set(nodes_list), set(prev_nodes))
    else:
        member_stability = 0.0 # New community or first snapshot
        
    # 7. Average Internal Degree
    # avg_deg = 2 * internal_edges / k_c
    avg_internal_degree = (2 * int_edges_count) / k_c if k_c > 0 else 0.0
    
    return {
        'community_size': feature_size,
        'internal_weight_sum': w_in_c,
        'internal_density': internal_density,
        'formation_speed': formation_speed,
        'isolation_degree': isolation_degree,
        'member_stability': member_stability,
        'avg_internal_degree': avg_internal_degree,
        'internal_edge_weight_var': internal_edge_weight_var
    }

def prepare_dataset(snapshots, communities_over_time, known_labels, user_ground_truth=None):
    """
    Prepares the dataset by iterating over snapshots and communities.
    
    Args:
        snapshots (list): List of networkx graphs.
        communities_over_time (list): List of dicts {comm_id: [nodes]}.
        known_labels (dict): Partial supervision labels.
        user_ground_truth (dict, optional): Full ground truth for generating targets.
        
    Returns:
        pd.DataFrame: Dataset with features and optional targets.
    """
    data = []
    
    # Track history: {comm_id: {'first_seen': t, 'prev_nodes': set(), 'size_history': []}}
    comm_history = {}
    
    for t, (G, comms) in enumerate(zip(snapshots, communities_over_time)):
        for comm_id, nodes in comms.items():
            nodes_set = set(nodes)
            curr_size = len(nodes)
            
            # History tracking
            if comm_id not in comm_history:
                comm_history[comm_id] = {
                    'first_seen': t, 
                    'prev_nodes': None,
                    'size_history': []
                }
            
            history = comm_history[comm_id]
            
            # Extract features
            features = extract_community_features(
                G=G,
                community_nodes=nodes,
                community_id=comm_id,
                prev_nodes=history['prev_nodes'],
                first_seen_t=history['first_seen'],
                current_t=t,
                size_history=history['size_history']
            )
            
            # Update history
            comm_history[comm_id]['prev_nodes'] = nodes_set
            comm_history[comm_id]['size_history'].append(curr_size)
            
            # Meta info
            features['snapshot_id'] = t
            features['community_id'] = comm_id
            
            # Generate Ground Truth (Target)
            if user_ground_truth:
                real_labels = [user_ground_truth.get(n, 0) for n in nodes]
                if len(real_labels) > 0:
                    real_fake_ratio = sum(real_labels) / len(real_labels)
                    features['target'] = 1 if real_fake_ratio > 0.3 else 0
                else:
                    features['target'] = 0
            
            # Auxiliary features for Rule-Based Filtering
            labels_in_comm = [known_labels[n] for n in nodes if n in known_labels]
            num_known = len(labels_in_comm)
            known_fake_ratio = sum(labels_in_comm) / num_known if num_known > 0 else 0.0
            labeled_coverage = num_known / len(nodes) if len(nodes) > 0 else 0.0
            
            features['known_fake_ratio'] = known_fake_ratio
            features['labeled_coverage'] = labeled_coverage
            
            data.append(features)
            
    return pd.DataFrame(data)
