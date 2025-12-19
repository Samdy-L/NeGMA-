from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import pandas as pd
import networkx as nx

def post_hoc_verification(results_df, snapshots, communities_over_time, user_ground_truth):
    """
    Performs post-hoc verification on detected fake communities.
    
    Logic:
    - For each community labeled as Fake (final_pred == 1):
        - Identify top 10% core nodes by internal degree/strength.
        - Check if these core nodes have a history of fake spreading (using user_ground_truth).
        - If repeated fake behavior is observed (> 50% of core nodes are fake), reinforce label.
        - Otherwise, mark as "Uncertain".
        
    Args:
        results_df (pd.DataFrame): Dataframe with 'final_pred', 'community_id', 'snapshot_id'.
        snapshots (list): List of graph snapshots.
        communities_over_time (list): List of community partitions.
        user_ground_truth (dict): User labels (Ground Truth) for verification.
        
    Returns:
        pd.DataFrame: Results dataframe with added 'verification_status' column.
    """
    print("\n--- Post-hoc Verification ---")
    
    df = results_df.copy()
    df['verification_status'] = 'Not Checked'
    
    # Filter for predicted fake communities
    fake_comms_indices = df[df['final_pred'] == 1].index
    
    for idx in fake_comms_indices:
        row = df.loc[idx]
        t = int(row['snapshot_id'])
        comm_id = row['community_id']
        
        # Retrieve community nodes
        # Note: communities_over_time is a list of dicts: [{comm_id: [nodes]}, ...]
        if t < len(communities_over_time) and comm_id in communities_over_time[t]:
            nodes = communities_over_time[t][comm_id]
            G = snapshots[t]
            
            # 1. Identify Top 10% Core Nodes by Internal Degree
            subgraph = G.subgraph(nodes)
            degrees = dict(subgraph.degree(weight='weight'))
            
            # Sort by degree descending
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            
            top_k = max(1, int(len(nodes) * 0.1))
            core_nodes = [n for n, deg in sorted_nodes[:top_k]]
            
            # 2. Check Historical Fake Behavior (using Ground Truth as proxy for history here)
            # In a real system, this would query a historical database.
            fake_core_count = 0
            for node in core_nodes:
                if user_ground_truth.get(node, 0) == 1:
                    fake_core_count += 1
            
            core_fake_ratio = fake_core_count / len(core_nodes)
            
            # 3. Decision
            if core_fake_ratio > 0.5:
                df.at[idx, 'verification_status'] = 'Reinforced Fake'
            else:
                df.at[idx, 'verification_status'] = 'Uncertain'
        else:
            df.at[idx, 'verification_status'] = 'Error: Data Missing'
            
    # Summary
    status_counts = df['verification_status'].value_counts()
    print("Verification Status Counts:")
    print(status_counts)
    
    return df

def format_output(results_df):
    """
    Formats the output dataframe to include only requested columns.
    
    Output columns:
    - community_id
    - rule_flag
    - predicted_probability (final_prob)
    - predicted_label (final_pred)
    - ground_truth_label (target, if available)
    - verification_status (if available)
    """
    cols = ['community_id', 'snapshot_id', 'rule_flag', 'final_prob', 'final_pred']
    
    rename_map = {
        'final_prob': 'predicted_probability',
        'final_pred': 'predicted_label'
    }
    
    if 'target' in results_df.columns:
        cols.append('target')
        rename_map['target'] = 'ground_truth_label'
        
    if 'verification_status' in results_df.columns:
        cols.append('verification_status')
        
    out_df = results_df[cols].rename(columns=rename_map)
    return out_df

def evaluate_results(results_df):
    """
    Evaluates the classification results.
    
    Args:
        results_df (pd.DataFrame): Dataframe containing 'target', 'final_pred', and 'final_prob'.
    """
    if 'target' not in results_df.columns:
        print("No ground truth 'target' found in results. Skipping evaluation.")
        return

    print("\n--- Evaluation Results (Community Level) ---")
    
    y_true = results_df['target']
    y_pred = results_df['final_pred']
    y_prob = results_df['final_prob']
    
    # Check if we have enough classes to evaluate
    if len(y_true.unique()) < 2:
        print("Only one class present in target. Cannot compute ROC AUC.")
        print(classification_report(y_true, y_pred))
        return

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = classification_report(y_true, y_pred, output_dict=True)['1']['precision']
    rec = classification_report(y_true, y_pred, output_dict=True)['1']['recall']
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC-ROC:   {auc:.4f}")
    except ValueError as e:
        print(f"Could not compute AUC-ROC: {e}")
        
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
