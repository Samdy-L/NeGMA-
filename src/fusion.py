import pandas as pd
import numpy as np

def apply_fusion_logic(df):
    """
    Combines Stage 1 (Rule-based) and Stage 2 (Logistic Regression) results.
    
    Decision Logic:
    - If rule_flag == 1 (Potential Fake by rules), classify as Fake (1).
    - Otherwise, use the Logistic Regression prediction (lr_label).
    
    Rule-based decision has higher priority.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'rule_flag', 'lr_label', and 'lr_prob'.
        
    Returns:
        pd.DataFrame: Dataframe with final 'final_pred' and 'final_prob'.
    """
    print("\n--- Fusion Stage: Combining Rules and Model ---")
    
    result_df = df.copy()
    
    # Ensure required columns exist
    if 'rule_flag' not in result_df.columns:
        raise ValueError("Missing 'rule_flag' column from Stage 1.")
    if 'lr_label' not in result_df.columns or 'lr_prob' not in result_df.columns:
        raise ValueError("Missing 'lr_label' or 'lr_prob' columns from Stage 2.")
        
    # Apply Logic
    # Final Label: 1 if rule_flag is 1, else lr_label
    result_df['final_pred'] = np.where(result_df['rule_flag'] == 1, 1, result_df['lr_label'])
    
    # Final Probability:
    # If rule_flag is 1, we can assign a high probability (e.g., max(0.9, lr_prob))
    # or just keep the lr_prob but force the label.
    # Here, to reflect the "high priority" of the rule, we boost the probability if rule says fake.
    # Let's set prob = 1.0 if rule_flag == 1 for simplicity in this logic, 
    # or we can just use the lr_prob. 
    # Given the prompt "classify directly", let's set prob to 1.0 to be consistent with the hard decision.
    result_df['final_prob'] = np.where(result_df['rule_flag'] == 1, 1.0, result_df['lr_prob'])
    
    # Statistics
    total = len(result_df)
    rule_fakes = result_df['rule_flag'].sum()
    model_fakes = result_df[(result_df['rule_flag'] == 0) & (result_df['lr_label'] == 1)].shape[0]
    total_fakes = result_df['final_pred'].sum()
    
    print(f"Total Communities: {total}")
    print(f"Detected by Rules: {rule_fakes}")
    print(f"Detected by Model (only): {model_fakes}")
    print(f"Total Detected Fake: {total_fakes}")
    
    return result_df
