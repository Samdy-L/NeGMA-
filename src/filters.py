def rule_based_filtering(df):
    """
    Stage 1: Rule-based pre-classification.
    
    Rules:
    - Rule 1: internal_density D > 0.7
    - Rule 2: formation_speed V <= 2 (and V != -1)
    - Rule 3: isolation_degree I < 0.3
    
    A community is pre-labeled as "potential fake" (rule_flag=1) 
    if at least 2 out of 3 rules are satisfied.
    
    Args:
        df (pd.DataFrame): Input dataframe with community features.
        
    Returns:
        pd.DataFrame: Dataframe with added 'rule_flag' column.
    """
    print("\n--- Stage 1: Rule-based Pre-classification ---")
    
    # Rule 1: High Internal Density
    r1 = df['internal_density'] > 0.7
    
    # Rule 2: Fast Formation Speed
    # Note: formation_speed == -1 means not yet stable, so it doesn't satisfy "fast formation"
    r2 = (df['formation_speed'] <= 2) & (df['formation_speed'] != -1)
    
    # Rule 3: Low Isolation Degree (High Internal Focus)
    r3 = df['isolation_degree'] < 0.3
    
    # Count satisfied rules
    satisfied_count = r1.astype(int) + r2.astype(int) + r3.astype(int)
    
    # Decision: At least 2 rules satisfied
    df['rule_flag'] = (satisfied_count >= 2).astype(int)
    
    num_potential_fake = df['rule_flag'].sum()
    print(f"Rule-based check: {num_potential_fake} / {len(df)} communities flagged as potential fake.")
    
    return df
