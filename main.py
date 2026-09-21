import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import utils, features, filters, model, fusion, evaluation

def main():
    print("============================================================")
    print("Experiment 4: Community-level Fake Information Detection")
    print("============================================================")
    
    # 1. Data Generation / Loading
    # In a real experiment, load your G(t) and C(t) here.
    # We use mock data for demonstration.
    snapshots, communities, known_labels, ground_truth = utils.generate_mock_data(
        num_snapshots=10, num_nodes=200
    )
    
    # 2. Feature Extraction
    print("\n[Step 1] Extracting Features...")
    full_df = features.prepare_dataset(snapshots, communities, known_labels, ground_truth)
    print(f"Total communities extracted: {len(full_df)}")
    
    # 3. Train/Test Split
    # We split by community_id or randomly. Random split is used here.
    print("\n[Step 2] Splitting Data (Train/Test)...")
    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size:  {len(test_df)}")
    
    # 4. Stage 1: Rule-based Pre-classification
    # Apply to both Train and Test
    print("\n[Step 3] Stage 1: Rule-based Pre-classification...")
    train_df = filters.rule_based_filtering(train_df)
    test_df = filters.rule_based_filtering(test_df)
    
    # 5. Stage 2: Logistic Regression Training
    # We train on the training set. 
    # Note: We train on ALL training data to let the model learn the general boundary,
    # even though rules will override some decisions later.
    print("\n[Step 4] Stage 2: Model Training...")
    classifier = model.CommunityClassifier()
    classifier.train(train_df, train_df['target'])
    
    # Predict on Test Set
    print("\n[Step 5] Model Prediction on Test Set...")
    test_df = classifier.predict(test_df)
    
    # 6. Fusion
    # Combine Rule and Model results
    print("\n[Step 6] Fusing Results...")
    final_test_df = fusion.apply_fusion_logic(test_df)
    
    # 7. Post-hoc Verification
    # Verify detected fake communities
    print("\n[Step 7] Post-hoc Verification...")
    final_test_df = evaluation.post_hoc_verification(
        final_test_df, snapshots, communities, ground_truth
    )
    
    # 8. Output and Evaluation
    print("\n[Step 8] Final Evaluation...")
    output_df = evaluation.format_output(final_test_df)
    evaluation.evaluate_results(final_test_df)
    
    print("\n--- Sample Output (Top 5) ---")
    print(output_df.head())
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'experiment_results.csv')
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save the full (community, snapshot) detail. visualization.py draws the
    # evolution trajectories from this file.
    evolution_path = os.path.join(os.path.dirname(__file__), 'data', 'community_evolution.csv')
    full_df.to_csv(evolution_path, index=False)
    print(f"Community detail saved to: {evolution_path}")

if __name__ == "__main__":
    main()
