from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

class CommunityClassifier:
    def __init__(self):
        # Use LogisticRegressionCV for built-in 5-fold cross-validation
        # to tune the regularization parameter C.
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(cv=5, class_weight='balanced', random_state=42, max_iter=1000)
        )
        self.features = [
            'community_size', 
            'internal_weight_sum', 
            'internal_density', 
            'formation_speed', 
            'isolation_degree', 
            'member_stability', 
            'avg_internal_degree', 
            'internal_edge_weight_var'
        ]
        self.is_trained = False
        
    def train(self, X, y):
        """
        Trains the logistic regression model with cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        print("\n--- Stage 2: Training Logistic Regression ---")
        
        # Check for class balance
        if len(y.unique()) < 2:
            print("Warning: Training data contains only one class. Model cannot be trained.")
            self.is_trained = False
            return

        # Fit the pipeline (StandardScaler + LogisticRegressionCV)
        self.model.fit(X[self.features], y)
        self.is_trained = True
        
        # Extract the best C and coefficients
        clf = self.model.named_steps['logisticregressioncv']
        print("Model trained.")
        print(f"Best C: {clf.C_[0]}")
        print(f"Coefficients: {dict(zip(self.features, clf.coef_[0]))}")
        
    def predict(self, df):
        """
        Predicts labels and probabilities.
        
        Args:
            df (pd.DataFrame): Dataframe with features.
            
        Returns:
            pd.DataFrame: Dataframe with added 'lr_prob' and 'lr_label' columns.
        """
        if len(df) == 0:
            return df
            
        result_df = df.copy()
        
        if not self.is_trained:
            print("Warning: Model not trained. Returning default predictions (0).")
            result_df['lr_prob'] = 0.0
            result_df['lr_label'] = 0
            return result_df

        X = df[self.features]
        
        # Predict probability and label
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        result_df['lr_prob'] = probs
        result_df['lr_label'] = preds
        
        return result_df
