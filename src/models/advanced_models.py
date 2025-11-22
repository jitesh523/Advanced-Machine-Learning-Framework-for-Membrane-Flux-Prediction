"""
Advanced Models for Membrane Flux Prediction

Implements:
- Random Forest Regressor
- XGBoost Regressor
- Neural Network (MLPRegressor)
- Support Vector Regression (SVR)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


class AdvancedModels:
    """Advanced regression models for flux prediction."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'SVR': SVR(
                kernel='rbf',
                C=10.0,
                epsilon=0.1
            )
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_model(self, X_train, y_train, X_test, y_test, model_name):
        """
        Train a single model and evaluate it.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        model_name : str, name of the model
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        model = self.models[model_name]
        
        print(f"  Training {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'model': model
        }
        
        # Cross-validation score (skip for SVR as it's slow)
        if model_name != 'SVR':
            print(f"  Running cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                        scoring='r2', n_jobs=-1)
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        else:
            metrics['cv_r2_mean'] = np.nan
            metrics['cv_r2_std'] = np.nan
        
        return metrics
    
    def train_all(self, X_train, y_train, X_test, y_test):
        """
        Train all advanced models.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        
        Returns:
        --------
        dict : Results for all models
        """
        print("Training advanced models...")
        print("="*80)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            metrics = self.train_model(X_train, y_train, X_test, y_test, model_name)
            self.results[model_name] = metrics
            
            print(f"  ✓ Train R²: {metrics['train_r2']:.4f}")
            print(f"  ✓ Test R²:  {metrics['test_r2']:.4f}")
            print(f"  ✓ Test RMSE: {metrics['test_rmse']:.4f}")
            if not np.isnan(metrics['cv_r2_mean']):
                print(f"  ✓ CV R² (5-fold): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        # Find best model based on test R²
        best_name = max(self.results.keys(), 
                       key=lambda k: self.results[k]['test_r2'])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        print("\n" + "="*80)
        print(f"✓ Best advanced model: {best_name}")
        print(f"  Test R²: {self.results[best_name]['test_r2']:.4f}")
        print("="*80)
        
        return self.results
    
    def get_summary_df(self):
        """Get summary DataFrame of all model results."""
        summary_data = []
        
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'CV R² (mean)': metrics['cv_r2_mean'],
                'CV R² (std)': metrics['cv_r2_std']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Test R²', ascending=False)
        return df
    
    def save_best_model(self, filepath='models/advanced_best.pkl'):
        """Save the best performing advanced model."""
        if self.best_model is None:
            raise ValueError("No models trained yet. Call train_all() first.")
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]
        }, filepath)
        
        print(f"✓ Best model ({self.best_model_name}) saved to {filepath}")


def main():
    """Main function to train advanced models."""
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed_data.csv')
    print(f"Data shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('Membrane Flux (LMH)', axis=1)
    y = df['Membrane Flux (LMH)']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(y)}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples\n")
    
    # Train advanced models
    advanced = AdvancedModels()
    results = advanced.train_all(X_train, y_train, X_test, y_test)
    
    # Display summary
    print("\n" + "="*80)
    print("ADVANCED MODELS SUMMARY")
    print("="*80)
    summary_df = advanced.get_summary_df()
    print(summary_df.to_string(index=False))
    
    # Load baseline results for comparison
    try:
        baseline_df = pd.read_csv('results/baseline_results.csv')
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE MODELS")
        print("="*80)
        print(baseline_df.to_string(index=False))
    except:
        pass
    
    # Save results
    summary_df.to_csv('results/advanced_results.csv', index=False)
    print(f"\n✓ Results saved to results/advanced_results.csv")
    
    # Save best model
    advanced.save_best_model()
    
    print("\n" + "="*80)
    print("Advanced model training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
