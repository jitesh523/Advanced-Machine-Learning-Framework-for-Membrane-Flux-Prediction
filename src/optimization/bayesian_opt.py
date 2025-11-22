"""
Bayesian Optimization for Hyperparameter Tuning

Uses Bayesian Optimization to find optimal hyperparameters for XGBoost model.
"""

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


class BayesianOptimizer:
    """Bayesian optimization for XGBoost hyperparameters."""
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.best_model = None
        self.optimization_history = []
        
    def xgb_evaluate(self, max_depth, learning_rate, n_estimators, 
                     min_child_weight, subsample, colsample_bytree, gamma):
        """
        Objective function for Bayesian Optimization.
        
        Returns negative RMSE (to maximize).
        """
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'min_child_weight': int(min_child_weight),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create model
        model = XGBRegressor(**params)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        
        # Return mean CV R² score
        return cv_scores.mean()
    
    def optimize(self, n_iter=30, init_points=5):
        """
        Run Bayesian Optimization.
        
        Parameters:
        -----------
        n_iter : int
            Number of optimization iterations
        init_points : int
            Number of random exploration points
        """
        print("Starting Bayesian Optimization...")
        print(f"  Iterations: {n_iter}")
        print(f"  Initial random points: {init_points}")
        print("="*80)
        
        # Define parameter bounds
        pbounds = {
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 300),
            'min_child_weight': (1, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 0.5)
        }
        
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=self.xgb_evaluate,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        # Run optimization
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        
        # Extract best parameters
        self.best_params = optimizer.max['params']
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
        self.best_params['min_child_weight'] = int(self.best_params['min_child_weight'])
        self.best_params['random_state'] = 42
        self.best_params['n_jobs'] = -1
        
        print("\n" + "="*80)
        print("Optimization complete!")
        print(f"Best CV R² score: {optimizer.max['target']:.4f}")
        print("\nBest parameters:")
        for param, value in self.best_params.items():
            if param not in ['random_state', 'n_jobs']:
                print(f"  {param}: {value}")
        
        return self.best_params
    
    def train_best_model(self):
        """Train final model with optimized parameters."""
        if self.best_params is None:
            raise ValueError("Run optimize() first!")
        
        print("\n" + "="*80)
        print("Training final model with optimized parameters...")
        
        # Train model
        self.best_model = XGBRegressor(**self.best_params)
        self.best_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        
        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred)
        }
        
        print("\nOptimized Model Performance:")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²:  {metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Test MAE:  {metrics['test_mae']:.4f}")
        print("="*80)
        
        return metrics
    
    def save_model(self, filepath='models/xgboost_optimized.pkl'):
        """Save optimized model."""
        if self.best_model is None:
            raise ValueError("Train model first!")
        
        joblib.dump({
            'model': self.best_model,
            'params': self.best_params
        }, filepath)
        
        print(f"✓ Optimized model saved to {filepath}")


def main():
    """Main function for Bayesian Optimization."""
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed_data.csv')
    
    # Separate features and target
    X = df.drop('Membrane Flux (LMH)', axis=1)
    y = df['Membrane Flux (LMH)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples\n")
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(X_train, y_train, X_test, y_test)
    
    # Run optimization (reduced iterations for speed)
    best_params = optimizer.optimize(n_iter=20, init_points=5)
    
    # Train final model
    metrics = optimizer.train_best_model()
    
    # Save model
    optimizer.save_model()
    
    # Save results
    results_df = pd.DataFrame([{
        'Model': 'XGBoost (Optimized)',
        'Train R²': metrics['train_r2'],
        'Test R²': metrics['test_r2'],
        'Test RMSE': metrics['test_rmse'],
        'Test MAE': metrics['test_mae']
    }])
    
    results_df.to_csv('results/optimized_results.csv', index=False)
    print(f"✓ Results saved to results/optimized_results.csv")
    
    print("\n" + "="*80)
    print("Bayesian Optimization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
