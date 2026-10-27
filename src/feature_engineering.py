"""
Feature Engineering Pipeline for Membrane Flux Prediction

This module handles:
1. Derived feature creation (physics-based)
2. Categorical variable encoding
3. Feature scaling and normalization
4. Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib


class FeatureEngineer:
    """Feature engineering pipeline for FO membrane flux prediction."""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.onehot_columns = {}
        self.feature_names = None
        self.categorical_cols = []
        self.numerical_cols = []
        
    def create_derived_features(self, df):
        """
        Create physics-based derived features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame : DataFrame with added derived features
        """
        df = df.copy()
        
        # 1. Osmotic pressure difference (primary driving force)
        df['Osmotic_Pressure_Diff'] = (
            df['DS Osmotic pressure (atm)'] - df['FS Osmotic pressure (atm)']
        )
        
        # 2. Temperature difference
        df['Temperature_Diff'] = (
            df['DS Temperature (°C)'] - df['FS Temperature (°C)']
        )
        
        # 3. Average temperature (affects viscosity and diffusivity)
        df['Temperature_Avg'] = (
            df['DS Temperature (°C)'] + df['FS Temperature (°C)']
        ) / 2
        
        # 4. Velocity ratio (hydrodynamic balance)
        df['Velocity_Ratio'] = (
            df['DS Velocity (cm/s)'] / (df['FS Velocity (cm/s)'] + 1e-6)
        )
        
        # 5. Concentration ratio
        df['Concentration_Ratio'] = (
            df['DS Molarity (M)'] / (df['FS Molarity (M)'] + 1e-6)
        )
        
        # 6. Molecular weight effect (diffusion coefficient proxy)
        df['MW_Normalized'] = df['DS Molecular weight'] / 100  # Normalize to reasonable scale
        
        # 7. Osmotic pressure per molarity (osmotic coefficient indicator)
        df['Osmotic_Per_Molarity'] = (
            df['DS Osmotic pressure (atm)'] / (df['DS Molarity (M)'] + 1e-6)
        )
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit encoders (True for training, False for test)
            
        Returns:
        --------
        pd.DataFrame : DataFrame with encoded features
        """
        df = df.copy()
        
        # Define categorical columns for one-hot encoding
        onehot_cols = [
            'Membrane AL Type',
            'Membrane Direction', 
            'Flow Orientation (counter/ cocurrent)'
        ]
        
        # Keep the training columns and reference category for every batch.
        if fit:
            self.onehot_columns = {}
        for col in onehot_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=fit)
                if fit:
                    self.onehot_columns[col] = dummies.columns.tolist()
                else:
                    columns = getattr(self, 'onehot_columns', {}).get(col)
                    if columns is None and self.feature_names is not None:
                        # Pipelines saved before onehot_columns stored this schema
                        # in the complete feature list instead.
                        columns = [name for name in self.feature_names
                                   if name.startswith(f'{col}_')]
                    if columns is None:
                        raise ValueError("Categorical encoder not fitted. Call with fit=True first.")
                    # Missing and unseen categories contribute no active indicators.
                    dummies = dummies.reindex(
                        columns=columns, fill_value=False
                    )
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Label encode high-cardinality categorical features
        label_encode_cols = [
            'Manufacture',
            'Feed Solution (FS)',
            'Draw Solution (DS)'
        ]
        
        for col in label_encode_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[f'{col}_encoded'] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                df = df.drop(columns=[col])
        
        # Drop non-numeric columns that shouldn't be used
        cols_to_drop = ['Author', 'Year']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        return df
    
    def scale_features(self, df, target_col='Membrane Flux (LMH)', fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column (will not be scaled)
        fit : bool
            Whether to fit scaler (True for training, False for test)
            
        Returns:
        --------
        pd.DataFrame : DataFrame with scaled features
        """
        df = df.copy()
        
        # Separate target if present
        if target_col in df.columns:
            target = df[target_col]
            features = df.drop(columns=[target_col])
        else:
            target = None
            features = df
        
        # Get numerical columns
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler = StandardScaler()
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
            self.numerical_cols = numerical_cols
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            features[numerical_cols] = self.scaler.transform(features[numerical_cols])
        
        # Recombine with target
        if target is not None:
            features[target_col] = target
        
        return features
    
    def fit_transform(self, df, target_col='Membrane Flux (LMH)'):
        """
        Complete feature engineering pipeline (fit and transform).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
            
        Returns:
        --------
        pd.DataFrame : Fully processed dataframe
        """
        print("Starting feature engineering pipeline...")
        
        # Step 1: Create derived features
        print("  1. Creating derived features...")
        df = self.create_derived_features(df)
        
        # Step 2: Encode categorical features
        print("  2. Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        # Step 3: Scale numerical features
        print("  3. Scaling numerical features...")
        df = self.scale_features(df, target_col=target_col, fit=True)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != target_col]
        
        print(f"✓ Feature engineering complete. Final shape: {df.shape}")
        print(f"✓ Number of features: {len(self.feature_names)}")
        
        return df
    
    def transform(self, df, target_col='Membrane Flux (LMH)'):
        """
        Transform new data using fitted pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
            
        Returns:
        --------
        pd.DataFrame : Fully processed dataframe
        """
        # Step 1: Create derived features
        df = self.create_derived_features(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Step 3: Scale numerical features
        df = self.scale_features(df, target_col=target_col, fit=False)
        
        return df
    
    def save(self, filepath='models/feature_engineer.pkl'):
        """Save the feature engineer object."""
        joblib.dump(self, filepath)
        print(f"✓ Feature engineer saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/feature_engineer.pkl'):
        """Load a saved feature engineer object."""
        return joblib.load(filepath)


def main():
    """Example usage of the feature engineering pipeline."""
    
    # Load clean data
    print("Loading clean data...")
    df = pd.read_csv('data/clean_data.csv')
    print(f"Original shape: {df.shape}")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Apply feature engineering
    df_processed = fe.fit_transform(df)
    
    # Display results
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Original features: 20")
    print(f"Final features: {len(fe.feature_names)}")
    print(f"\nFeature names:")
    for i, name in enumerate(fe.feature_names, 1):
        print(f"  {i}. {name}")
    
    # Save processed data
    df_processed.to_csv('data/processed_data.csv', index=False)
    print(f"\n✓ Processed data saved to data/processed_data.csv")
    
    # Save feature engineer
    fe.save()
    
    print("\n" + "="*80)
    print("Feature engineering pipeline complete!")
    print("="*80)


if __name__ == "__main__":
    main()
