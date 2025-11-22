"""
Forward Osmosis Membrane Flux Dataset Explorer
This script provides utilities to load, explore, and analyze the FO membrane dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class FODatasetExplorer:
    """Class to explore and analyze the Forward Osmosis dataset."""
    
    def __init__(self, filepath='Supplementary Data.xlsx'):
        """Initialize the explorer with the dataset filepath."""
        self.filepath = filepath
        self.raw_data = None
        self.input_data = None
        self.comparison_data = None
        self.load_data()
    
    def load_data(self):
        """Load all sheets from the Excel file."""
        print(f"Loading data from {self.filepath}...")
        
        # Load raw data
        self.raw_data = pd.read_excel(self.filepath, sheet_name='Raw_data')
        # Remove unnamed columns
        self.raw_data = self.raw_data.loc[:, ~self.raw_data.columns.str.contains('^Unnamed')]
        
        # Load input data (encoded)
        self.input_data = pd.read_excel(self.filepath, sheet_name='Input_data')
        
        # Load comparison data
        self.comparison_data = pd.read_excel(self.filepath, sheet_name='Comparison')
        
        print(f"✓ Loaded Raw_data: {self.raw_data.shape}")
        print(f"✓ Loaded Input_data: {self.input_data.shape}")
        print(f"✓ Loaded Comparison: {self.comparison_data.shape}")
    
    def summary(self):
        """Print a summary of the dataset."""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total observations: {len(self.raw_data)}")
        print(f"Total features: {len(self.raw_data.columns)}")
        print(f"Target variable: Membrane Flux (LMH)")
        print(f"\nDate range: {self.raw_data['Year'].min():.0f} - {self.raw_data['Year'].max():.0f}")
        print(f"Number of authors: {self.raw_data['Author'].nunique()}")
        print(f"Membrane types: {self.raw_data['Membrane AL Type'].nunique()}")
        print(f"Manufacturers: {self.raw_data['Manufacture'].nunique()}")
    
    def missing_values_report(self):
        """Generate a detailed missing values report."""
        print("\n" + "="*80)
        print("MISSING VALUES REPORT")
        print("="*80)
        
        missing = self.raw_data.isnull().sum()
        missing_pct = (missing / len(self.raw_data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("✓ No missing values found!")
    
    def categorical_analysis(self):
        """Analyze categorical features."""
        print("\n" + "="*80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*80)
        
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['Author']:  # Skip author names
                print(f"\n{col}:")
                value_counts = self.raw_data[col].value_counts()
                print(value_counts)
    
    def numerical_analysis(self):
        """Analyze numerical features."""
        print("\n" + "="*80)
        print("NUMERICAL FEATURES STATISTICS")
        print("="*80)
        
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'Year']
        
        stats = self.raw_data[numerical_cols].describe()
        print(stats)
    
    def target_analysis(self):
        """Analyze the target variable (Membrane Flux)."""
        print("\n" + "="*80)
        print("TARGET VARIABLE ANALYSIS: Membrane Flux (LMH)")
        print("="*80)
        
        flux = self.raw_data['Membrane Flux (LMH)']
        
        print(f"Mean: {flux.mean():.2f} LMH")
        print(f"Median: {flux.median():.2f} LMH")
        print(f"Std Dev: {flux.std():.2f} LMH")
        print(f"Min: {flux.min():.2f} LMH")
        print(f"Max: {flux.max():.2f} LMH")
        print(f"Range: {flux.max() - flux.min():.2f} LMH")
    
    def correlation_analysis(self, top_n=10):
        """Analyze correlations with the target variable."""
        print("\n" + "="*80)
        print(f"TOP {top_n} CORRELATIONS WITH MEMBRANE FLUX")
        print("="*80)
        
        # Get numerical columns
        numerical_data = self.raw_data.select_dtypes(include=[np.number])
        
        # Calculate correlations with target
        correlations = numerical_data.corr()['Membrane Flux (LMH)'].abs().sort_values(ascending=False)
        
        # Exclude self-correlation
        correlations = correlations[correlations.index != 'Membrane Flux (LMH)']
        
        print(correlations.head(top_n))
    
    def get_clean_data(self, drop_high_missing=True, missing_threshold=0.5):
        """
        Get a cleaned version of the dataset.
        
        Parameters:
        -----------
        drop_high_missing : bool
            Whether to drop columns with high missing percentage
        missing_threshold : float
            Threshold for dropping columns (default: 50%)
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataset
        """
        df = self.raw_data.copy()
        
        if drop_high_missing:
            # Calculate missing percentage
            missing_pct = df.isnull().sum() / len(df)
            
            # Drop columns with high missing percentage
            cols_to_drop = missing_pct[missing_pct > missing_threshold].index
            print(f"\nDropping columns with >{missing_threshold*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Drop rows with any remaining missing values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing values")
        
        print(f"\nCleaned dataset shape: {df.shape}")
        
        return df
    
    def export_clean_data(self, output_path='clean_data.csv'):
        """Export cleaned data to CSV."""
        clean_df = self.get_clean_data()
        clean_df.to_csv(output_path, index=False)
        print(f"\n✓ Exported clean data to {output_path}")
        return clean_df


def main():
    """Main function to run the explorer."""
    print("="*80)
    print("FORWARD OSMOSIS MEMBRANE FLUX DATASET EXPLORER")
    print("="*80)
    
    # Initialize explorer
    explorer = FODatasetExplorer()
    
    # Run analyses
    explorer.summary()
    explorer.missing_values_report()
    explorer.target_analysis()
    explorer.correlation_analysis()
    explorer.categorical_analysis()
    explorer.numerical_analysis()
    
    # Export clean data
    print("\n" + "="*80)
    print("EXPORTING CLEAN DATA")
    print("="*80)
    clean_df = explorer.export_clean_data()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the clean_data.csv file")
    print("2. Perform exploratory data analysis (EDA)")
    print("3. Build predictive models")
    print("4. Validate using the Comparison sheet")


if __name__ == "__main__":
    main()
