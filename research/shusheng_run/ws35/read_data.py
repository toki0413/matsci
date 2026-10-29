#!/usr/bin/env python3
"""Read and analyze the MgH2 activation energy dataset."""

import pandas as pd
import json
import sys

def load_datasets(data_dir):
    """Load both the original and cleaned datasets."""
    original = pd.read_csv(f"{data_dir}/ea_dataset.csv")
    cleaned = pd.read_csv(f"{data_dir}/cleaned_ea_dataset.csv")
    return original, cleaned

def print_basic_info(name, df):
    """Print basic information about a dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Dataset")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")

def print_statistics(name, df):
    """Print statistical summary for numeric columns."""
    print(f"\n{'='*60}")
    print(f"{name} - Statistical Summary")
    print(f"{'='*60}")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns: {list(numeric_cols)}")
        print(f"\n{df[numeric_cols].describe()}")
    else:
        print("No numeric columns to summarize.")

def print_categorical_summary(name, df):
    """Print categorical value counts."""
    print(f"\n{'='*60}")
    print(f"{name} - Categorical Summary")
    print(f"{'='*60}")
    categorical_cols = df.select_dtypes(include=['string', 'category', 'object']).columns
    for col in categorical_cols:
        if col in ['material', 'quote', 'locator', 'url', 'notes']:
            continue  # Skip long text columns
        value_counts = df[col].value_counts()
        print(f"\n{col}:")
        print(f"  Total unique values: {len(value_counts)}")
        print(f"  Top 10 values:")
        for val, count in value_counts.head(10).items():
            print(f"    {val}: {count}")

def compare_datasets(original, cleaned):
    """Compare the original and cleaned datasets."""
    print(f"\n{'='*60}")
    print("Dataset Comparison")
    print(f"{'='*60}")
    print(f"\nOriginal rows: {len(original)}, Cleaned rows: {len(cleaned)}")
    print(f"Rows removed: {len(original) - len(cleaned)}")
    
    # Check if cleaned is a subset of original (by material and other key fields)
    print(f"\nColumn comparison:")
    print(f"  Original columns: {len(original.columns)}")
    print(f"  Cleaned columns: {len(cleaned.columns)}")
    extra_cols = set(cleaned.columns) - set(original.columns)
    if extra_cols:
        print(f"  Extra columns in cleaned: {extra_cols}")

def analyze_ea_distribution(df):
    """Analyze the distribution of activation energies."""
    print(f"\n{'='*60}")
    print("Activation Energy Analysis")
    print(f"{'='*60}")
    
    # Filter out rows with missing Ea values
    ea_col = 'Ea_parsed' if 'Ea_parsed' in df.columns else 'Ea_kJ_per_mol'
    ea_values = df[ea_col].dropna()
    
    print(f"\nEa values ({ea_col}):")
    print(f"  Count: {len(ea_values)}")
    print(f"  Mean: {ea_values.mean():.2f} kJ/mol")
    print(f"  Median: {ea_values.median():.2f} kJ/mol")
    print(f"  Min: {ea_values.min():.2f} kJ/mol")
    print(f"  Max: {ea_values.max():.2f} kJ/mol")
    print(f"  Std: {ea_values.std():.2f} kJ/mol")
    
    # Group by material_class
    if 'material_class' in df.columns:
        print(f"\nEa by material_class:")
        grouped = df.groupby('material_class')[ea_col].agg(['count', 'mean', 'min', 'max'])
        print(grouped)

def main():
    data_dir = "/workspace/research/shusheng_run/ws35/data"
    
    print("Loading datasets...")
    original, cleaned = load_datasets(data_dir)
    
    # Print basic info
    print_basic_info("Original", original)
    print_basic_info("Cleaned", cleaned)
    
    # Print statistics
    print_statistics("Original", original)
    print_statistics("Cleaned", cleaned)
    
    # Print categorical summaries
    print_categorical_summary("Original", original)
    print_categorical_summary("Cleaned", cleaned)
    
    # Compare datasets
    compare_datasets(original, cleaned)
    
    # Analyze Ea distribution
    analyze_ea_distribution(cleaned)
    
    # Print some specific notes about the dataset
    print(f"\n{'='*60}")
    print("Dataset Notes Summary")
    print(f"{'='*60}")
    
    if 'notes' in cleaned.columns:
        notes_with_issues = cleaned[cleaned['notes'].str.contains('QUESTIONABLE|Must be filtered|Kept as a separate row', na=False, case=False)]
        print(f"\nRows with special notes ({len(notes_with_issues)}):")
        for _, row in notes_with_issues.iterrows():
            print(f"  - {row['material']}: {row['notes'][:80]}...")
    
    # Save summary to JSON
    summary = {
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "original_columns": list(original.columns),
        "cleaned_columns": list(cleaned.columns),
        "ea_mean_cleaned": float(cleaned['Ea_parsed'].mean()) if 'Ea_parsed' in cleaned.columns else None,
        "ea_median_cleaned": float(cleaned['Ea_parsed'].median()) if 'Ea_parsed' in cleaned.columns else None
    }
    
    with open(f"{data_dir}/dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nSummary saved to {data_dir}/dataset_summary.json")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()