"""Extract features from SWE-bench Verified dataset for PCA analysis."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis import config, feature_utils


def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load SWE-bench dataset from JSON file."""
    print(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        instances = data
    elif isinstance(data, dict):
        instances = data.get('instances', data.get('data', []))
    else:
        raise ValueError(f"Unexpected dataset format: {type(data)}")
    
    print(f"Loaded {len(instances)} instances")
    return instances


def extract_features_from_instance(instance: Dict) -> Dict:
    """
    Extract all features from a single instance.
    
    Args:
        instance: SWE-bench instance dictionary
        
    Returns:
        Dictionary of extracted features
    """
    instance_id = instance.get('instance_id', 'unknown')
    
    # basic information
    problem_statement = instance.get('problem_statement', '')
    patch = instance.get('golden_patch', instance.get('patch', ''))
    repo_type = feature_utils.get_repo_type(instance_id)
    
    patch_stats = feature_utils.parse_git_patch(patch)
    
    fail_to_pass, pass_to_pass = feature_utils.extract_test_info(instance)
    
    functions_edited = feature_utils.count_functions_edited(patch)
    
    context_stats = feature_utils.compute_context_stats(instance)
    
    features = {
        'instance_id': instance_id,
        'repo_type': repo_type,
        
        # problem characteristics
        'problem_length_chars': len(problem_statement),
        'problem_length_words': len(problem_statement.split()),
        'problem_length_lines': problem_statement.count('\n') + 1 if problem_statement else 0,
        
        # context characteristics
        'context_num_files': context_stats['context_files'],
        'context_num_lines': context_stats['context_lines'],
        'context_avg_file_size': context_stats['avg_file_size'],
        'context_max_file_size': context_stats['max_file_size'],
        'context_num_directories': context_stats['num_directories'],
        
        # patch characteristics
        'patch_files_edited': patch_stats['files_edited'],
        'patch_functions_edited': functions_edited,
        'patch_lines_added': patch_stats['lines_added'],
        'patch_lines_removed': patch_stats['lines_removed'],
        'patch_lines_edited': patch_stats['lines_edited'],
        'patch_num_hunks': patch_stats['num_hunks'],
        'patch_entropy': patch_stats['patch_entropy'],
        
        # test characteristics
        'test_fail_to_pass': fail_to_pass,
        'test_pass_to_pass': pass_to_pass,
        'test_total': fail_to_pass + pass_to_pass,
        
        # computed ratios and metrics
        'patch_add_remove_ratio': (
            patch_stats['lines_added'] / max(patch_stats['lines_removed'], 1)
            if patch_stats['lines_removed'] > 0
            else patch_stats['lines_added']
        ),
        'patch_avg_hunk_size': (
            patch_stats['lines_edited'] / max(patch_stats['num_hunks'], 1)
            if patch_stats['num_hunks'] > 0
            else 0
        ),
    }
    
    return features


def extract_all_features(instances: List[Dict]) -> pd.DataFrame:
    """
    Extract features from all instances.
    
    Args:
        instances: List of SWE-bench instances
        
    Returns:
        DataFrame with extracted features
    """
    print("Extracting features from all instances...")
    
    all_features = []
    for i, instance in enumerate(instances):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(instances)} instances...")
        
        try:
            features = extract_features_from_instance(instance)
            all_features.append(features)
        except Exception as e:
            print(f"  Error processing instance {instance.get('instance_id', 'unknown')}: {e}")
            continue
    
    print(f"Successfully extracted features from {len(all_features)}/{len(instances)} instances")
    
    df = pd.DataFrame(all_features)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features based on existing columns.
    
    Args:
        df: DataFrame with basic features
        
    Returns:
        DataFrame with additional derived features
    """
    print("Adding derived features...")
    
    # TODO: review and select a definitice complexity score, for now it's a simple heuristic
    # UPDATE: is not used then since other features are being used
    df['complexity_score'] = (
        df['patch_files_edited'] * 0.3 +
        df['patch_lines_edited'] * 0.002 +
        df['patch_functions_edited'] * 0.5 +
        df['test_total'] * 0.2
    )
    
    # problem size category
    df['problem_size'] = pd.cut(
        df['problem_length_chars'],
        bins=[0, 500, 1500, float('inf')],
        labels=['small', 'medium', 'large']
    )
    
    # patch scope category
    df['patch_scope'] = pd.cut(
        df['patch_files_edited'],
        bins=[0, 1, 3, float('inf')],
        labels=['focused', 'moderate', 'widespread']
    )
    
    return df


def main():
    """Main execution function."""
    
    instances = load_dataset(config.DATASET_PATH)
    
    df = extract_all_features(instances)
    
    df = add_derived_features(df)
    
    print(f"\nTotal instances: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print("\nFeature columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nBasic statistics:")
    print(df.describe())
    
    output_path = config.OUTPUT_CSV
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    
    print("\nSample of extracted features:")
    print(df.head())
    
    print("Feature extraction complete...")
    
    return df


if __name__ == "__main__":
    df = main()

