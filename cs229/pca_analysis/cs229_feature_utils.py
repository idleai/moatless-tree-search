"""
Utility functions for feature extraction from SWE-bench dataset.
We can see what is already done in the original SWE-bench codebase not to repeat steps
"""

import re
from typing import Dict, List, Tuple, Optional
import hashlib


def parse_git_patch(patch: str) -> Dict[str, any]:
    """
    Parse a git patch to extract statistics.
    
    Args:
        patch: Git patch as a string
        
    Returns:
        Dictionary with patch statistics
    """
    if not patch:
        return {
            'lines_added': 0,
            'lines_removed': 0,
            'lines_edited': 0,
            'files_edited': 0,
            'num_hunks': 0,
            'patch_entropy': 0.0
        }
    
    lines_added = 0
    lines_removed = 0
    files_edited = set()
    num_hunks = 0
    
    # split patch into lines
    lines = patch.split('\n')
    
    for line in lines:
        # count file markers
        if line.startswith('diff --git') or line.startswith('--- a/') or line.startswith('+++ b/'):
            # extract filename
            match = re.search(r'[ab]/(.+?)(?:\s|$)', line)
            if match:
                files_edited.add(match.group(1))
        
        # count hunks
        elif line.startswith('@@'):
            num_hunks += 1
        
        # count added/removed lines (ignore file markers)
        elif line.startswith('+') and not line.startswith('+++'):
            lines_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_removed += 1
    
    lines_edited = lines_added + lines_removed
    
    # calculate patch entropy (diversity of changes)
    patch_entropy = calculate_patch_entropy(patch)
    
    return {
        'lines_added': lines_added,
        'lines_removed': lines_removed,
        'lines_edited': lines_edited,
        'files_edited': len(files_edited),
        'num_hunks': num_hunks,
        'patch_entropy': patch_entropy
    }


def calculate_patch_entropy(patch: str) -> float:
    """
    Calculate Shannon entropy of patch content as a measure of change diversity.
    
    Args:
        patch: Git patch as string
        
    Returns:
        Entropy value (0 = uniform, higher = more diverse)
    """
    if not patch:
        return 0.0
    
    # count character frequencies (simplified entropy)
    char_counts = {}
    for char in patch:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    total_chars = len(patch)
    entropy = 0.0
    
    for count in char_counts.values():
        if count > 0:
            p = count / total_chars
            entropy -= p * (p if p == 0 else __import__('math').log2(p))
    
    return entropy


# this is not used then
def count_functions_edited(patch: str) -> int:
    """
    Estimate number of functions edited in a patch.
    Uses heuristics to detect function definitions.
    
    Args:
        patch: Git patch as string
        
    Returns:
        Estimated number of functions edited
    """
    if not patch:
        return 0
    
    # patterns for function definitions in different languages
    function_patterns = [
        r'^\s*def\s+\w+\s*\(',  # Python
        r'^\s*function\s+\w+\s*\(',  # JavaScript
        r'^\s*\w+\s+\w+\s*\([^)]*\)\s*{',  # Java/C/C++
        r'^\s*public\s+\w+\s+\w+\s*\(',  # Java public methods
        r'^\s*private\s+\w+\s+\w+\s*\(',  # Java private methods
    ]
    
    functions_seen = set()
    lines = patch.split('\n')
    
    for line in lines:
        # only look at added/removed lines
        if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
            clean_line = line[1:].strip()
            
            for pattern in function_patterns:
                if re.match(pattern, clean_line):
                    # extract function name as identifier
                    func_match = re.search(r'\b(\w+)\s*\(', clean_line)
                    if func_match:
                        functions_seen.add(func_match.group(1))
    
    return len(functions_seen)


def extract_test_info(instance: Dict) -> Tuple[int, int]:
    """
    Extract test information from instance metadata.
    
    Args:
        instance: SWE-bench instance dictionary
        
    Returns:
        Tuple of (fail_to_pass_count, pass_to_pass_count)
    """
    import json
    
    fail_to_pass = instance.get('fail_to_pass', instance.get('FAIL_TO_PASS', []))
    pass_to_pass = instance.get('pass_to_pass', instance.get('PASS_TO_PASS', []))
    
    # handle both list and string formats
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass) if fail_to_pass else []
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass) if pass_to_pass else []
    
    return len(fail_to_pass), len(pass_to_pass)


def get_repo_type(instance_id: str) -> str:
    """
    Extract repository type from instance ID.
    
    Args:
        instance_id: Instance ID (e.g., 'django__django-12345')
        
    Returns:
        Repository name (e.g., 'django')
    """
    # format is typically: repo__project-number
    parts = instance_id.split('__')
    if len(parts) >= 2:
        return parts[0]
    return 'unknown'


def compute_context_stats(instance: Dict) -> Dict[str, any]:
    """
    Compute statistics about the context (files and lines).
    Uses expected_spans from evaluation files to estimate context size.
    
    Args:
        instance: SWE-bench instance dictionary
        
    Returns:
        Dictionary with context statistics
    """
    expected_spans = instance.get('expected_spans', {})
    test_file_spans = instance.get('test_file_spans', {})
    
    # combine all context files
    all_files = set(expected_spans.keys()) | set(test_file_spans.keys())
    
    if not all_files:
        # fallback to 0 if no context available
        return {
            'context_files': 0,
            'context_lines': 0,
            'avg_file_size': 0,
            'max_file_size': 0,
            'num_directories': 0,
        }
    
    # count total spans and estimate lines
    total_spans = 0
    file_span_counts = []
    directories = set()
    
    for file_path in all_files:
        # count spans in this file
        file_spans = 0
        if file_path in expected_spans:
            file_spans += len(expected_spans[file_path])
        if file_path in test_file_spans:
            file_spans += len(test_file_spans[file_path])
        
        total_spans += file_spans
        file_span_counts.append(file_spans)
        
        # extract directory
        if '/' in file_path:
            directory = '/'.join(file_path.split('/')[:-1])
            directories.add(directory)
    
    return {
        'context_files': len(all_files),
        'context_lines': total_spans,  # approximate: each span is 1 unit of context
        'avg_file_size': sum(file_span_counts) / len(file_span_counts) if file_span_counts else 0,
        'max_file_size': max(file_span_counts) if file_span_counts else 0,
        'num_directories': len(directories),
    }


def normalize_features(features_dict: Dict[str, float], 
                       method: str = 'standard') -> Dict[str, float]:
    """
    Normalize a dictionary of features.
    
    Args:
        features_dict: Dictionary of feature names to values
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Normalized features dictionary
    """
    # TODO: move processing from notebook to this function, for now it's only a placeholder
    return features_dict


def hash_instance_id(instance_id: str) -> str:
    """
    Create a short hash of instance ID for use as unique identifier.
    
    Args:
        instance_id: Full instance ID
        
    Returns:
        8-character hash
    """
    return hashlib.md5(instance_id.encode()).hexdigest()[:8]

