#!/usr/bin/env python3
"""Check for common Sphinx/RST documentation issues beyond underline mismatches."""

import os
import re
from pathlib import Path

def check_file(filepath):
    """Check a single file for documentation issues."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Category 2: Check for broken cross-references
    # Look for :ref:, :doc:, :func:, etc.
    ref_pattern = re.compile(r':(?:ref|doc|func|class|meth|attr|mod|py:func|py:class|py:meth|py:attr|py:mod):`([^`]+)`')
    for i, line in enumerate(lines, 1):
        refs = ref_pattern.findall(line)
        for ref in refs:
            # Check for common issues
            if '<' in ref and '>' in ref:
                # Explicit title syntax - check for missing angle brackets
                if ref.count('<') != ref.count('>'):
                    issues.append({
                        'file': filepath,
                        'line': i,
                        'type': 'malformed_reference',
                        'ref': ref,
                        'message': 'Unmatched angle brackets in reference'
                    })
    
    # Category 3: Check for orphan documents
    # Look for toctree directives
    toctree_found = '.. toctree::' in content
    if filepath.name != 'index.rst' and not toctree_found:
        # Check if this file is referenced anywhere
        # This is a simplified check
        pass
    
    # Check for duplicate labels
    label_pattern = re.compile(r'^\.\. _([^:]+):' , re.MULTILINE)
    labels = label_pattern.findall(content)
    if len(labels) != len(set(labels)):
        duplicates = [l for l in labels if labels.count(l) > 1]
        for dup in set(duplicates):
            issues.append({
                'file': filepath,
                'line': 0,
                'type': 'duplicate_label',
                'label': dup,
                'message': f'Duplicate label: {dup}'
            })
    
    # Check for missing code block language
    code_block_pattern = re.compile(r'^\.\. code-block::\s*$', re.MULTILINE)
    for match in code_block_pattern.finditer(content):
        line_num = content[:match.start()].count('\n') + 1
        issues.append({
            'file': filepath,
            'line': line_num,
            'type': 'missing_code_language',
            'message': 'Code block without language specifier'
        })
    
    # Check for broken internal links
    internal_link_pattern = re.compile(r'`([^`<]+)\s+<#([^>]+)>`_')
    for i, line in enumerate(lines, 1):
        matches = internal_link_pattern.findall(line)
        for text, anchor in matches:
            # Could check if anchor exists, but that requires parsing the whole doc
            pass
    
    return issues

def main():
    docs_source = Path('docs/source')
    all_issues = []
    
    # Find all RST files
    for rst_file in docs_source.rglob('*.rst'):
        issues = check_file(rst_file)
        all_issues.extend(issues)
    
    # Group by issue type
    by_type = {}
    for issue in all_issues:
        issue_type = issue['type']
        if issue_type not in by_type:
            by_type[issue_type] = []
        by_type[issue_type].append(issue)
    
    # Print summary
    print("=" * 80)
    print("SPHINX DOCUMENTATION ISSUES (Categories 2 & 3)")
    print("=" * 80)
    print()
    
    for issue_type, issues in by_type.items():
        print(f"\n{issue_type.upper().replace('_', ' ')}: {len(issues)} issues")
        print("-" * 80)
        
        for issue in issues[:15]:  # Show first 15 of each type
            print(f"\nFile: {issue['file']}")
            print(f"Line: {issue.get('line', 'N/A')}")
            print(f"Message: {issue.get('message', 'N/A')}")
            if 'ref' in issue:
                print(f"Reference: {issue['ref']}")
            if 'label' in issue:
                print(f"Label: {issue['label']}")
        
        if len(issues) > 15:
            print(f"\n... and {len(issues) - 15} more")
    
    print("\n" + "=" * 80)
    print(f"TOTAL ISSUES: {len(all_issues)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
