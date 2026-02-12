#!/usr/bin/env python3
"""Check RST documentation for common issues."""

import os
import sys
from pathlib import Path

def check_rst_file(filepath):
    """Check a single RST file for common documentation issues."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(len(lines) - 1):
        current = lines[i].rstrip()
        next_line = lines[i + 1].rstrip()
        
        # Check for title underlines (Category 1: Title Underline Length Mismatches)
        if next_line and all(c in '=-~^"#*' for c in next_line) and len(next_line) > 0:
            if len(current) != len(next_line):
                issues.append({
                    'file': filepath,
                    'line': i + 1,
                    'type': 'underline_length_mismatch',
                    'title': current,
                    'underline': next_line,
                    'title_len': len(current),
                    'underline_len': len(next_line)
                })
        
        # Check for missing blank lines before sections
        if next_line and all(c in '=-~^"#*' for c in next_line):
            if i > 0 and lines[i - 1].strip():
                prev_line = lines[i - 1].rstrip()
                if not all(c in '=-~^"#*' for c in prev_line):
                    issues.append({
                        'file': filepath,
                        'line': i + 1,
                        'type': 'missing_blank_before_section',
                        'title': current
                    })
    
    return issues

def main():
    docs_source = Path('docs/source')
    all_issues = []
    
    # Find all RST files
    for rst_file in docs_source.rglob('*.rst'):
        issues = check_rst_file(rst_file)
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
    print("DOCUMENTATION ISSUES SUMMARY")
    print("=" * 80)
    print()
    
    for issue_type, issues in by_type.items():
        print(f"\n{issue_type.upper().replace('_', ' ')}: {len(issues)} issues")
        print("-" * 80)
        
        for issue in issues[:10]:  # Show first 10 of each type
            print(f"\nFile: {issue['file']}")
            print(f"Line: {issue['line']}")
            if issue_type == 'underline_length_mismatch':
                print(f"Title: '{issue['title']}' (length: {issue['title_len']})")
                print(f"Underline: '{issue['underline']}' (length: {issue['underline_len']})")
            else:
                print(f"Title: '{issue.get('title', 'N/A')}'")
        
        if len(issues) > 10:
            print(f"\n... and {len(issues) - 10} more")
    
    print("\n" + "=" * 80)
    print(f"TOTAL ISSUES: {len(all_issues)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
