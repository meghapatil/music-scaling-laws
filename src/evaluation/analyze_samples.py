import json
from pathlib import Path
import re

def analyze_abc_file(filepath):
    """Analyze a single ABC file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    analysis = {
        'has_header': bool(re.search(r'X:\d+', content)),
        'has_title': bool(re.search(r'T:.+', content)),
        'has_meter': bool(re.search(r'M:\d+/\d+', content)),
        'has_key': bool(re.search(r'K:[A-G]', content)),
        'has_notes': bool(re.search(r'[A-Ga-g]', content)),
        'has_barlines': '|' in content,
        'length': len(content),
        'num_barlines': content.count('|')
    }
    
    analysis['is_valid'] = all([
        analysis['has_header'],
        analysis['has_meter'],
        analysis['has_key'],
        analysis['has_notes'],
        analysis['has_barlines']
    ])
    
    return analysis, content

def analyze_all_samples(sample_dir):
    """Analyze all samples in a directory"""
    sample_dir = Path(sample_dir)
    
    if not sample_dir.exists():
        return [], 0
    
    all_analyses = []
    valid_count = 0
    total_length = 0
    total_barlines = 0
    
    for abc_file in sorted(sample_dir.glob('sample_*.abc')):
        analysis, content = analyze_abc_file(abc_file)
        analysis['filename'] = abc_file.name
        all_analyses.append(analysis)
        
        if analysis['is_valid']:
            valid_count += 1
            total_length += analysis['length']
            total_barlines += analysis['num_barlines']
    
    avg_length = total_length / valid_count if valid_count > 0 else 0
    avg_barlines = total_barlines / valid_count if valid_count > 0 else 0
    
    return all_analyses, valid_count, avg_length, avg_barlines

def main():
    models = ['tiny', 'small', 'large']
    
    all_results = {}
    
    print("="*60)
    print("SAMPLE ANALYSIS RESULTS")
    print("="*60)
    
    for model in models:
        sample_dir = f'results/generated_samples/{model}'
        if not Path(sample_dir).exists():
            print(f"\n❌ {model.upper()}: Directory not found")
            continue
        
        analyses, valid_count, avg_length, avg_barlines = analyze_all_samples(sample_dir)
        
        all_results[model] = {
            'total_samples': len(analyses),
            'valid_samples': valid_count,
            'validity_rate': valid_count / len(analyses) if analyses else 0,
            'avg_length': avg_length,
            'avg_barlines': avg_barlines,
            'details': analyses
        }
        
        print(f"\n{model.upper()} Model:")
        print(f"  Total samples: {len(analyses)}")
        print(f"  Valid samples: {valid_count}")
        print(f"  Validity rate: {valid_count/len(analyses)*100:.1f}%")
        print(f"  Avg length: {avg_length:.0f} chars")
        print(f"  Avg barlines: {avg_barlines:.1f}")
    
    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE (Copy to Report)")
    print("="*60)
    print("\nModel   | Valid Samples | Validity Rate | Avg Length | Avg Barlines")
    print("--------|---------------|---------------|------------|-------------")
    
    for model in models:
        if model in all_results:
            r = all_results[model]
            total = r['total_samples']
            valid = r['valid_samples']
            validity = r['validity_rate'] * 100
            length = r['avg_length']
            barlines = r['avg_barlines']
            print(f"{model.capitalize():7} | {valid}/{total:2}          | {validity:5.1f}%        | {length:4.0f} chars | {barlines:4.1f}")
    
    # Save results
    with open('results/sample_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: results/sample_analysis.json")
    print("="*60)

if __name__ == '__main__':
    main()
