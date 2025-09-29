import numpy as np
import pandas as pd

# Data from your results
data = {
    'seed_42': {
        'v1.6': [91.75, 93.16, 88.60],
        'v3': [86.00, 92.76, 87.05]
    },
    'seed_188': {
        'v1.6': [96.47, 90.63, 88.74],
        'v3': [93.92, 86.96, 82.29]
    },
    'seed_888': {
        'v1.6': [97.25, 90.27, 87.82],
        'v3': [90.88, 90.63, 87.09]
    },
    'seed_0': {
        'v1.6': [93.14, 91.49, 88.78],
        'v3': [91.67, 90.85, 87.07]
    }
}

metrics = ['UA', 'IRA', 'CRA']
versions = ['v1.6', 'v3']

def compute_statistics():
    print("="*60)
    print("UNLEARNCANVAS BENCHMARK STATISTICS")
    print("="*60)
    
    # Organize data for analysis
    results = {}
    
    for version in versions:
        results[version] = {
            'UA': [],
            'IRA': [],
            'CRA': [],
            'overall': []
        }
        
        for seed_data in data.values():
            ua, ira, cra = seed_data[version]
            results[version]['UA'].append(ua)
            results[version]['IRA'].append(ira)
            results[version]['CRA'].append(cra)
            results[version]['overall'].append(np.mean([ua, ira, cra]))
    
    # Compute and display statistics for each version
    for version in versions:
        print(f"\n{version.upper()} STATISTICS:")
        print("-" * 40)
        
        for metric in ['UA', 'IRA', 'CRA']:
            values = results[version][metric]
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation
            var = np.var(values, ddof=1)  # Sample variance
            
            print(f"{metric}:")
            print(f"  Mean: {mean:.2f}%")
            print(f"  Std Dev: {std:.2f}")
            print(f"  Variance: {var:.2f}")
            print(f"  Values: {values}")
            print()
        
        # Overall statistics (average of the three metrics per seed)
        overall_values = results[version]['overall']
        overall_mean = np.mean(overall_values)
        overall_std = np.std(overall_values, ddof=1)
        overall_var = np.var(overall_values, ddof=1)
        
        print(f"OVERALL (Average of UA, IRA, CRA per seed):")
        print(f"  Mean: {overall_mean:.2f}%")
        print(f"  Std Dev: {overall_std:.2f}")
        print(f"  Variance: {overall_var:.2f}")
        print(f"  Values: {[f'{v:.2f}' for v in overall_values]}")
        print()
    
    # Comparison between versions
    print("COMPARISON BETWEEN VERSIONS:")
    print("-" * 40)
    
    for metric in ['UA', 'IRA', 'CRA']:
        v16_mean = np.mean(results['v1.6'][metric])
        v3_mean = np.mean(results['v3'][metric])
        diff = v16_mean - v3_mean
        
        print(f"{metric}: v1.6 ({v16_mean:.2f}%) vs v3 ({v3_mean:.2f}%) | Diff: {diff:+.2f}%")
    
    # Overall comparison
    v16_overall = np.mean(results['v1.6']['overall'])
    v3_overall = np.mean(results['v3']['overall'])
    overall_diff = v16_overall - v3_overall
    
    print(f"OVERALL: v1.6 ({v16_overall:.2f}%) vs v3 ({v3_overall:.2f}%) | Diff: {overall_diff:+.2f}%")
    
    # Create summary DataFrame
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    summary_data = []
    
    for version in versions:
        for metric in ['UA', 'IRA', 'CRA', 'Overall']:
            if metric == 'Overall':
                values = results[version]['overall']
            else:
                values = results[version][metric]
            
            summary_data.append({
                'Version': version,
                'Metric': metric,
                'Mean': np.mean(values),
                'Std Dev': np.std(values, ddof=1),
                'Variance': np.var(values, ddof=1),
                'Min': np.min(values),
                'Max': np.max(values)
            })
    
    df = pd.DataFrame(summary_data)
    print(df.round(2))
    
    return results, df

# Run the analysis
if __name__ == "__main__":
    results, summary_df = compute_statistics()