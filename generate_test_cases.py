import pandas as pd
import numpy as np
import json
import os

def generate_test_cases(num_cases=10):
    """
    Generate test cases from the dataset for FHE validation.
    """
    # Load data
    X = pd.read_csv('data/X_train.csv')
    y = pd.read_csv('data/y_train.csv')
    
    # Convert to numpy arrays
    X_values = X.values
    y_values = y.values.flatten()
    
    # Create test_cases directory in app folder
    test_cases_dir = 'app/test_cases'
    os.makedirs(test_cases_dir, exist_ok=True)
    
    # Select a fixed range of samples deterministically
    start_index = 0  # Change this to any fixed starting point
    indices = list(range(start_index, start_index + num_cases))
    
    # Generate test cases
    for i, idx in enumerate(indices):
        features = X_values[idx].tolist()
        target = float(y_values[idx])
        
        # Create test case in the required format
        test_case = [{
            "scheme": "CKKS",
            "significant_slots_number": 1,
            "runs": [
                {
                    "input": [
                        {
                            "name": "sample",
                            "value": features
                        }
                    ],
                    "output": [
                        target
                    ]
                }
            ]
        }]
        
        # Save test case to file
        test_case_path = os.path.join(test_cases_dir, f'test_case_{i+1}.json')
        with open(test_case_path, 'w') as f:
            json.dump(test_case, f, indent=4)
        
        print(f"Generated test case {i+1} with target value: {target}")
    
    # Create a summary file with all test cases
    summary = {
        "test_cases": [
            {
                "file": f"test_case_{i+1}.json",
                "expected_output": float(y_values[idx])
            } for i, idx in enumerate(indices)
        ]
    }
    
    with open(os.path.join(test_cases_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nGenerated {num_cases} test cases in {test_cases_dir}")

if __name__ == "__main__":
    generate_test_cases(10) 