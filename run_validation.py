import os
import json
import subprocess
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_validation():
    """
    Run all test cases through the FHERMA validator and calculate accuracy metrics.
    """
    # Start timing the validation process
    start_time = time.time()
    
    # Check if test cases exist
    test_cases_dir = 'test_cases'
    if not os.path.exists(test_cases_dir):
        print("Test cases directory not found. Please run generate_test_cases.py first.")
        return
    
    # Load summary file
    summary_path = os.path.join(test_cases_dir, 'summary.json')
    if not os.path.exists(summary_path):
        print("Summary file not found. Please run generate_test_cases.py first.")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Create results directory
    results_dir = 'app/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run validation for each test case
    actual_values = []
    predicted_values = []
    expected_values = []
    
    # Get current working directory
    current_dir = os.getcwd()
    
    for i, test_case in enumerate(summary['test_cases']):
        test_case_file = test_case['file']
        expected_output = test_case['expected_output']
        expected_values.append(expected_output)
        
        test_case_path = os.path.join(test_cases_dir, test_case_file)
        result_path = os.path.join(results_dir, f'result_{i+1}.json')
        
        # Convert paths to use forward slashes for Docker
        test_case_path_docker = test_case_path.replace('\\', '/')
        
        print(f"\nRunning validation for test case {i+1}...")
        
        # Run the validator using Docker with Windows PowerShell syntax
        cmd = f'docker run -v "{current_dir}:/fherma" yashalabinc/fherma-validator --project-folder=/fherma/app --testcase=/fherma/{test_case_path_docker}'
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            
            # Wait for the result file to be created
            time.sleep(2)
            
            # Check if result.json exists in the app directory
            app_result_path = 'app/result.json'
            if os.path.exists(app_result_path):
                # Copy the result to our results directory
                with open(app_result_path, 'r') as f:
                    result = json.load(f)
                
                # Save the result to our results directory
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=4)
                
                if result.get('compilation_error') is None:
                    # Extract the predicted value
                    predicted_value = result['testcases'][0]['runs'][0]['result'][0]
                    actual_value = result['testcases'][0]['runs'][0]['expected_output'][0]
                    
                    predicted_values.append(predicted_value)
                    actual_values.append(actual_value)
                    
                    print(f"  Expected: {actual_value:.2f}, Predicted: {predicted_value:.2f}")
                    print(f"  Absolute Error: {abs(actual_value - predicted_value):.2f}")
                else:
                    print(f"  Compilation error: {result['compilation_error']}")
            else:
                print(f"  Result file not found: {app_result_path}")
        
        except subprocess.CalledProcessError as e:
            print(f"  Error running validator: {e}")
    
    # Calculate accuracy metrics
    if len(predicted_values) > 0:
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predicted_values)
        
        # Calculate mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
        
        # Create summary report
        report = {
            "metrics": {
                "mean_absolute_error": mae,
                "mean_squared_error": mse,
                "root_mean_squared_error": rmse,
                "r2_score": r2,
                "mean_absolute_percentage_error": mape
            },
            "test_cases": [
                {
                    "test_case": summary['test_cases'][i]['file'],
                    "expected": actual_values[i],
                    "predicted": predicted_values[i],
                    "absolute_error": abs(actual_values[i] - predicted_values[i]),
                    "percentage_error": abs(actual_values[i] - predicted_values[i]) / actual_values[i] * 100
                } for i in range(len(actual_values))
            ]
        }
        
        # Save report
        report_path = os.path.join(results_dir, 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Calculate total time taken
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print("\n===== Validation Summary =====")
        print(f"Total test cases: {len(summary['test_cases'])}")
        print(f"Successfully validated: {len(predicted_values)}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Total validation time: {total_time:.2f} seconds")
        print(f"Average time per test case: {total_time/len(predicted_values):.2f} seconds")
        print(f"Validation report saved to {report_path}")
    else:
        print("\nNo successful validations to calculate metrics.")

if __name__ == "__main__":
    run_validation() 