import os
import json
import subprocess
import argparse
import time

def run_single_test(test_case_number):
    """
    Run a single test case through the FHERMA validator.
    """
    # Check if test case exists
    test_cases_dir = 'test_cases'
    test_case_path = os.path.join(test_cases_dir, f'test_case_{test_case_number}.json')
    
    if not os.path.exists(test_case_path):
        print(f"Test case {test_case_number} not found at {test_case_path}")
        return
    
    # Create results directory
    results_dir = 'app/results'
    os.makedirs(results_dir, exist_ok=True)
    
    result_path = os.path.join(results_dir, f'result_{test_case_number}.json')
    
    print(f"Running validation for test case {test_case_number}...")
    
    # Convert paths to use forward slashes for Docker
    test_case_path_docker = test_case_path.replace('\\', '/')
    
    # Get current working directory
    current_dir = os.getcwd()
    
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
                
                print(f"Expected: {actual_value:.2f}, Predicted: {predicted_value:.2f}")
                print(f"Absolute Error: {abs(actual_value - predicted_value):.2f}")
                print(f"Percentage Error: {abs(actual_value - predicted_value) / actual_value * 100:.2f}%")
            else:
                print(f"Compilation error: {result['compilation_error']}")
        else:
            print(f"Result file not found: {app_result_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running validator: {e}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single test case through the FHERMA validator.')
    parser.add_argument('test_case_number', type=int, help='The test case number to run (1-10)')
    
    args = parser.parse_args()
    
    if args.test_case_number < 1 or args.test_case_number > 10:
        print("Test case number must be between 1 and 10")
    else:
        run_single_test(args.test_case_number) 