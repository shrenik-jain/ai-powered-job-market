import subprocess
from time import sleep

def run_script(script_name):
    """
    Function to run a Python script using subprocess module

    Args:
        script_name (str): the name of the script to run
    """
    assert isinstance(script_name, str), "script_name must be a string"
    assert script_name.endswith(".py"), "script_name must be a path to a Python script"
    subprocess.run(["python3", script_name], check=True)

def main():
    """"
    "Main function to run the scripts on the AI Impact On Jobs and AI Job Market Insights datasets"
    """
    script1 = "scripts/ai_impact_on_jobs.py"
    script2 = "scripts/ai_job_market_insights.py"

    print("\nRunning ai_impact_on_jobs.py on AI Impact On Jobs Dataset...\n")
    run_script(script_name=script1)
    print("\nSleeping for 2 seconds...\n")
    sleep(2)
    print("\nRunning ai_job_market_insights.py on AI Job Market Insights Dataset...\n")
    run_script(script_name=script2)

if __name__ == "__main__":
    main()