import subprocess
from time import sleep

def run_script(script_name):
    """
    Function to run a Python script using subprocess module

    Args:
        script_name (str): the name of the script to run
    """
    subprocess.run(["python3", script_name], check=True)

def main():
    print("\nRunning ai_impact_on_jobs.py on AI Impact On Jobs Dataset...\n")
    run_script("ai_impact_on_jobs.py")
    print("\nSleeping for 2 seconds...\n")
    sleep(2)
    print("\nRunning ai_job_market_insights.py on AI Job Market Insights Dataset...\n")
    run_script("ai_job_market_insights.py")

if __name__ == "__main__":
    main()