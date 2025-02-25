import subprocess
import sys
import time


def run_script(script_name):
    """Run a Python script and handle its execution."""
    print(f"\nExecuting {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"{script_name} completed successfully.")
        else:
            print(f"{script_name} failed with return code {result.returncode}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)


def main():
    # List of scripts to run in order
    scripts = [
        "run_preparation.py",
        "run_training.py"
    ]

    start_time = time.time()

    # Run each script sequentially
    for script in scripts:
        run_script(script)

    end_time = time.time()
    print(f"\nAll scripts completed successfully!")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
