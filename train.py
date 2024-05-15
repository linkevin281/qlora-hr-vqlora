import subprocess

def train():
    # Define the path to the script
    script_path = f'scripts/tiny.sh'
    output_file_path = './output/out.txt'

    try:
        # Open the output file
        with open(output_file_path, 'w') as file:
            # Start the subprocess and redirect the output to the file
            with subprocess.Popen(script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                # Read from the process output and write to the file line by line
                for line in proc.stdout:
                    print(line, end='')  # Optionally print to the console in real-time
                    file.write(line)
                    file.flush()  # Ensure each line is flushed to the file

            # Check for process completion and if it ended with an error
            if proc.returncode != 0:
                print("Script finished with an error.")
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    train()
