import os
import signal
import platform
import subprocess

def kill_processes_on_ports(ports):
    """
    Kills processes running on the specified list of ports.

    Args:
        ports (list of int or str): A list of port numbers to check and kill processes on.
    """
    system = platform.system()
    ports = [str(p) for p in ports] # Ensure ports are strings

    print(f"--- Attempting to kill processes on ports: {', '.join(ports)} ---")

    if system in ["Linux", "Darwin"]: # Linux or macOS
        # Use 'lsof -i :<port>' to find the process PID
        for port in ports:
            try:
                # lsof returns lines like: 'COMMAND PID USER ... TCP *:<port> (LISTEN)'
                # We need the PID, which is the second column.
                command = f"lsof -i :{port} -t"
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5 # Set a timeout for safety
                )

                pids = [pid.strip() for pid in result.stdout.split() if pid.strip()]

                if pids:
                    print(f"Found processes on port {port} with PIDs: {', '.join(pids)}")
                    for pid in pids:
                        try:
                            # Use signal.SIGKILL to forcefully terminate the process
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"  ✅ Successfully killed PID {pid} on port {port}.")
                        except Exception as e:
                            print(f"  ❌ Failed to kill PID {pid} on port {port}: {e}")
                else:
                    print(f"  No process found running on port {port}.")

            except subprocess.CalledProcessError as e:
                # lsof returns non-zero if no process is found, which is fine
                if "no processes found" in e.stderr.lower():
                    print(f"  No process found running on port {port}.")
                else:
                    print(f"  Error running lsof for port {port}: {e.stderr.strip()}")
            except Exception as e:
                print(f"  An unexpected error occurred for port {port}: {e}")


    elif system == "Windows": # Windows
        # Use 'netstat -ano | findstr :<port>' to get the PID, then 'taskkill'
        for port in ports:
            try:
                # 1. Find the PID using netstat
                command = f"netstat -ano | findstr LISTENING | findstr :{port}"
                result = subprocess.run(
                    command,
                    shell=True,
                    check=False, # Don't raise an error if netstat finds nothing
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                pids = []
                if result.stdout:
                    # Output is like: 'TCP 127.0.0.1:5000 0.0.0.0:0 LISTENING 1234'
                    # The PID is the last column.
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split()
                        if parts and len(parts) > 4:
                            pids.append(parts[-1].strip())
                    
                    # Remove duplicates and ensure all found parts are valid PIDs
                    pids = sorted(list(set([p for p in pids if p.isdigit()])))

                if pids:
                    print(f"Found processes on port {port} with PIDs: {', '.join(pids)}")
                    for pid in pids:
                        # 2. Kill the process using taskkill
                        kill_command = f"taskkill /PID {pid} /F"
                        kill_result = subprocess.run(
                            kill_command,
                            shell=True,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if kill_result.returncode == 0:
                            print(f"  ✅ Successfully killed PID {pid} on port {port}.")
                        else:
                            # A common error is "The process with PID XXXX could not be terminated."
                            # This usually means it's a protected system process or the user lacks permission.
                            print(f"  ❌ Failed to kill PID {pid} on port {port}. Output: {kill_result.stderr.strip()}")
                else:
                    print(f"  No process found running on port {port}.")

            except Exception as e:
                print(f"  An unexpected error occurred for port {port}: {e}")

    else:
        print(f"Unsupported operating system: {system}. This function only supports Linux, macOS, and Windows.")