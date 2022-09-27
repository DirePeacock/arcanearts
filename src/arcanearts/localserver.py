import os
import subprocess

server_obj = None

ff_path = "/c/Program\\ Files/Mozilla\\ Firefox/firefox.exe"


def start_server():
    return
    notebook_cmd = "nohup jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8082 --NotebookApp.port_retries=0".split(
        " "
    )
    firefox_cmd = [ff_path, "http://localhost:8082/tree", "&"]
    print(f"\nrunning:\n\t{' '.join(notebook_cmd)}\n")
    server_process = subprocess.Popen(
        notebook_cmd, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )
    print(f"\nrunning:\n\t{' '.join(firefox_cmd)}\n")
    server_process = os.system(" ".join(firefox_cmd))
    print("ayy")


def close_server():
    pass
