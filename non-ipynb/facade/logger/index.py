import requests
from datetime import datetime
from pathlib import Path

"""
Example log:
[2023-10-01T00:00][INFO] Some message

Put in @see log_path
File name is current time session with format of [Implementation - Session Ymd H:i]
"""
def write_log(log: str, type: str, namespace: str):
    operation = "x"
    time = datetime.now()
    log_location = log_path + "/" + time.strftime("%Y%m%d") + ".adv_mse_perceptual.log"

    if(Path(log_location).is_file()):
        operation = "a"

    fopen = open(log_location, operation)

    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{type}] [{namespace.upper()}] {log}\n"

    fopen.write(message)
    fopen.close()