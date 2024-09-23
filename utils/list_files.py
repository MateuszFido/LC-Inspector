import os
import re
from pathlib import Path

def read_files(file_path):
    file_path = Path(file_path)
            
    folder_contents = os.listdir(file_path)
    
    cal_files = []   # Create empty placeholder list for calibration files
    res_files = []   # Create empty placeholder list for measurement files
    
    for file in folder_contents:
        if file.endswith(".txt"):
            match = re.search(r'STMIX', file)
            if match:
                cal_files.append(file)
            elif file.endswith("log.txt"):
                continue
            else:
                res_files.append(file)
        else:
            continue

    cal_files = sorted(cal_files)
    res_files = sorted(res_files)
    
    # Create paths and check if already present 
    
    return cal_files, res_files

# TODO: Include MS/LC separation>? 