import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
"""
list_of_files = [
    "data/f",
    "output/plots",
    "notebooks/eda.ipynb",
    "notebooks/colab.ipynb",
    "src/__init__.py",
    "src/data_cleaning.py",
    "src/prepare_data.py",
    "src/modeling.py",
    "src/utils.py",
    "main.py",
    "requirements.txt"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    # Handle files
    if filename:  # If it's a file
        if os.path.exists(filepath):
            os.remove(filepath)  # Delete the file if it exists
            logging.info(f"Deleted existing file: {filepath}")

        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:  # If it's a folder
        os.makedirs(filepath, exist_ok=True)
        logging.info(f"Ensuring folder exists: {filepath}")"""