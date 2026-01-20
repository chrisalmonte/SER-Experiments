import os
from datetime import datetime

class Directories:
    @staticmethod
    def make_unique(path: str):
        if os.path.exists(path):
            path = os.path.join(path, f"run_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}")
            print("The directory already exists. A new directory has been created:", path)
        try:
            os.makedirs(path, exist_ok=True)
            os.mkdir(os.path.join(path, "checkpoints"))
            print("Directory created at:", path)
        except Exception as e:
            raise OSError(f"The directory could not be created: {e}")
        return path
            