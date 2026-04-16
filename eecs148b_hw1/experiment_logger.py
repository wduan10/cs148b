import json
import time
from pathlib import Path


class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log(self, **kwargs):
        record = dict(kwargs)
        record["wallclock_time"] = time.time() - self.start_time
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
