import subprocess
import time
from datetime import datetime
from pathlib import Path

RPI_USER = "pi"
RPI_HOST = "10.32.92.205"   # replace if needed
RPI_DATA_DIR = "~/your_project/data"

today = datetime.utcnow().strftime("%Y-%m-%d")
LOCAL_DIR = Path(f"data/{today}")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL_SEC = 60

def run_rsync():
    source = f"{RPI_USER}@{RPI_HOST}:{RPI_DATA_DIR}/{today}/"
    dest = str(LOCAL_DIR) + "/"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "--ignore-existing",
        source,
        dest,
    ]

    print(f"[sync] Pulling from {source} -> {dest}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            fits_files = list(LOCAL_DIR.glob("*.fits"))
            log_exists = (LOCAL_DIR / "observation_log.csv").exists()
            print(f"[sync] Sync complete. {len(fits_files)} FITS file(s) locally.")
            print(f"[sync] Log present: {log_exists}")

            if result.stdout.strip():
                for line in result.stdout.splitlines():
                    if ".fits" in line or "observation_log.csv" in line:
                        print(f"[sync] Transferred: {line.strip()}")
        else:
            print(f"[sync] rsync error:\n{result.stderr}")

    except FileNotFoundError:
        print("[sync] rsync not found.")

def main():
    print("[sync] Starting data sync watcher.")
    print(f"[sync] Source: {RPI_USER}@{RPI_HOST}:{RPI_DATA_DIR}/{today}/")
    print(f"[sync] Dest:   {LOCAL_DIR.resolve()}")
    print(f"[sync] Every {SYNC_INTERVAL_SEC}s\n")

    while True:
        run_rsync()
        time.sleep(SYNC_INTERVAL_SEC)

if __name__ == "__main__":
    main()
