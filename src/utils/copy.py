r"""Utils for copying data to the temporary directory of the compute node
"""

import logging
import os
from pathlib import Path

from filelock import FileLock


log = logging.getLogger(__name__)


def copy_data(origin: Path, destination: Path):
    cmd = "cp -r {origin} {destination}"
    os.system(cmd.format(origin=origin, destination=destination))

def copy_to_compute_node(data_path: str | Path,
                         local_dir: str | Path = "/local/job"):
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    
    if not isinstance(local_dir, Path):
        local_dir = Path(local_dir)
    
    # If we are not in a SLURM job, this function is a no-op
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return data_path
    
    # First, we check if the data we're looking for is already in the compute node
    origin_folder = None

    for folder in local_dir.iterdir():
        if not os.access(folder, os.R_OK):
            continue

        subfolder = folder / data_path.name
        if subfolder.exists():
            origin_folder = subfolder
            break

    # Set the destination folder
    dest_folder = local_dir / job_id / data_path.name
    if dest_folder.exists():
        log.info(f"Data found in {dest_folder}.")
        return dest_folder
    
    dest_folder.mkdir()

    if origin_folder is None:
        # In that case, we will copy from the login node directly (slow...)
        origin_folder = data_path
    
    else:
        # Wait for the origin folder to have been fully filled to copy from the compute node (much faster!)
        log.info(f"Data found in {origin_folder}. Waiting for the lock to be released...")
        with FileLock(origin_folder / "lock"):
            pass
    
    # Lock the destination folder and copy the data inside it
    log.info(f"Copying data from {origin_folder} to {dest_folder}...")
    with FileLock(dest_folder / "lock"):
        copy_data(origin_folder, dest_folder)
    
    log.info("Done.")
    return dest_folder