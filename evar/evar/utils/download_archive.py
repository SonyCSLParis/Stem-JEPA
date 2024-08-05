import os
import tempfile
import zipfile

import requests


def download_and_extract_zip_archive(url: str, tempdir: str):
    # Ensure tempdir exists or create it
    os.makedirs(tempdir, exist_ok=True)

    # Download the archive
    response = requests.get(url)
    response.raise_for_status()

    # Write the downloaded content to a temporary file
    with tempfile.NamedTemporaryFile(dir=tempdir) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

        # Extract the archive
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(tempdir)

    # Return the directory where the contents are extracted
    return tempdir
