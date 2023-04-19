import hashlib
import os
import tempfile
import urllib.request
import zipfile


def download_zip(url):
    hash_str = hashlib.sha256(url.encode('utf-8')).hexdigest()
    temp_dir = os.path.join(tempfile.gettempdir(), hash_str)

    if not os.path.exists(temp_dir):
        print(f'Downloading {url}')
        # Download the zip file
        zip_file, _ = urllib.request.urlretrieve(url)

        # Unzip the file into the temporary directory
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Clean up the downloaded zip file
        os.remove(zip_file)
    else:
        print(f'File {url} already downloaded.')
    return temp_dir

