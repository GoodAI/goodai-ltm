import hashlib
import os
import tempfile
import urllib.request
import zipfile
import urllib.request
from urllib.parse import urlparse
from tqdm import tqdm

_cache_dir = os.environ.get('GOODAI_CACHE', os.path.expanduser('~/.cache/goodai'))


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


def open_url_as_file(url):
    # Create the cache directory if it does not exist
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)

    # Convert the URL to a valid file name
    url_parts = urlparse(url)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    file_name = os.path.join(_cache_dir, url_hash + '_' + os.path.basename(url_parts.path))
    progress_fn = os.path.join(file_name, '-in-progress')

    if not os.path.exists(file_name) or os.path.exists(progress_fn):
        with urllib.request.urlopen(url) as url_file:
            with open(progress_fn, 'w') as fd:
                fd.write('.')
            expected_size = url_file.getheader('Content-Length')
            if expected_size is not None:
                expected_size = int(expected_size)
            progress_bar = tqdm(total=expected_size, unit='B', unit_scale=True, desc='Download', leave=True)
            with open(file_name, 'wb') as local_file:
                while True:
                    chunk = url_file.read(100000)
                    if not chunk:
                        break
                    local_file.write(chunk)
                    progress_bar.update(len(chunk))
            os.remove(progress_fn)
    return open(file_name, 'rb')
