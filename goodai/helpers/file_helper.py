import codecs
import hashlib
import logging
import os
import tempfile
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


def open_url_as_file(url, mode='rb'):
    file_name = url_as_file(url)
    return open(file_name, mode=mode)


def codecs_open_url_as_file(url, mode='r', encoding='utf-8'):
    file_name = url_as_file(url)
    return codecs.open(file_name, mode=mode, encoding=encoding)


def url_as_file(url) -> str:
    # Create the cache directory if it does not exist
    if not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir)

    # Convert the URL to a valid file name
    url_parts = urlparse(url)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    base_name = os.path.basename(url_parts.path)
    file_name = os.path.join(_cache_dir, url_hash + '_' + base_name)
    progress_fn = file_name + '-in-progress'

    if not os.path.exists(file_name) or os.path.exists(progress_fn):
        logging.warning(f'Downloading {url}')
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
    else:
        logging.info(f'Found cached version of {url}')
    return file_name
