import codecs
import os

from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(include=['goodai.*'])
exec(open('goodai/ltm/version.py').read())
version = globals()['__version__']
print(f'Version: {version}')


def read_file(filename):
    with codecs.open(os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8') as file:
        return file.read()


setup(name='goodai-ltm',
      description='A text memory meant to be used with conversational language models.',
      long_description=read_file('README-Pypi.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/GoodAI/goodai-ltm',
      version=version,
      packages=packages,
      package_data={'goodai.ltm.data': ['**/*.json']},
      install_requires=['torch>=1.8.0', 'pytest>=7.0.0', 'numpy>=1.19.0', 'transformers>=4.34.0',
                        'openai>=1.0.0', 'faiss-cpu', 'datasets', 'boto3', 'python-dotenv',
                        'sentence-transformers>=2.2.2', 'FlagEmbedding>=1.1', 'tiktoken>=0.5.0',
                        'litellm', 'cohere']
)