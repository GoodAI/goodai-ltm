from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(include=['goodai.*'])
exec(open('goodai/ltm/version.py').read())
version = globals()['__version__']
print(f'Version: {version}')


setup(name='goodai-ltm',
      description='A text memory meant to be used with conversational language models.',
      url='https://github.com/GoodAI/goodai-ltm',
      version=version,
      packages=packages,
      package_data={'goodai.ltm.data': ['**/*.json']},
      install_requires=['torch>=1.8.0', 'pytest>=7.0.0', 'numpy>=1.19.0', 'transformers>=4.0.0',
                        'openai>=0.27.0', 'faiss-cpu', 'datasets', 'boto3', 'python-dotenv',
                        'sentence-transformers>=2.2.2']
)
