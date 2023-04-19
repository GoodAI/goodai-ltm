from setuptools import setup, find_packages, find_namespace_packages

packages = find_namespace_packages(include=['goodai_ltm', 'goodai_ltm.*'])

setup(name='goodai-ltm',
      version='0.0.1',
      packages=packages,
      package_data={},
      install_requires=['torch>=1.8.0', 'pytest>=7.0.0', 'numpy>=1.19.0', 'transformers>=4.0.0',
                        'openai>=0.27.0', 'faiss-cpu', 'datasets', 'boto3', 'python-dotenv']
)
