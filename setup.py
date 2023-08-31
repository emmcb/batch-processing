from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='batch-processing',
      version='1.0',
      description='A Python library to make batchable Python scripts.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Emmanuel Chaboud',
      url='https://github.com/emmcb/batch-processing',
      classifiers=[
          'License :: OSI Approved :: Apache Software License', 'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only', 'Operating System :: OS Independent',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      python_requires='>=3',
      install_requires=['mergedeep', 'tqdm', 'unified-path'])
