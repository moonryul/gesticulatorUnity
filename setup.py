from setuptools import setup

#The presence of either setup.py or pyproject.toml file in the current directory signals
# to pip that the directory is a package folder.

setup(name='gesticulator',
      version='0.1',
      description='Gesture generation',
      url='https://github.com/Svito-zar/gesticulator',
      author='Taras Kucherenko',
      author_email='tarask@kth.se',
      license='MIT',
      packages=['gesticulator'],
      zip_safe=False)
