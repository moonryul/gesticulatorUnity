# Always prefer setuptools over distutils

# setup.py:   Python's answer to a multi-platform installer and make file.https://stackoverflow.com/questions/1471994/what-is-setup-py#:~:text=setup.py%20is%20a%20python,to%20easily%20install%20Python%20packages.:
# setup.py is a python file, the presence of which is an indication that the module/package 
# you are about to install has likely been packaged and distributed with Distutils, 
# which is the standard for distributing Python Modules.

#This allows you to easily install Python packages. Often it's enough to write:

# $ pip install . : Avoid calling setup.py directly.

from setuptools import setup, find_packages

setup(
    name="motion_visualizer", # https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
# The name of the package, which is the name that pip will use for your package.
# This does not have to be the same as the folder name the package lives
# in, although it may be confusing if it is not. An example of where the package
# name and the directory do not match is Scikit-Learn: you install it
# using pip install scikit-learn, while you use it by importing from sklearn.
    # 
    version="0.0.1",
    packages=["motion_visualizer", "pymo"], #  # list folders, not files
    #   packages=find_packages(where='src'),
    #   package_dir={'': 'src'},
    install_requires=[
        "matplotlib",
        "scipy",
        "pyquaternion",
        "pandas",
        "sklearn",
        "transforms3d",
        "bvh",
    ],
    package_data={"motion_visualizer": ["data/data_pipe.sav"]},
    package_dir={"motion_visualizer": "motion_visualizer"}, # relative to the folder where setup.py is located
)
