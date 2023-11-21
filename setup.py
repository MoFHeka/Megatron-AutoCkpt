#coding=utf-8

from setuptools import setup, find_packages

setup(
    name='megatron_autockpt',  # Package name
    version='1.1',  # Version
    description=
    'A Megatron checkpoint auto-saving patch at the end of each iteration, inspired by Alibaba PAI EasyCkpt for Megatron.',  # Description
    author='Jia He',  # Name of the author or organization
    author_email='mofhejia@163.com',  # Contact email address
    url='https://github.com/MoFHeka/Megatron-AutoCkpt',  # Project home page
    packages=find_packages(),  # Discover all packages automatically
    include_package_data=True,  # Contain all data files
    zip_safe=False,  # Make it install easily by git
    install_requires=[  # List dependent third-party packages
        'torch',
    ],
    classifiers=[
        # How mature is this project? Common values are
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        'Development Status :: 5 - Production/Stable',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache License :: 2',
        # Supported python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
)
