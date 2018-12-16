#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0',
                'python-dotenv',
                'requests',
                'sklearn',
                'pandas',
                'xlrd',
                ]


setup(
    author="Michael Dereszynski",
    author_email='mlderes@hotmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Environment :: Console',
    ],
    description="Suggestions for how to dress based on the expected weather during the activity",
    entry_points={
        'console_scripts': [
            'wtw=what_to_wear_outdoors.cli:main',
            'wtw-train=what_to_wear_outdoors.cli:train_models',
            'wtw-demo=what_to_wear_outdoors.cli:demo_mode',
            'wtw-auto=what_to_wear_outdoors.cli:auto_mode',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='what_to_wear_outdoors',
    name='what_to_wear_outdoors',
    packages=find_packages(include=['what_to_wear_outdoors']),
    package_data= {
        'what_to_wear_outdoors': ['data/*','models/*']
    },
    #setup_requires=setup_requirements,
    test_suite='tests',
    #tests_require=test_requirements,
    url='https://github.com/mlderes/what_to_wear_outdoors',
    version='0.1.0',
    zip_safe=False,
)
