import os
from setuptools import setup, find_packages

NAME = "easy_word2vec"
VERSION = os.environ.get("EASY_WORD2VEC_VERSION", '0.0.0')

with open('README.rst', "r") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', "r") as history_file:
    history = history_file.read()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name=NAME,
    version=VERSION,
    keywords='word2vec, word embeddings, word vectors',
    url='https://github.com/tilaboy/easy-word2vec',
    description="a light version of word2vec training package in python",
    author="Chao Li",
    author_email="chaoli.job@gmail.com",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    long_description=readme + '\n\n' + history,
    test_suite="tests",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "easy-word2vec=easy_word2vec.easy_word2vec:main"
        ],
    },
    license="MIT license",
    zip_safe=False
)
