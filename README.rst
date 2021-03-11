XML/TRXML Selector
==================

Description
-----------

A lighter version of the gensim word2vec model.

Status
------------

.. image:: https://travis-ci.org/tilaboy/xml-miner.svg?branch=master
    :target: https://travis-ci.org/tilaboy/xml-miner

.. image:: https://readthedocs.org/projects/xml-miner/badge/?version=latest
    :target: https://xml-miner.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://pyup.io/repos/github/tilaboy/xml-miner/shield.svg
    :target: https://pyup.io/repos/github/tilaboy/xml-miner/
    :alt: Updates

Requirements
------------

Python 3.6+

Installation
------------

::

    pip install easy-word2vec


Usage
-----

Use easy-word2vec model
~~~~~~~~~~~~~~~~~~~~~~~

The easy-word2vec supports:
^^^^^^^^^^^^^^^^^^^^^^^^^^

-  train the word2vec model

-  load and check similar words


examples:
^^^^^^^^^

::

    #train word2vec from corpus
    easy-word2vec --train


Development
-----------

To install package and its dependencies, run the following from project
root directory:

::

    python setup.py install

To work the code and develop the package, run the following from project
root directory:

::

    python setup.py develop

To run unit tests, execute the following from the project root
directory:

::

    python setup.py test
