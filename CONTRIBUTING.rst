============
Contributing
============

Welcome to ``python-sprint-zero`` contributor's guide. This project embraces clean architecture, Hexagonal Architecture, and Domain-Driven Design principles, so contributions should align with its modular, testable, and secure structure.

If you're new to Git or contributing to open source projects, check out the `FreeCodeCamp contribution guide`_ and `contribution-guide.org`_.

All contributors are expected to act in accordance with the `Python Software Foundation's Code of Conduct`_.

Issue Reports
=============

Found a bug or have an idea? Please search the `issue tracker`_ first, including closed issues. If none match, feel free to open a new one.

Be sure to include:

- OS and Python version
- Steps to reproduce
- A minimal working example, if possible

Documentation Improvements
==========================

Docs are written in reStructuredText_ using Sphinx_. To improve them, submit changes via GitHub’s web editor or in a local dev environment.

Quick preview via:

.. code:: bash

    tox -e docs
    python3 -m http.server --directory 'docs/_build/html'

Code Contributions
==================

Our architecture separates the system into domain, application, infrastructure, and interface layers. Follow this structure when contributing.

Setup
-----

1. Fork the `repository`_ and clone it locally.

.. code:: bash

    git clone git@github.com:svo/python-sprint-zero.git
    cd python-sprint-zero
    pip install -U pip setuptools -e .

2. Create a virtual environment:

.. code:: bash

    python -m venv venv
    source venv/bin/activate

3. Install development tools:

.. code:: bash

    pip install pre-commit tox
    pre-commit install

Start Coding
------------

1. Create a new branch:

.. code:: bash

    git checkout -b my-feature

2. Implement changes. Structure your code within the appropriate layer (e.g., ``domain/model``, ``application/use_case``).

3. Add docstrings and meaningful tests.

4. Add yourself to ``AUTHORS.rst``.

Testing
-------

Run all checks via:

.. code:: bash

    tox

We require 100% test coverage and use:

- flake8
- black
- bandit
- xenon (grade A)
- mypy
- safety & dependency-check
- semgrep

Push and PR
-----------

1. Push your branch:

.. code:: bash

    git push -u origin my-feature

2. Open a Pull Request via GitHub.

Release Process (Maintainers Only)
==================================

1. Tag the release:

.. code:: bash

    git tag vX.Y.Z
    git push upstream vX.Y.Z

2. Clean old builds:

.. code:: bash

    tox -e clean

3. Build and publish:

.. code:: bash

    tox -e build
    tox -e publish -- --repository pypi

Resources
==================================

.. _FreeCodeCamp contribution guide: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _issue tracker: https://github.com/svo/python-sprint-zero/issues
.. _repository: https://github.com/svo/python-sprint-zero
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/

