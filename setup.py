from setuptools import setup

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            entry_points={
                "console_scripts": [
                    "python-sprint-zero=python_sprint_zero.interface.api.main:run",
                    "python-sprint-zero-cli=python_sprint_zero.interface.cli.main:run",
                ]
            },
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
