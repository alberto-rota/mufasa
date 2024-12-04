from setuptools import setup, find_packages

setup(
    name="mufasa-polimi",
    version="0.0.5",
    packages=find_packages(),
    install_requires=[],  # Add dependencies listed in requirements.txt
    # entry_points={
    #     "console_scripts": [
    #         "groundcontrol = ground_control_tui.monitor:main",  # Change main to your entry function
    #     ],
    # },
    description="Utilities and Helpers for optimal usage of the MUFASA HPC cluster at Politecnico di Milano",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alberto-rota/mufasa",
    author="Alberto Rota",
    author_email="alberto1.rota@polimi.it",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
