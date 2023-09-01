import csv
from pathlib import Path

import setuptools

with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

install_requires = []
with open("requirements.txt", "r", encoding='UTF-8') as fh:
    reader = csv.reader(fh)
    for row in reader:
        install_requires.append(row[0])


def get_scm_files():
    """
    Returns a list of all files in the SCM directory and its subdirectories.
    """
    scm_dir = Path('insar_process/SCM')
    scm_files = []
    for file_path in scm_dir.glob('**/*'):
        if file_path.is_file():
            scm_files.append('/'.join(list(file_path.parts[1:])))
    return scm_files


setuptools.setup(
    name="INSAR_PROCESS",
    version="0.0.1",
    author="Fancy",
    author_email="fanchy14@lzu.edu.cn",
    description="Scripts for processing InSAR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fanchengyan/InSAR_Process",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    package_data={'insar_process': get_scm_files()},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
