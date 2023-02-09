from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ravpy",
    version="0.14",
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravpy',
    keywords='Ravpy, python client library for providers',
    install_requires=[
        "numpy==1.21.5",
        "pandas==1.3.5",
        "pyftpdlib==1.5.6",
        "python-engineio==4.2.1",
        "python-socketio==5.4.1",
        "requests==2.27.1",
        "python-dotenv",
        "speedtest-cli",
        "terminaltables==3.1.10",
        "websocket-client",
        "pyinstaller",
        "scikit-learn",
        "psutil",
        "hurry.filesize",
        "sqlalchemy",
        "sqlalchemy-utils",
        'scipy',
        'pillow',
        'tinyaes',
        'torch'
    ],
    app=["gui.py"],
    setup_requires=["py2app"],
)
