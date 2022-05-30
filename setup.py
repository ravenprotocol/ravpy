from setuptools import setup, find_packages

setup(
    name="ravpy",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.5",
        "pandas==1.3.5",
        "pyftpdlib==1.5.6",
        "python-engineio==4.2.1",
        "python-socketio==5.4.1",
        "requests==2.27.1",
        "tenseal==0.3.6",
        "python-dotenv",
        "scipy"
    ],
    dependency_lins=["git+https://github.com/ravenprotocol/ravop.git"]
)
