from pathlib import Path
from setuptools import setup, find_packages
import versioneer

# Use readme file as long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text()

setup(
    name="Brian2CUDA",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_data={
        # include template files
        "brian2cuda": [
            "templates/*.cu",
            "templates/*.h",
            "templates/makefile",
            "templates/win_makefile",
            "brianlib/*.cu",
            "brianlib/*.h",
        ],
        # Include external code for tests
        "brian2cuda.tests": [
            "func_def_cuda.cu",
            "func_def_cuda.h",
        ],
    },
    install_requires=["brian2==2.4.2"],
    provides=["brian2cuda"],
    extras_require={
        "test": ["pytest", "pytest-xdist>=1.22.3"],
        "docs": ["sphinx>=1.8"],
    },
    url="http://github.com/brian-team/brian2cuda",
    project_urls={
        "Documentation": "https://brian2cuda.readthedocs.io/en/latest/",
        "Bug tracker": "https://github.com/brian-team/brian2cuda/issues",
    },
    description="A Brian2 extention to simulate spiking neural networks on GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Denis Alevi, Moritz Augustin and Marcel Stimberg",
    author_email="mail@denisalevi.de",
    keywords="computational neuroscience simulation",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.6",
)
