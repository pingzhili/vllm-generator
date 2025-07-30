from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vllm-generator",
    version="0.1.0",
    author="Your Name",
    author_email="pingzhili86@gmail.com",
    description="Scalable text generation for dataframes using vLLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pingzhili/vllm-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "pyarrow>=5.0.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "typer>=0.4.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "ray": ["ray>=2.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "all": ["vllm>=0.2.0", "ray>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "vllm-generator=vllm_generator.main:main",
            "vllm-gen=vllm_generator.main:main",
        ],
    },
    package_data={
        "vllm_generator": ["configs/*.yaml", "configs/*.json"],
    },
)