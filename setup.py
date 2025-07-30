from setuptools import setup, find_packages

setup(
    name="vllm-generator",
    version="0.1.0",
    description="A scalable data generation pipeline using vLLM models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "pyarrow>=10.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "httpx>=0.24.0",
        "loguru>=0.7.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "tenacity>=8.2.0",
        "tqdm>=4.65.0",
        "aiofiles>=23.0.0",
        "asyncio>=3.4.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vllm-generator=vllm_generator.__main__:main",
        ],
    },
)