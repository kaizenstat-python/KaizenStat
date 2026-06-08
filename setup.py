from setuptools import setup, find_packages

setup(
    name="kaizenstat",
    version="0.5.1",
    author="Masuddar Rahman",
    url="https://www.kaizenstat.com",
    description="Data Health Measurement and ML Model Debugging Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.7.0",
        "rich>=12.0.0",
        "joblib>=1.1.0",
        "typer>=0.9.0",
    ],
    extras_require={
        "ai":  ["anthropic>=0.20.0"],
        "gpu": ["xgboost", "lightgbm"],
        "fast": ["polars"],
        "nlp": ["sentence-transformers>=2.2.0"],
        "all": ["anthropic>=0.20.0", "xgboost", "lightgbm", "polars",
                "streamlit", "sentence-transformers>=2.2.0"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "kz=kaizenstat.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
