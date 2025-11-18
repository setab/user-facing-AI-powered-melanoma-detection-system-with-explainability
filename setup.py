"""
Setup script for melanoma detection project.
Allows installation as a package: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="melanoma-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered melanoma detection system with explainability and interactive Q&A",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pillow>=9.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "train": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "seaborn>=0.12.0",
            "opencv-python>=4.7.0",
            "scipy>=1.10.0",
        ],
        "serve": [
            "gradio>=4.0.0",
            "torchcam>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "melanoma-cli=src.inference.cli:main",
            "melanoma-serve=src.serve_gradio:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="melanoma detection deep-learning explainable-ai medical-imaging pytorch",
    project_urls={
        "Documentation": "https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability#readme",
        "Source": "https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability",
        "Bug Reports": "https://github.com/setab/user-facing-AI-powered-melanoma-detection-system-with-explainability/issues",
    },
)
