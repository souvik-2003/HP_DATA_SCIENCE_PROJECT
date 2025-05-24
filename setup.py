"""
Setup script for the project.
"""
from setuptools import setup, find_packages

setup(
    name="project_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "streamlit",
        "openpyxl",  # For Excel support
        "pytest",    # For testing
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A data science project with Streamlit interface",
    keywords="data science, machine learning, streamlit",
    python_requires=">=3.8",
)
