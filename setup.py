from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memvid",
    version="0.1.3",
    author="Saleban Olow",
    author_email="olow30@gmail.com",
    description="Video-based AI memory library for fast semantic search and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olow304/memvid",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "qrcode[pil]>=7.3",
        "opencv-python>=4.5.0",
        "opencv-contrib-python>=4.5.0",  # Includes QR decoder
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0,<2.0.0",
        "tqdm>=4.50.0",
        "faiss-cpu>=1.7.0",
        "Pillow>=9.0.0",
        "python-dotenv>=0.19.0",
        "PyPDF2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "google-genai>=1.39.1",
            "anthropic>=0.52.0",
        ],
        "epub": [
            "beautifulsoup4>=4.0.0",
            "ebooklib>=0.18",
        ],
        "web": [
            "fastapi>=0.100.0",
            "gradio>=4.0.0",
        ],
    },
)