from setuptools import setup, find_packages


REQUIREMENTS = [
    "pandas",
    "opencv-python",
    "fire",
    "pillow",
    "huggingface-hub",
    "tqdm",
    "pydantic",
    "toml",
    "pyyaml",
] + [
    "numpy",
    "torch",
    "tensorflow==2.10.1",
    "transformers",
]

setup(
    name="nazrin",
    packages=find_packages(),
    version="0.0.1",
    url="https://github.com/ddPn08/nazrin",
    description="",
    author="ddPn08",
    author_email="contact@ddpn.world",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "nazrin=nazrin.main:cli",
        ]
    },
    install_requires=REQUIREMENTS,
)
