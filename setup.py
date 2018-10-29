import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cricpy",
    version="0.0.9",
    author="Tinniam V Ganesh",
    author_email="tvganesh.85@gmail.com",
    description="Analyze Cricketers Based on ESPN Cricinfo Statsguru",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tvganesh/cricpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
