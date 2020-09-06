import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="surgo-bayesian-network",
    version="0.0.1",
    author="Roni Kobrosly",
    author_email="roni.kobrosly@gmail.com",
    description="Case study package for Surgo Foundation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ronikobrosly/surgo_bayesian_network",
    packages=setuptools.find_packages(include=['surgo_bayesian_network']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'pomegranate',
        'pygraphviz',
        'pytest'
    ]
)
