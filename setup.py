from setuptools import setup

setup(
    name="SAGESim",
    version="0.1.0",
    author="Chathika Gunaratne",
    author_email="gunaratnecs@ornl.gov",
    packages=["sagesim"],
    include_package_data=True,
    url="https://code.ornl.gov/sagesim/sagesim",
    license="GPL",
    description="Scalable Agent-based GPU Enabled Simulator.",
    long_description="""A GPU-based general purpose multi-agent simulation framework.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://code.ornl.gov/sagesim/sagesim"},
    install_requires=[
        "cupy",
        "numpy==1.21.6",
        "tqdm==4.64.1",
        "distributed==2021.7.1",
    ],
)
