<img src="SAGESim-inline-tag-color.png" alt="SAGESim: Scalable Agent-based GPU Enabled Simulator" height="200"/>



# Requirements

 - Python 3.7+
 - GPU device with compute capability 3.0+. See https://developer.nvidia.com/cuda-gpus .
 - `conda create -n sagesimenv python=3.9`
 - `conda activate sagesimenv`
 - `conda install -c anaconda cudatoolkit`
 - `pip install -r requirements.txt`
 
# Run Example

 - `git clone https://code.ornl.gov/sagesim/sagesim`
 - `export PYTHONPATH=/path/to/clone_repo`
 - `cd /path/to/clone_repo`
 - `python examples/sir/run.py`
