## AWS GPU Instance

Tested with a p2.xlarge and p2.8xlarge EC2 instances, starting from the Deep Learning AMI (Ubuntu) Version 14.0 (ami-0466e26ccc0e752c1).
Started from this AMI because it contains a working install of CUDA 9, which is needed by TensorFlow.jl

## blender

Download v2.79b of [blender](https://www.blender.org/download/), e.g.:
```
wget https://builder.blender.org/download/blender-2.79-5c10c92b23c-linux-glibc224-x86_64.tar.bz2
tar -xjf blender-2.79-5c10c92b23c-linux-glibc224-x86_64.tar.bz2
export PATH=/home/ubuntu/blender-2.79-5c10c92b23c-linux-glibc224-x86_64:$PATH
```

Create an environment in which the search Python path will include an installation of the `rpyc` Python module, using `PYTHONPATH` or something else.
For example:
```
pip3 install --user rpyc
export PYTHONPATH=/home/ubuntu/.local/lib/python3.5/site-packages:$PYTHONPATH
```

Install `xvfb-run`, so that we can run blender headless:
```
sudo apt install xvfb
```

## Julia
Install Git if not already installed:
```
sudo apt install git
```

Get julia-0.7.0:
```
wget https://julialang-s3.julialang.org/bin/linux/x64/0.7/julia-0.7.0-linux-x86_64.tar.gz
tar -xzf https://julialang-s3.julialang.org/bin/linux/x64/0.7/julia-0.7.0-linux-x86_64.tar.gz
export PATH=/home/ubuntu/julia-0.7.0/bin:$PATH
```

## Julia packages

To install necessary packages in the default Julia environment:
```
git clone git@github.com:probcomp/Gen.git
julia -e 'using Pkg; Pkg.develop(PackageSpec(path="/home/ubuntu/Gen"))'
git clone git@github.com:probcomp/GenTF.git
julia -e 'using Pkg; ENV["PYTHON"] = ""; ENV["TF_USE_GPU"] = 1; Pkg.develop(PackageSpec(path="/home/ubuntu/GenTF"))'
```

Ensure you are using a self-contained version of Python (this builds PyCall as well as TensorFlow).
Compile to use GPU support for TensorFlow:
```
julia -e 'ENV["PYTHON"] = ""; ENV["TF_USE_GPU"] = 1; Pkg.build("GenTF")'
```

Install other packages which are in the public registry:
```
julia -e 'Pkg.add("FileIO")'
julia -e 'Pkg.add("Images")'
julia -e 'Pkg.add("ImageFiltering")'
julia -e 'Pkg.add("DataFrames")'
julia -e 'Pkg.add("CSV")'
julia -e 'Pkg.add("PyCall")'
julia -e 'Pkg.add("ImageMagick")'
julia -e 'Pkg.add("JLD2")'
julia -e 'Pkg.add("ReverseDiff")'
julia -e 'Pkg.add("TensorFlow")'
julia -e 'Pkg.add("MacroTools")'
```

Also, need to install the Python module `rpyc` for use by the Python used by PyCall:
```
julia -e 'import PyCall; dir=join(split(PyCall.python, "/")[1:end-1], "/"); run(`$dir/pip install rpyc`)'
```

TODO: We should use a [Project](https://docs.julialang.org/en/v1/stdlib/Pkg/) to manage these dependencies.
But, that is harder when Gen and GenTF are not in a registry and are not public repositories.

## Testing

Clone this repository onto the machine (use ssh agent forwarding).

Run the `visualize.jl` script to test:
```
julia visualize.jl
```

It should produce some `.png` files showing ground truth, observed image, and approximate posterior samples.

## Training deep net proposals

Can run them in parallel, but need to make sure they are using separate GPUs (because [multiple TensorFlow processes cannot reliably share GPUs](https://github.com/tensorflow/tensorflow/issues/4196) verified by my own experience).
```
CUDA_VISIBLE_DEVICES=0 julia train_neural_proposal_tiny.jl &
CUDA_VISIBLE_DEVICES=1 julia train_neural_proposal_small.jl &
CUDA_VISIBLE_DEVICES=2 julia train_neural_proposal_large.jl &
```

## Blender process and ports

The Julia scripts spawn blender processes and connect to them over sockets.
If running multiple scripts at once you need to make sure that no port number is used more than once.

The Julia scripts don't currently reliably kill them when they return.
Therefore, you will need to clean them up (e.g. with kill or pkill).
