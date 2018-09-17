## AWS GPU Instance

Tested with a p2.xlarge EC2 instance, starting with the Deep Learning AMI (Ubuntu) Version 14.0 (ami-0466e26ccc0e752c1).
We use this AMI because it contains a working install of CUDA 9, which is needed by TensorFlow.jl

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
```

Install `xvfb-run`, so that we can run blender headless:
```
sudo apt install xvfb
```

## Julia

Get julia-0.7.0:
```
wget https://julialang-s3.julialang.org/bin/linux/x64/0.7/julia-0.7.0-linux-x86_64.tar.gz
tar -xzf https://julialang-s3.julialang.org/bin/linux/x64/0.7/julia-0.7.0-linux-x86_64.tar.gz
export PATH=/home/ubuntu/julia-0.7.0/bin:$PATH
```

## Julia packages

To install necessary packages in the default Julia environment:

Note that `/home/marcoct/dev/` directory needs to be changed:
```
git clone git@github.com:probcomp/Gen.git
julia -e 'using Pkg; Pkg.develop(PackageSpec(path="/home/marcoct/dev/Gen"))'
git clone git@github.com:probcomp/GenTF.git
julia -e 'using Pkg; Pkg.develop(PackageSpec(path="/home/marcoct/dev/GenTF"))'
```

Ensure you are using a self-contained version of Python (this builds PyCall as well as TensorFlow):
```
julia -e 'ENV["PYTHON"] = ""; Pkg.build("GenTF")'
```

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

Also, need to install the Python module `rpyc` for use by the Python used by PyCall.
To do this manually in Julia:
```julia
import PyCall
python_path = PyCall.python
dir = join(split(python_path, "/")[1:end-1], "/")
run(`$dir/pip install rpyc`)
```

TODO: We should use a [Project](https://docs.julialang.org/en/v1/stdlib/Pkg/) to manage these dependencies.
But, that is harder when Gen and GenTF are not in a registry and are not public repositories.

## Testing

Run the `visualize.jl` script to test:
```
julia visualize.jl
```

It should produce some `.png` files showing ground truth, observed image, and approximate posterior samples.
