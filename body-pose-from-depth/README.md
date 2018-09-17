## blender

Download v2.79b from https://www.blender.org/download/

Create an environment in which the search Python path will include an installation of the `rpyc` Python module, using `PYTHONPATH` or something else.
For example:
```
pip3 install --user rpyc
```

Install `xvfb-run`, so that we can run blender headless:
```
sudo apt install xvfb
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
