# dl-probprog-genlite
Extended abstract on deep learning for inference in GenLite, highlighting combination of custom Monte Carlo and custom deep learning based proposals.

## Python environment

We first set up a python environment that contains a specific version of TensorFlow installed.
Create a python3 virtual environment, and install TensorFlow and matplotlib:
```
virtualenv -p python3 my-env
source my-env/bin/activate
pip3 install tensorflow==1.8.0
pip3 install matplotlib
```

## Julia environment

Install the TensorFlow.jl package, and pin it to a specific version.
```
Pkg.add("TensorFlow")
Pkg.pin("TensorFlow", v"0.9.1")
```
The version of TensorFlow python library and the TensorFlow.jl Julia package need to be synchronized.
Given a version of TensorFlow.jl, the version of python TensorFlow to install can be found in the file [deps/build.jl](https://github.com/malmaud/TensorFlow.jl/blob/master/deps/build.jl) in the TensorFlow.jl source code.

Install the GenLite and GenLiteTF packages.
These are not publicly registered packages.
One way to install them is to clone the repositories locally (e.g. to `~/dev/Genlite.jl` and `~/dev/GenLiteTF.jl`) and then create symbolic links from your Julia packages directory to the cloned repositories, e.g.:
```
ln -s ~/dev/GenLite.jl ~/.julia/v0.6/GenLite
ln -s ~/dev/GenLiteTF.jl ~/.julia/v0.6/GenLiteTF
```

## Blender

Download [blender](https://www.blender.org/download/).
Experiments used blender version 2.79b.
Ensure that the `blender` executable is on your `PATH`.
