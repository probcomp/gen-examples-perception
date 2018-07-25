# How to run experiments

First, start the blender server:
```
blender -b HumanKTH.decimated.blend -P blender_depth_server.py
```
Warning: Make sure you are blocking incoming connections on the given port, the server does not authenticate.

To train the deep-learning based proposal (generate `inference_network_params.jld`), use:
```
julia train.jl
```

Then, generate `importance_sampling.jld`:
```
julia importance_sampling.jl
```

Then, generate `mcmc.jld`:
```
julia mcmc.jl
```

Then, generate renderings of the results (populates `results/` directory with renderings of latent states):
```
julia render_inference_results.jl
```

Trained network weights (`inference_network_params.jld`), importance sampling results (`importance_sampling.jld`), and MCMC results (`mcmc.jld`) are also storted in this [S3 bucket](https://s3.console.aws.amazon.com/s3/buckets/probcomp-marcoct-dl-probprog-genlite-20180724).

# Performance

The bottleneck is the renderer.

Below are the results of profiling `mcmc.jl`. The red columns are all calls to `render_depth.jl`:
![ProfileView of mcmc.jl](images/mcmc-profile.png)
