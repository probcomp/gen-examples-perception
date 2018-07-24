First, start the blender server:
```
blender -b HumanKTH.decimated.blend -P blender_depth_server.py
```

To train the deep-learning based proposal, use:
```
julia train.jl
```

Then, generate `importance_sampling.jld`:
```
julia importance_sampling.jld
```
Results are here also int

Then, generate `mcmc.jld`:
```
julia importance_sampling.jld
```

Then, generate renderings of the results:
```
julia 
```

Trained network weights (`inference_network_params.jld`), importance sampling results (`importance_sampling.jld`), and MCMC results (`mcmc.jld`) are also storted in this [S3 bucket](https://s3.console.aws.amazon.com/s3/buckets/probcomp-marcoct-dl-probprog-genlite-20180724).
