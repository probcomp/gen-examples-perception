Start the blender server:
```
blender -b HumanKTH.decimated.blend -P blender_depth_server.py
```

Simulate some images from the model:
```
julia model.jl
```
