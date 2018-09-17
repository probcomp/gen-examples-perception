# blender

Download v2.79b from https://www.blender.org/download/

Create an environment in which the search Python path will include an installation of the `rpyc` Python module, using `PYTHONPATH` or something else.
For example:
```
pip3 install --user rpyc
```

Install `xvfb-run`, so that we can run blender headless:
```
sudo aptinstall xvfb
```
