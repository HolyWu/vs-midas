# MiDaS
Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer, based on https://github.com/isl-org/MiDaS.


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later


## Installation
```
pip install -U vsmidas
python -m vsmidas
```


## Usage
```python
from vsmidas import midas

ret = midas(clip)
```

See `__init__.py` for the description of the parameters.
