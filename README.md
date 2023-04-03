# MiDaS
Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer, based on https://github.com/isl-org/MiDaS.


## Dependencies
- [NumPy](https://numpy.org/install)
- [OpenCV-Python](https://github.com/opencv/opencv-python)
- [PyTorch](https://pytorch.org/get-started) 1.13.1
- [VapourSynth](http://www.vapoursynth.com/) R55+


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
