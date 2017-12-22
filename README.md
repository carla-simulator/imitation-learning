Conditional Imitation Learning at CARLA
===============

Repository to store the conditional imitation learning based
AI that runs on carla.

Requirements
-------
tensorflow_gpu 1.1 or more

numpy

scipy

carla 0.7.1


Running
------
Basically run:

$ python run_CIL.py

Note that you must have a carla server running  <br>
To check the other options run

$ python run_CIL.py --help

Paper
-----

If you use the conditional imitation learning, please cite our Arxiv paper.

_End-to-end driving via conditional imitation learning, <br>
Codevilla, Felipe and M{\"u}ller, Matthias and Dosovitskiy, Alexey and L{\'o}pez, Antonio and Koltun, Vladlen;  <br> arXiv preprint arXiv:1710.02410
[[PDF](http://vladlen.info/papers/conditional-imitation.pdf)]


```
@article{codevilla2017end,
  title={End-to-end driving via conditional imitation learning},
  author={Codevilla, Felipe and M{\"u}ller, Matthias and Dosovitskiy, Alexey and L{\'o}pez, Antonio and Koltun, Vladlen},
  journal={arXiv preprint arXiv:1710.02410},
  year={2017}
}

