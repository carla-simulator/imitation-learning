Conditional Imitation Learning at CARLA
===============

Repository to store the conditional imitation learning based
AI that runs on carla. The trained model is the one used 
on CARLA paper.

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


Dataset
------

[The dataset can be downloaded here](https://drive.google.com/file/d/1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY/view)

The dataset is stored on HDF5 files.
Each HDF5 file contains 200 data points.
The HDF5 contains two datasets:
'images_center': <br>
The RGB images stored at 200x88 resolution

'targets': <br>
All the controls and measurements collected. 
They are stored on the dataset vector.

1.Steer, float <br>
2.Gas, float <br>
3.Brake, float <br>
4.Hand Brake, boolean <br>
5.Reverse Gear, boolean <br>
6.Steer Noise, float <br>
7.Gas Noise, float <br>
8.Brake Noise, float <br>
9.Position X, float <br>
10.Position Y, float <br>
11.Speed, float <br>
12.Collision Other, float <br>
13.Collision Pedestrian, float <br>
14.Collision Car, float <br>
15. Opposite Lane Inter, float <br>
16. Sidewalk Intersect, float <br>
17.Acceleration X,float <br>
18. Acceleration Y, float <br>
19. Acceleration Z, float <br>
20. Platform time, float <br>
21. Game Time, float <br>
22. Orientation X, float <br>
23. Orientation Y, float <br>
24. Orientation Z, float <br>
25. Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight) <br>
26. Noise, Boolean ( If the noise ( perturbatin) is activated) <br>
27. Camera ( which camera was used) <br>




Paper
-----

If you use the conditional imitation learning, please cite our Arxiv paper.

End-to-end driving via conditional imitation learning, <br>
Codevilla, Felipe and Müller, Matthias and Dosovitskiy, Alexey and López, Antonio and Koltun, Vladlen;  <br> arXiv preprint arXiv:1710.02410
[[PDF](http://vladlen.info/papers/conditional-imitation.pdf)]


```
@article{codevilla2017end,
  title={End-to-end driving via conditional imitation learning},
  author={Codevilla, Felipe and M{\"u}ller, Matthias and Dosovitskiy, Alexey and L{\'o}pez, Antonio and Koltun, Vladlen},
  journal={arXiv preprint arXiv:1710.02410},
  year={2017}
}

