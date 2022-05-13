Conditional Imitation Learning at CARLA
===============

Repository to store the conditional imitation learning based
AI that runs on carla. The trained model is the one used 
on "CARLA: An Open Urban Driving Simulator" paper.

Requirements
-------
tensorflow_gpu 1.1 or more

numpy

scipy

carla 0.8.2

PIL


Running
------
Basically run:

$ python run_CIL.py

Note that you must have a carla server running . <br>
To check the other options run

$ python run_CIL.py --help


Dataset
------

[The dataset can be downloaded here](https://drive.google.com/file/d/1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY/view) 24 GB

The data is stored on HDF5 files.
Each HDF5 file contains 200 data points.
The HDF5 contains two "datasets":
'rgb': <br>
The RGB images stored at 200x88 resolution

'targets': <br>
All the controls and measurements collected. 
They are stored on the "dataset" vector.

1. Steer, float 
2. Gas, float
3. Brake, float 
4. Hand Brake, boolean 
5. Reverse Gear, boolean
6. Steer Noise, float 
7. Gas Noise, float 
8. Brake Noise, float
9. Position X, float 
10. Position Y, float 
11. Speed, float 
12. Collision Other, float 
13. Collision Pedestrian, float 
14. Collision Car, float 
15. Opposite Lane Inter, float 
16. Sidewalk Intersect, float 
17. Acceleration X,float 
18. Acceleration Y, float 
19. Acceleration Z, float 
20. Platform time, float 
21. Game Time, float 
22. Orientation X, float 
23. Orientation Y, float 
24. Orientation Z, float 
25. High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight) 
26. Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) ) 
27. Camera (Which camera was used) 
28. Angle (The yaw angle for this camera)




Paper
-----

If you use the conditional imitation learning, please cite our ICRA 2018 paper.

End-to-end Driving via Conditional Imitation Learning. <br> Codevilla,
Felipe and Müller, Matthias and López, Antonio and Koltun, Vladlen and
Dosovitskiy, Alexey. ICRA 2018
[[PDF](http://vladlen.info/papers/conditional-imitation.pdf)]


```
@inproceedings{Codevilla2018,
  title={End-to-end Driving via Conditional Imitation Learning},
  author={Codevilla, Felipe and M{\"u}ller, Matthias and L{\'o}pez,
Antonio and Koltun, Vladlen and Dosovitskiy, Alexey},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2018},
}

