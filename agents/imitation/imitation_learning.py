
import sys
import os

import scipy
import re

import math

import tensorflow as tf
import time



sys.path.append('carla/PythonClient')


import pygame
from pygame.locals import *
sys.path.append('../train')

from carla import sensor
from carla.client import CarlaClient
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from benchmarks.agent import Agent

from carla import image_converter
from agents.imitation.imitation_learning_network import load_imitation_learning_network
import numpy as np

from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
#from codification import *

import copy
import random

slim = tf.contrib.slim


from carla.carla_server_pb2 import Control

"""
number_of_seg_classes = 5
classes_join = {0:2,1:2,2:2,3:2,5:2,12:2,9:2,11:2,4:0,10:1,8:3,6:3,7:4}


def join_classes(labels_image):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in classes_join.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value


  return compressed_labels_image



def restore_session(sess,saver,models_path):

  ckpt = 0
  if not os.path.exists(models_path):
    os.mkdir( models_path)
  
  ckpt = tf.train.get_checkpoint_state(models_path)
  if ckpt:
    print 'Restoring from ',ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    ckpt = 0

  return ckpt


def load_system(config):
  config.batch_size =1
  config.is_training=False

  training_manager= TrainManager(config,None)
  if hasattr(config, 'plug_segmentation_soft'):

    training_manager.build_seg_network_dif()
  else:
    if hasattr(config, 'plug_segmentation'):

      training_manager.build_seg_network()
    else:
      if hasattr(config, 'input_one_hot'):
        training_manager.build_network_one_hot()
      else:
        training_manager.build_network()



  
  



  return training_manager
"""
""" Initializing Session as variables that control the session """





class ImitationLearning(Agent):


  def __init__(self,city_name,memory_fraction=0.35,image_cut =[115,510]):


    Agent.__init__(self,city_name)


    
    self.dropout_vec = [1.0]*8 + [0.7]*2 + [0.5]*2 + [0.5]*1 + [0.5, 1.]*5

    
    config_gpu = tf.ConfigProto()

    # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

    config_gpu.gpu_options.visible_device_list='0' 

    config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction

    self._image_size = (88,200,3)


    self._sess = tf.Session(config=config_gpu)
      

    with tf.device('/gpu:0'):
      

      self._input_images = tf.placeholder("float",shape=[None,self._image_size[0],self._image_size[1],\
      self._image_size[2]], name="input_image")


      self._input_data = []
      
      self._input_data.append(tf.placeholder(tf.float32,\
       shape=[None, 4],name="input_control"))

      self._input_data.append(tf.placeholder(tf.float32,\
       shape=[None, 1],name="input_speed"))

          
      self._dout = tf.placeholder("float",shape=[len(self.dropout_vec)])


    with tf.name_scope("Network"):
      self._network_tensor =  load_imitation_learning_network(self._input_images,self._input_data, self._image_size,self._dout)

    import os
    dir_path = os.path.dirname(__file__)

    self._models_path = dir_path + '/model/'

    #tf.reset_default_graph()
    self._sess.run(tf.global_variables_initializer())


    self.load_model()


    self._image_cut = image_cut



  def load_model(self):



    variables_to_restore = tf.global_variables()

    saver = tf.train.Saver(variables_to_restore,max_to_keep=0)
    #cpkt = restore_session(self._sess,saver,self._config.models_path,checkpoint_name)
    ckpt = 0
    if not os.path.exists(self._models_path):
      
      raise RuntimeError('failed to find the models path')

    ckpt = tf.train.get_checkpoint_state(self._models_path)
    if ckpt:
      print 'Restoring from ',ckpt.model_checkpoint_path  
      saver.restore(self._sess,ckpt.model_checkpoint_path)
    else:
      ckpt = 0

    return ckpt

    


  def run_step(self,measurements,sensor_data,target):



    direction,_ = self._planner.get_next_command((measurements.player_measurements.transform.location.x,measurements.player_measurements.transform.location.y,22),\
      (measurements.player_measurements.transform.orientation.x,measurements.player_measurements.transform.orientation.y,measurements.player_measurements.transform.orientation.z),\
      (target.location.x,target.location.y,22),(1.0,0.02,-0.001))


   


    control = self._compute_action(sensor_data['CameraRGB'].data,measurements.player_measurements.forward_speed,direction)



    return control

  def _compute_action(self,rgb_image,speed,direction=None):
    
    capture_time = time.time()




    rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1],:] 

    image_input = scipy.misc.imresize(rgb_image,[self._image_size[0],self._image_size[1]])

    
    #image_result = Image.fromarray(sensor)
    #image_result.save('image.png')

    image_input = image_input.astype(np.float32)
    image_input = np.multiply(image_input, 1.0 / 255.0)


    steer,acc,brake = self._control_function(image_input,speed,direction,self._sess)


    # This a bit biased

    if brake < 0.1:
      brake =0.0

    if acc> brake:
      brake =0.0
    if speed > 35.0 and brake == 0.0:
      acc=0.0
      
    control = Control()
    control.steer = steer
    control.throttle =acc 
    control.brake =brake
    # print brake

    control.hand_brake = 0
    control.reverse = 0



    return control
  
  # The augmentation should be dependent on speed


  def _control_function(self,image_input,speed,control_input,sess):


    
    branches = self._network_tensor
    x = self._input_images 
    dout = self._dout
    input_speed = self._input_data[1]
    input_control =  self._input_data[0]
    #image_result = Image.fromarray((image_input*255).astype(np.uint8))
    #image_result.save('image.png')

    image_input = image_input.reshape((1,self._image_size[0],self._image_size[1],self._image_size[2]))

    speed = np.array(speed/90.0)

    speed = speed.reshape((1,1))

    if control_input ==2 or control_input==0.0:
      all_net = branches[0]
    elif control_input == 3:
      all_net = branches[2]
    elif control_input == 4:
      all_net = branches[3]
    elif control_input == 5:
      all_net = branches[1]

    print control_input
    #print clip_input.shape


        

    feedDict = {x: image_input,input_speed:speed,dout: [1]*len(self.dropout_vec) }


    output_all = sess.run(all_net, feed_dict=feedDict)


    
    predicted_steers = (output_all[0][0])

    predicted_acc = (output_all[0][1])

    predicted_brake = (output_all[0][2])

    #predicted_speed =  sess.run(branches[4], feed_dict=feedDict)
    #predicted_speed = predicted_speed[0][0]
    #real_speed = speed*config.speed_factor
    #print ' REAL PREDICTED ',predicted_speed*config.speed_factor

    #print ' REAL SPEED ',real_speed
    #real_predicted =predicted_speed*config.speed_factor
    #if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
    #  print 'BOOSTING'
    #  predicted_acc =  1*(20.0/config.speed_factor -speed) + predicted_acc  #print "DURATION"

    #  predicted_brake=0.0

    #  predicted_acc = predicted_acc[0][0]


      
    return  predicted_steers,predicted_acc,predicted_brake

