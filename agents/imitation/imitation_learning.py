
import sys
import os

import socket
import scipy
import re

import math

from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
import tensorflow as tf
import time
from ConfigParser import ConfigParser



import pygame
from pygame.locals import *
sys.path.append('../train')

from carla import sensor
from carla.client import CarlaClient
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from carla import image_converter
from agents.imitation.imitation_learning_network import load_imitation_learning_network


#from codification import *

import copy
import random

slim = tf.contrib.slim



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


  def __init__(self,experiment_name ='None',driver_conf=None,memory_fraction=0.9,image_cut =[115,510] ):


    #use_planner=False,graph_file=None,map_file=None,augment_left_right=False,image_cut = [170,518]):

    Agent.__init__(self)

      
    conf_module  = __import__(experiment_name)
    #self._config = conf_module.configInput()
    #self._config_train = conf_module.configTrain()
    
    config_gpu = tf.ConfigProto()
    # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

    config_gpu.gpu_options.visible_device_list='0' 

    config_gpu.gpu_options.per_process_gpu_memory_fraction=memory_fraction


    self._sess = tf.Session(config=config_gpu)
      
    self._network_tensor =  load_imitation_learning_network(input_image,input_data, input_size,dropout,config)




    self._sess.run(tf.global_variables_initializer())




    self._image_cut = driver_conf.image_cut


"""
  def _load_model(self,checkpoint_name=None):

    if self._config_train.restore_seg_test:
      if  self._config.segmentation_model != None:
        exclude = ['global_step']

        variables_to_restore = slim.get_variables(scope="ENet_Small")

        saver = tf.train.Saver(variables_to_restore,max_to_keep=0)

        seg_ckpt = restore_session(self._sess,saver,self._config.segmentation_model)


      variables_to_restore = list(set(tf.global_variables()) - set(slim.get_variables(scope="ENet_Small")))

    else:
      variables_to_restore = tf.global_variables()

    saver = tf.train.Saver(variables_to_restore)
    cpkt = restore_session(self._sess,saver,self._config.models_path,checkpoint_name)
"""

    


  def compute_direction(self,pos,ori):  # This should have maybe some global position... GPS stuff
    

      command,made_turn,completed = self.planner.get_next_command(pos,ori,(self.positions[self._target].location.x,self.positions[self._target].location.y,22),(1.0,0.02,-0.001))
      return command

    

  def run_step(self,measurements,sensor_data,target):



    direction,_ = self._planner.get_next_command((measurements.player_measurements.transform.location.x,measurements.player_measurements.transform.location.y,22),\
      (measurements.player_measurements.transform.orientation.x,measurements.player_measurements.transform.orientation.y,measurements.player_measurements.transform.orientation.z),\
      (target.location.x,target.location.y,22),(1.0,0.02,-0.001))


   
    print (sensor_data['RGB'].data).shape

    print (sensor_data['Labels'].data).shape


    sensors = []
    sensors.append(sensor_data['RGB'].data) 
    sensors.append(sensor_data['Labels'].data)

    control = self._compute_action(sensors,measurements.player_measurements.forward_speed,direction)



    return control

  def _compute_action(self,sensors,speed,direction=None):
    
    capture_time = time.time()


    if direction == None:
      direction = self.compute_direction((0,0,0),(0,0,0))

    sensor_pack =[]


    for i in range(len(sensors)):

      sensor = sensors[i] 
      if self._config.sensor_names[i] =='rgb':

        
        sensor = scipy.misc.imresize(sensor,[self._config.sensors_size[i][0],self._config.sensors_size[i][1]])


      elif self._config.sensor_names[i] =='labels':

        sensor = sensor[self._image_cut[0]:self._image_cut[1],:] 

        sensor = scipy.misc.imresize(sensor,[self._config.sensors_size[i][0],self._config.sensors_size[i][1]],interp='nearest')

        sensor = join_classes(sensor) * int(255/(number_of_seg_classes-1))

        sensor = sensor[:,:,np.newaxis]


      sensor_pack.append(sensor)

      
     

    
    
    if len(sensor_pack) > 1:

      print sensor_pack[0].shape

      print sensor_pack[1].shape
      image_input =  np.concatenate((sensor_pack[0],sensor_pack[1]),axis=2)

    else:
      image_input = sensor_pack[0]
    
    #image_result = Image.fromarray(sensor)
    #image_result.save('image.png')

    image_input = image_input.astype(np.float32)
    image_input = np.multiply(image_input, 1.0 / 255.0)


    steer,acc,brake = self._control_function(image_input,speed,direction,self._config,self._sess,self._train_manager)



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


def _control_function(image_input,speed,control_input,config,sess,train_manager):

  
  branches = train_manager._output_network
  x = train_manager._input_images 
  dout = train_manager._dout
  input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
  input_control =  train_manager._input_data[config.inputs_names.index("Control")]
  #image_result = Image.fromarray((image_input*255).astype(np.uint8))
  #image_result.save('image.png')

  image_input = image_input.reshape((1,config.image_size[0],config.image_size[1],config.image_size[2]))

  speed = np.array(speed/config.speed_factor)

  speed = speed.reshape((1,1))

  if control_input ==2 or control_input==0.0:
    all_net = branches[0]
  elif control_input == 3:
    all_net = branches[2]
  elif control_input == 4:
    all_net = branches[3]
  elif control_input == 5:
    all_net = branches[1]


  #print clip_input.shape


      

  feedDict = {x: image_input,input_speed:speed,dout: [1]*len(config.dropout) }


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

