#!/usr/bin/env python



#######################################################
# Assicurarsi che il subscriber e il publisher della comunicazione funzionano anche con 3 robot



#######################################################



# import os
# from math import copysign, cos, log, sin, sqrt
 
import roslib
import rospy
import sys
import torch
from network import DistributedNet
from com_network import ComNet
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from asebaros_msgs.msg import AsebaEvent
from thymio_msgs.msg import Comm
from math import log
from std_msgs.msg import Int16
# import std_srvs.srv
# from asebaros_msgs.msg import AsebaEvent
# from asebaros_msgs.srv import GetNodeList, LoadScripts
# from dynamic_reconfigure.server import Server
# from geometry_msgs.msg import Quaternion, Twist
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Imu, JointState, Joy, LaserScan, Range, Temperature
# from std_msgs.msg import Bool, ColorRGBA, Empty, Float32, Int8, Int16
# from tf.broadcaster import TransformBroadcaster
# from thymio_driver.cfg import ThymioConfig
# from thymio_msgs.msg import Led, LedGesture, Sound, SystemSound, Comm

BASE_WIDTH = 91.5     # millimeters
MAX_SPEED = 500.0     # units
SPEED_COEF = 2.93     # 1mm/sec corresponds to X units of real thymio speed
WHEEL_RADIUS = 0.022   # meters
GROUND_MIN_RANGE = 9     # millimeters
GROUND_MAX_RANGE = 30     # millimeters

PROXIMITY_NAMES = ['left', 'center_left', 'center',
				   'center_right', 'right', 'rear_left', 'rear_right']

sensor_max_range = 0.14
sensor_min_range = 0.0215
max_vel = 0.05 #0.14


class ThymioController(object):

#    def frame_name(self, name):
#        if self.tf_prefix:
#            return '{self.tf_prefix}/{name}'.format(**locals())
#        return name


	def __init__(self, comm_cmd):
		rospy.init_node('thymio_controller')
		self.thymio_name=rospy.get_namespace()[1:]
		self.cmd=comm_cmd


		self.state = np.zeros(2) #distanza avanti e dietro (if 2) # velocit dei vicini se 4
		self.received_comm = np.zeros(2) #distanza avanti e dietro (if 2) # velocit dei vicini se 4

		# Carico rete    
		net_file = rospy.get_param('net')
		c_net_file = rospy.get_param('c_net')
		self.net = torch.load(str(net_file))
		self.c_net = torch.load(str(c_net_file))


		# Inizializzo tutto

		# Subscriber e publiscer
			# Publish to the topic '/thymioX/cmd_vel'.  # forse serve la / iniziale
		self.velocity_publisher = rospy.Publisher('/'+self.thymio_name[6:]+'cmd_vel',Twist, queue_size=10)
  	#	self.aseba_pub = rospy.Publisher('events/set_speed', AsebaEvent, queue_size=10)

		# Subscribers to sensors
	#	self.sensor_subscribers=[
	#	rospy.Subscriber('proximity/'+name, Range, self.update_range)
	#	for name in PROXIMITY_NAMES]

		# real sensor subscribe
		self.sensor_subscriber = rospy.Subscriber('events/proximity', AsebaEvent, self.update_range2)

		# communication subscriber
	#	self.comm_subscriber = rospy.Subscriber('events/comm', AsebaEvent, self.update_comm)
		self.comm_subscriber = rospy.Subscriber('/'+self.thymio_name[6:]+'comm/rx',Comm, self.update_comm) 	
		self.comm_publisher = rospy.Publisher('/'+self.thymio_name[6:]+'comm/tx',Int16, queue_size=10)



		self.current_twist = Twist()
		self.current_range = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


		#debug
	

		# publish at this rate
		self.rate = rospy.Rate(20) #30

	#def set_speed(self, values):
	#	self.aseba_pub.publish(AsebaEvent(rospy.get_rostime(), 0, values))

	def proximity2range(self, raw):
		if raw > 4000:
			return -float('inf')
		if raw < 800:
			return float('inf')
		return -0.0736 * log(raw) + 0.632

	def map_tuple(self, func, tup):
		new_tuple = ()
		for itup in tup:
			new_tuple += (func(itup),)
		return new_tuple
		
#	def update_range(self, data):

#np.clip(data.range, sensor_min_range, sensor_max_range)
		# update range state any time a message is received
		# min_range, max_range
#		if data.header.frame_id == self.thymio_name + "proximity_center_link":
#			self.current_range[0] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_center_left_link":
#			self.current_range[1] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_center_right_link":
#			self.current_range[2] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_left_link":
#			self.current_range[3] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_right_link":
#			self.current_range[4] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_rear_left_link":
#			self.current_range[5] = np.clip(data.range, sensor_min_range, sensor_max_range)
#		elif data.header.frame_id == self.thymio_name + "proximity_rear_right_link":
#			self.current_range[6] = np.clip(data.range, sensor_min_range, sensor_max_range)
#
#		self.update_sensing()
	

	def update_range2(self, data):

#np.clip(data.range, sensor_min_range, sensor_max_range)
		# update range state any time a message is received
		# min_range, max_range
		sense = self.map_tuple(self.proximity2range, data.data)
		self.current_range = np.clip(sense, sensor_min_range, sensor_max_range)
		self.update_sensing()

	def com2int16(self, com):		
		return int(round(com*1000))

	def int162com(self,com):
		return com.astype(float)/1000.0



	def update_comm(self, data):

#np.clip(data.range, sensor_min_range, sensor_max_range)
		# update range state any time a message is received
		# min_range, max_range
		comms = data.payloads
		intensities= data.intensities
		front = np.argmax(intensities[0:5])
		rear = np.argmax(intensities[5:])
	#	if self.thymio_name=='aseba/thymio5/':
	#		print(self.thymio_name,'---  in_comm: ',[comms[front],comms[rear+5]])
			
		self.received_comm = np.array([comms[front],comms[rear+5]])



	def update_sensing(self):
		self.state= np.array([np.min(self.current_range[0:5]),np.mean(self.current_range[5:])])
	#	print(self.state)

	def move(self, seconds):

		vel_msg = Twist()
		vel_msg.angular.z = 0.0 # rad/s

		t = 0


		if comm_cmd:
			net=self.c_net
		else:
			net=self.net

		while t < seconds*10:

			#print(self.thymio_name + str(self.state))
			#if self.thymio_name=='aseba/thymio5/':
			#	print(self.thymio_name,'---  received_comm: ',self.received_comm)
			vel_msg.linear.x =  np.clip(- self.step(self.state, self.received_comm, net), -max_vel, max_vel)# m/s
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()
			t=t+1


#	def move2(self, seconds):
#
#		vel_angular = 0.0 # rad/s
#
#		t = 0
#
#
#		while t < seconds*10:
#
#			print(self.thymio_name + str(self.state))
#			vel_linear =  np.clip(- self.step(self.state,self.net), -max_vel, max_vel)# m/s
#			self.set_speed([vel_linear, vel_angular])
#			self.rate.sleep()
#			t=t+1

	def stop(self):

		vel_msg = Twist()
		vel_msg.linear.x = 0. # m/s
		vel_msg.angular.z = 0. # rad/s


		for t in range(5):
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()


	##### Presuppone che ogni robot trasmette un solo valore e ne riceve 2.

	def step(self, sensing, comm,net):
		net_controller = net.controller()
		#print(self.thymio_name,'---  in_comm: ',comm)
			
		if self.cmd:  ##### Questo  da rivedere
			in_comm=self.int162com(comm)
			control, next_com =net.single_step(sensing, in_comm)
			#if self.thymio_name=='aseba/thymio5/':
				#print(self.thymio_name,'--- sensing: ',sensing, ' in_comm: ',in_comm,' control: ',control.item(), 'next_com ',next_com.item())
			com_msg = Int16()
			com_msg.data = self.com2int16(next_com.item())
			self.comm_publisher.publish(com_msg)
			control=control.item()
		else:
			control =net_controller(sensing)
			control = control[0]
	#	control, *communication =net_controller(sensing)

	#	print('sensing: ', sensing, '. control: ',control)
	#	print('communication: ',communication)
		return control


if __name__ == '__main__':


	comm_cmd=True if sys.argv[1]=='com' else False
	comm_cmd=True

	controller = ThymioController(comm_cmd)



	rospy.sleep(8)

	controller.move(60)
	controller.stop()
	print('END')

		# Wait for sincronization

	#    rospy.wait_for_service('aseba/get_node_list')
	#    get_aseba_nodes = rospy.ServiceProxy(
	#        'aseba/get_node_list', GetNodeList)

		# Agisco
	rospy.spin()


def thymioDance():
	return 'Not Yet :D'