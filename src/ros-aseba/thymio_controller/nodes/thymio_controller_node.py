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
from network import DistributedNet, DistributedNetL
from com_network import ComNet, ComNetL, ComNetLnoSensingN
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from asebaros_msgs.msg import AsebaEvent
from thymio_msgs.msg import Comm, Led
from math import log
from std_msgs.msg import Int16
from std_msgs.msg import ColorRGBA

import diagnostic_updater
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
max_vel = 0.06 #0.14


class ThymioController(object):

	def __init__(self, comm_cmd):
		rospy.init_node('thymio_controller')
		self.thymio_name=rospy.get_namespace()[1:]
		self.cmd=comm_cmd
		self.frequence = 5  # 0.2 
		self.state = np.zeros(2) #distanza avanti e dietro (if 2) # velocit dei vicini se 4
		self.received_comm = np.zeros(2) #distanza avanti e dietro (if 2) # velocit dei vicini se 4


		# diagnostic
		self.updater = diagnostic_updater.Updater()
		self.updater.setHardwareID("Thymio")
		self.comm_freq = diagnostic_updater.HeaderlessTopicDiagnostic('comm/rx', self.updater, diagnostic_updater.FrequencyStatusParam({'min':5, 'max':40}))



		# Carico rete    
		net_file = rospy.get_param('net')
		self.path = net_file
		c_net_file = rospy.get_param('c_net')
		l_net_file = rospy.get_param('l_net')

		self.net = torch.load(str(net_file))
		self.c_net = torch.load(str(c_net_file))
		self.l_net = torch.load(str(l_net_file))


		# debug log
		self.sensing_log = None
		self.communication_log = None
		self.control_log = None
		self.log = None


		# Inizializzo tutto

		# Subscriber e publiscer
			# Publish to the topic '/thymioX/cmd_vel'.  # forse serve la / iniziale
		self.velocity_publisher = rospy.Publisher('/'+self.thymio_name[6:]+'cmd_vel',Twist, queue_size=10)
  	
		# the publisher for leds
		self.led_publisher_top = rospy.Publisher('/'+self.thymio_name[6:] + 'led/body/top',
												  ColorRGBA, queue_size=10)

		self.led_publisher_right = rospy.Publisher('/'+self.thymio_name[6:] + 'led/body/bottom_right',
												  ColorRGBA, queue_size=10)

		self.led_publisher_left = rospy.Publisher('/'+self.thymio_name[6:] + 'led/body/bottom_left',
												  ColorRGBA, queue_size=10)


	
		# real sensor subscribe
		self.sensor_subscriber = rospy.Subscriber('events/proximity', AsebaEvent, self.update_range2)

		# communication subscriber
		self.comm_subscriber = rospy.Subscriber('/'+self.thymio_name[6:]+'comm/rx',Comm, self.update_comm) 	
		self.comm_publisher = rospy.Publisher('/'+self.thymio_name[6:]+'comm/tx',Int16, queue_size=10)

		# COmmunication Led publisher
		self.cled_publisher = rospy.Publisher('/'+self.thymio_name[6:] + 'led',
												  Led, queue_size=10)


		self.current_twist = Twist()
		self.current_range = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


		#debug	

		# publish at this rate
		self.rate = rospy.Rate(self.frequence) #30

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
		

	def update_range2(self, data):
		sense = self.map_tuple(self.proximity2range, data.data)
		self.current_range = np.clip(sense, sensor_min_range, sensor_max_range)
		self.update_sensing()

	def com2int16(self, com):		
		return int(round(com*1000))

	def int162com(self,com):
		return com.astype(float)/1000.0

	def update_comm_old(self, data):

		comms = data.payloads
		intensities= data.intensities
		front = np.argmax(intensities[0:5])
		rear = np.argmax(intensities[5:])
	#	if self.thymio_name=='aseba/thymio1/':
	#		print(self.thymio_name,'---  intensities: ',intensities, ' - payloads -: ', comms)
			
		self.received_comm = np.array([comms[front],comms[rear+5]])


	def update_comm(self, data):

		comms = data.payloads
		intensities= data.intensities
		idx = np.argmax(intensities)
		comm_value = comms[idx]
	#	if self.thymio_name=='aseba/thymio1/':
	#		print(self.thymio_name,'---  intensities: ',intensities, ' - payloads -: ', comms)
			
		# front
		if idx<=4:
			self.received_comm = np.array([comm_value,self.received_comm[1]])
		# rear
		elif idx > 4:
			self.received_comm = np.array([self.received_comm[0],comm_value])

		self.comm_freq.tick()
		self.updater.update()


	def update_sensing(self):
		self.state= np.array([np.min(self.current_range[0:5]),np.mean(self.current_range[5:])])


	def set_parameter(self, light1, light2, light3, value):
		light1 = value
		light2 = value
		light3 = value

	def illumination(self, comm):

		light_msg = ColorRGBA()


		# trasformo comm
		light_msg.b=10.0 
		light_msg.g=10.0
		light_msg.r=comm
		
		self.led_publisher_top.publish(light_msg)
		self.led_publisher_right.publish(light_msg)
		self.led_publisher_left.publish(light_msg)

	def illumination2(self, comm):

		# rescaling tra 0 e 8
		comm = comm.data/1000.0
		com_data = comm * 8
		values = np.zeros(8)


		for i in range(int(np.floor(com_data))):
			values[i]=1

		led_message = Led()
		led_message.id =0
		led_message.values = values 

		self.cled_publisher.publish(led_message)


	def task2(self,seconds):
		t = 0

		net = self.l_net

		light_msg = ColorRGBA()

		while t < seconds*self.frequence:

			light_msg.b, light_msg.g, light_msg.r= self.step2(self.state, self.received_comm, net)# m/s
	

			self.led_publisher_top.publish(light_msg)
			self.led_publisher_right.publish(light_msg)
			self.led_publisher_left.publish(light_msg)


		#	self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()
			t=t+1



	def task1(self, seconds):

		vel_msg = Twist()
		vel_msg.angular.z = 0.0 # rad/s

		t = 0

		if comm_cmd:
			net=self.c_net
		else:
			net=self.net

		while t < seconds*self.frequence:

			#print(self.thymio_name + str(self.state))
			#if self.thymio_name=='aseba/thymio5/':
			#	print(self.thymio_name,'---  received_comm: ',self.received_comm)
			vel_msg.linear.x =  np.clip(- self.step(self.state, self.received_comm, net), -max_vel, max_vel)/2# m/s
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()
			t=t+1

	def stop(self):

		vel_msg = Twist()
		vel_msg.linear.x = 0. # m/s
		vel_msg.angular.z = 0. # rad/s


		for t in range(5):
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()

		light_msg = ColorRGBA()


		# trasformo comm
		light_msg.b=0.0 
		light_msg.g=0.0 
		light_msg.r=0.0
		
		self.led_publisher_top.publish(light_msg)
		self.led_publisher_right.publish(light_msg)
		self.led_publisher_left.publish(light_msg)


		values = np.zeros(8)
		led_message = Led()
		led_message.id =0
		led_message.values = values 

		self.cled_publisher.publish(led_message)


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
			next_com_d =self.com2int16(next_com.item())
			com_msg.data = next_com_d
			self.comm_publisher.publish(com_msg)

			###
			control=control.item()
		else:
			control =net_controller(sensing)
			control = control[0]
	#	control, *communication =net_controller(sensing)
	
	#	Write on log
		if self.log is None:
			self.log = np.array([sensing, in_comm, control,next_com.item()])
		else:
			self.log = np.append(self.log, np.array([sensing, in_comm, control,next_com.item()]), axis=0)
		
	#	if self.sensing_log is None:
	#		self.sensing_log = np.array([sensing])
	#	else:
	#		self.sensing_log = np.append(self.sensing_log, np.array([sensing]), axis=0)
		
	#	if self.control_log is None:
	#		self.control_log = np.array([control])
	#	else:
	#		self.control_log = np.append(self.control_log, np.array([control]), axis=0)

	#	if self.communication_log is None and self.cmd:
	#		self.communication_log = np.array([next_com.item()])
	#	else:
	#		if self.cmd:
	#			self.communication_log = np.append(self.communication_log, np.array([next_com.item()]), axis=0)


	#	print('sensing: ', sensing, '. control: ',control)
	#	print('communication: ',communication)
	
		return control

	def step2(self, sensing, comm,net):
		net_controller = net.controller()
			
		in_comm=self.int162com(comm)
		control, next_com =net.single_step(sensing, in_comm, only_comm = True)
		com_msg = Int16()
		next_com_d =self.com2int16(next_com.item())
		com_msg.data = next_com_d
		self.comm_publisher.publish(com_msg)
		self.illumination2(com_msg)

			###
		control=control.item()
		
	#	Write on log
		if self.log is None:
			self.log = np.array([sensing, in_comm, control,next_com.item()])
		else:
			self.log = np.append(self.log, np.array([sensing, in_comm, control,next_com.item()]), axis=0)
		

		if control > 0.5:
			blue= 100
			green = 0
			red = 0
		else:
			blue= 0
			green = 0
			red = 100

#		print(blue," ",green, " ", red)
#		if self.thymio_name=='aseba/thymio1/':
#			print(self.thymio_name,'--- sensing: ',sensing, ' in_comm: ',in_comm,' control: ',control, 'next_com ',next_com.item())


		return blue,green,red

#############################################################################
########################   DEBUG FUNCTIONS   ################################



   # devo testare il single step della rete

	def heatbeat(self):
		print(self.thymio_name +" state: " + str(self.state))
		print(self.thymio_name +" received com: " + str(self.received_comm))
			#if self.thymio_name=='aseba/thymio5/':
			#	print(self.thymio_name,'---  received_comm: ',self.received_comm)

	def transmit(self, next_com):
		com_msg = Int16()
		com_msg.data = self.com2int16(next_com)

		print(self.thymio_name +" wants to transmit " + str(next_com) + " encoded with " + str(self.com2int16(next_com)))
		
		self.comm_publisher.publish(com_msg)

	def debug_step(self):
		# lavoro con thymio 1 e 3

		# iniziamo a vedere lo stato iniziale
		self.heatbeat()

		rospy.sleep(5)
		# adesso il thymio3 trasmette 0.8
		if self.thymio_name=='aseba/thymio3/':
			self.transmit(0.8)

		rospy.sleep(1)

		# vediamo cosa ricevono altro robot
		self.heatbeat()
		# thymio1 decodifica il messaggio
		if self.thymio_name=='aseba/thymio1/':
			print("decode :", self.received_comm)
			print(" in: ", self.int162com(self.received_comm))





if __name__ == '__main__':


	comm_cmd=True if sys.argv[1]=='com' else False
	comm_cmd=True

	controller = ThymioController(comm_cmd)


#	controller.debug_step()


	#### test single net
	#print("sensing: 0.049246 - 0.040555, received comm: 0.826386 - 1.363300, expected control: 0.062325, previous control: 0.061401, next control: 0.063756, next_com: 1.077578")
	#control, next_com = controller.c_net.single_step(np.array([0.049246 , 0.040555]), np.array([0.826386 , 1.363300]))	
	#print("control: ", control)
	#print("next_com: ", next_com)
	
	rospy.sleep(8)

#	controller.task1(50)

	controller.task2(60)

	controller.stop()

#	np.savetxt(controller.path[0:len(controller.path)-26]+'logs/'+controller.thymio_name[6:13]+'_log.csv', controller.log, delimiter=",")

#	np.savetxt(controller.path[0:len(controller.path)-26]+'logs/'+controller.thymio_name[6:13]+'_sensing.csv', controller.sensing_log, delimiter=",",fmt='%f')
#	np.savetxt(controller.path[0:len(controller.path)-26]+'logs/'+controller.thymio_name[6:13]+'_control.csv', controller.control_log, delimiter=",",fmt='%f')
#	if not(controller.communication_log is None):
#		np.savetxt(controller.path[0:len(controller.path)-26]+'logs/'+controller.thymio_name[6:13]+'_comm.csv', controller.communication_log, delimiter=",",fmt='%f')
			


	print('END')

		# Wait for sincronization

	#    rospy.wait_for_service('aseba/get_node_list')
	#    get_aseba_nodes = rospy.ServiceProxy(
	#        'aseba/get_node_list', GetNodeList)

		# Agisco
	rospy.spin()


def thymioDance():
	return 'Not Yet :D'