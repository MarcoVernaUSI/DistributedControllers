#!/usr/bin/env python
import roslib
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8,Float32
from sensor_msgs.msg import Range

from keyboard.msg import Key

MAX_LINEAR_SPEED=0.15
MAX_ANGULAR_SPEED=2

UP=32
DOWN=33
LEFT=17
RIGHT=16
CENTER=37

class RemoteToTwist(object):
	def __init__(self):

                rospy.Subscriber('keyboard/keydown',Key,self.on_keydown)
                
		rospy.Subscriber('/remote',Int8, self.on_remote)
		rospy.Subscriber('/ground/left',Range, self.on_ground)
		rospy.Subscriber('/ground/right',Range, self.on_ground)
		self._x=self._y=None
                self.cmdvel_pub = rospy.Publisher('/cmd_vel', Twist,queue_size=1)
                self.gesture_pub = rospy.Publisher('/led/gesture/circle',Float32,queue_size=1)
                self.tw = Twist()
                self.x=0
                self.y=0
        
                
        @property
        def x(self):
                return self._x

        @x.setter
        def x(self,nx):
                nx=min(1,max(-1,nx))
                if(nx!=self._x):
                        self._x=nx
                        self.tw.linear.x = MAX_LINEAR_SPEED*self._x
                        self.cmdvel_pub.publish(self.tw)
                        if abs(self.tw.linear.x)!=0:
                                self.gesture_pub.publish(Float32(0.07/self.tw.linear.x))
                        else:
                                self.gesture_pub.publish(1000.0)

        @property
        def y(self):
                return self._y

        @y.setter
        def y(self,ny):
                ny=min(1,max(-1,ny))
                if(ny!=self._y):
                        self._y=ny
                        self.tw.angular.z = MAX_ANGULAR_SPEED*self._y
                        self.cmdvel_pub.publish(self.tw)

        def on_ground(self,msg):
                if not(msg.range<msg.max_range):
                        self.x=0
                        self.y=0


	def on_remote(self,msg):
		tw = Twist()
                if(msg.data==UP):
                        self.x=self.x+0.1
                if(msg.data==DOWN):
                        self.x=self.x-0.1
                if(msg.data==LEFT):
                        self.y=self.y+0.1
                if(msg.data==RIGHT):
                        self.y=self.y-0.1
                if(msg.data==CENTER):
                        self.x=0
                        self.y=0

	def on_keydown(self,msg):
		tw = Twist()
                if(msg.code==Key.KEY_UP):
                        self.x=self.x+0.1
                if(msg.code==Key.KEY_DOWN):
                        self.x=self.x-0.1
                if(msg.code==Key.KEY_LEFT):
                        self.y=self.y+0.1
                if(msg.code==Key.KEY_RIGHT):
                        self.y=self.y-0.1
                if(msg.code==Key.KEY_SPACE):
                        self.x=0
                        self.y=0



## Create a ROS node and instantiate the class.
def main():
	'''Create a ROS node and instantiate the class.'''
	try:
		rospy.init_node('remote_to_twist')
		rtt = RemoteToTwist()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

if __name__== '__main__':
	main()
	

