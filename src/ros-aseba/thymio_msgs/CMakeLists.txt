cmake_minimum_required(VERSION 2.8.3)
project(thymio_msgs)


find_package(catkin REQUIRED COMPONENTS
	std_msgs
	message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Led.msg
  LedGesture.msg
  Sound.msg
  SystemSound.msg
	Comm.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS  std_msgs  message_runtime
)

###########
## Build ##
###########


#############
## Install ##
#############
