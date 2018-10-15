roslaunch quad_controller_rl rl_controller.launch task:=Takeoff agent:=Task01_DDPG
roslaunch quad_controller_rl rl_controller.launch task:=Hover agent:=Task02_DDPG
roslaunch quad_controller_rl rl_controller.launch task:=Land agent:=Task03_DDPG
roslaunch quad_controller_rl rl_controller.launch task:=Combined agent:=Task04_DDPG