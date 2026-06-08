## Current Strategy
> CURRENT STATUS: It should be able to grasp and place normal objects. 

> AREAS TO IMPROVE: 1. Gemini cannot identify some visually ambiguous objects such as soap and book. 2. Gripper orientation is not guaranteed to be optimal 3. Failed attempts are not retried yet

Overall this strategy uses manually pre-defined poses to guide the arm instead of motion planning.
1. The arm is positioned at the "Scout" pose to view objects at an **oblique angle** (to better identify the objects by Gemini)
2. Gemini is asked to identify all objects and their bounding boxes, which are used to coarsely localize the arm
3. For each object, 
  - the arm is moved to one of the three "Home" poses based on the coarse localization
  - Gemini is asked to identify the bounding box of that specific object again, albeit from the **bird-eye view**, (to figure out the precise gripper orientation accordingly)
  - the arm with the gripped object is moved to pre-defined storage poses to place the object.

> Pre-defined poses are expressed in joint angles instead of poses because moving the arm by `self.arm.move_joint` is found to be more reliable than moving by `arm.move_pose`

## Commands
In Terminal 1, launch the Gazebo.
~~~bash
ros2 launch manip_challenge ur5_setup_random_picking.launch.py
# ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
~~~
In Terminal 2, build docker image
~~~bash
sudo docker build -t image_team_2 .
~~~
Run a container
~~~bash
sudo docker run -it --rm --net=host --ipc=host \
    --env-file .env \
    -v $HOME/cs477_WS/src/cs477_IIR/team_2:/ros2_ws/src/team_2 \
    image_team_2
~~~
Then, launch the `team_2`.
~~~bash
ros2 launch team_2 contest_run.launch.py
~~~
In Terminal 3, publish an instruction.
~~~bash
ros2 topic pub --once /task_commands std_msgs/String \
  "{data: 'Move the strawberry, the book, and the mustard bottle on the shelf. Move the coke can can on the right storage. Move the soap and eraser on the left storage'}"
~~~
