Build docker image.
~~~bash
sudo docker build -t image_team_2 .
~~~
Run a container.
~~~~bash
sudo docker run -it --rm --net=host --ipc=host \
    --env-file .env \
    -v ~/Desktop/models:/models \
    -v $HOME/cs477_WS/src/cs477_IIR/team_2:/ros2_ws/src/team_2 \
    image_team_2
~~~~
Useful commands
~~~~bash
ros2 launch team_2 contest_run.launch.py
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
ros2 launch manip_challenge ur5_setup_random_picking.launch.py
ros2 topic pub --once /task_commands std_msgs/String \
  "{data: 'Move the banana and the meat can on the shelf. Move the strawberry on the shelf. Move the coke can on the shelf.'}"
ros2 topic pub --once /task_commands std_msgs/String \
  "{data: 'Move the coke can on the shelf.'}"
~~~~

<!-- When you have GPU,
~~~~bash
sudo docker run --gpus all --net=host \
    -v ~/Desktop/models:/models \
    -v $HOME/cs477_ws/src/cs477_IIR/manip_challenge:/ros2_ws/src/manip_challenge \
    image_team_2
~~~~ -->
