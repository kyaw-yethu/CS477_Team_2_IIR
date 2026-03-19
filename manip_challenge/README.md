# Manipulation challenge tutorial 1

In this tutorial, you will learn how to pick an object from a tabletop environment containing multiple spawned objects within the Gazebo simulator.

[!IMPORTANT]
Always remember to source your workspace environment whenever you open a new terminal:
~~~~bash
source ./install/setup.bash
~~~~

## 1.0 Build Instructions
If you have not compiled this package yet, or if you have made changes to the code, run the following command from your workspace root:
~~~~bash
colcon build --symlink-install --packages-select manip_challenge
~~~~

## 1.1 Initiate the Pick-and-Place Environment

To launch the Gazebo simulation environment, open Terminal 1 and run:
~~~~bash
ros2 launch manip_challenge ur5_setup_bin_picking.launch.py
~~~~

**Configuration Steps**:
1. Select the 'position' controller by pressing 1 (do this twice as prompted).
2. Select 'GAZEBO (GUI)' mode by pressing 1.
3. Note: If this is your first time launching the environment, it may take a few minutes to download and load the models.

**Grasping an Object**
You can control the simulated UR5e arm and gripper by running a sequence of joint commands. Open Terminal 2 and execute:
~~~~bash
ros2 run manip_challenge example1_grasping
~~~~
Tip: Inspect example1.py to understand how to send joint position commands and toggle the gripper (Open/Close).


## 1.2 Access Ground-Truth Object Poses
While this challenge eventually requires perception, you can use Gazebo's internal state for debugging and dry runs. You can query the ground-truth pose of objects by running:
~~~~bash
ros2 run manip_challenge example2_gazebo_pose
~~~~

[!WARNING]
Subscribing to Gazebo internal topics for object states is strictly prohibited during the actual competition. This is provided for system testing and dry runs only.


## 1.3 Task Specification (Command System)
During the competition, the instructor/TA will broadcast task instructions via the /task_commands topic.

**Example Command Format**:
~~~~bash
d = "Move the book, eraser, and soap to the left storage. Move the snack, biscuits box, and soap to the right storage."
~~~~
Your robot system must subscribe to this topic and perform the pick-and-place tasks accordingly. You can test the communication flow using the following scripts:

**Terminal 1 (Listener)**:
Check if your system correctly receives the command.
~~~~bash
ros2 run manip_challenge example3_task_command_sub
~~~~

**Terminal 2 (Publisher)**:
Simulate the instructor sending a command.
~~~~bash
ros2 run manip_challenge example3_task_command_pub
~~~~


# Manipulation Challenge Tutorial 2

In this tutorial, we use the Gemini API to estimate the coordinates of a target object in a cluttered environment based on a textual instruction.

Before proceeding, please review the [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart).  
Note that you must first create a project and generate an [API key](https://ai.google.dev/gemini-api/docs/api-key#api-keys).

You also need to install the following packages:

~~~~bash
pip install -U google-genai Pillow
~~~~

## 2.1 Initialize the random object pick-and-place environment
In the competition environment, approximately 10 random objects will appear on the table, and a text instruction will be delivered through a ROS2 topic.

You can start the random object environment by running:
~~~~bash
ros2 launch manip_challenge ur5_setup_random_picking.launch.py
~~~~

## 2.2 Detect an object using a VLM
You can start a detection service that queries the Gemini model:
~~~~bash
ros2 run manip_challenge example4_detection_server
~~~~

Once the server is running, you can request the location of a designated object by calling the client:
~~~~bash
ros2 run manip_challenge example4_detection_client
~~~~
