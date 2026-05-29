# =============================================================================
# image_team_2 — CS477 IIR Picking Challenge
#
# ROS2 Humble + the custom perception/LLM deps team_2's nodes need at runtime.
# Gazebo and the rest of the sim stack run on the HOST, NOT in this image.
#
# Layer order is cache-friendly: static base/env -> apt -> pip -> model weights
# -> source -> colcon build. Editing source only re-runs the bottom layers; the
# expensive apt / pip / weight-download layers stay cached across rebuilds.
# =============================================================================
FROM osrf/ros:humble-desktop

# ---- Static environment -----------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
# Match the host's RMW so DDS discovery works over --net=host.
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# torch's cu124 wheels bundle their own CUDA runtime libs here. Harmless on the
# no-GPU laptop (just unused); used on the CUDA test server.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib

# ---- System deps (apt) ------------------------------------------------------
# Deliberately NO gazebo / gazebo plugins / ros2_control — those live on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-pykdl \
      python3-vcstool \
      ros-humble-moveit \
      ros-humble-py-trees \
      ros-humble-py-trees-ros-interfaces \
      ros-humble-rmw-fastrtps-cpp \
      ros-humble-xacro \
      ros-humble-diagnostic-updater \
      libnlopt-cxx-dev \
      git build-essential \
      wget ca-certificates \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python: ROS / manip_challenge deps -------------------------------------
# Base requirements.txt + manip_challenge/setup.py install_requires.
# google-genai + Pillow are only for the example4/5 Gemini nodes; drop them to
# slim the image if you no longer run those.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "setuptools<66.0" \
      py_trees pyassimp empy==3.3.4 lark "numpy~=1.24.4" \
      billiard FuzzyTM catkin_pkg \
      google-genai Pillow

# ---- Python: PyTorch --------------------------------------------------------
# cu124 wheels are binary-compatible with the server's 12.8 driver, and still
# import (CPU-only) on the AMD laptop.
RUN pip install --no-cache-dir --ignore-installed torch torchvision \
      --index-url https://download.pytorch.org/whl/cu124

# ---- Python: perception — Grounding DINO via transformers -------------------
# Open-vocabulary detector pulled from the HF hub (reachable on our network,
# unlike the OpenAI-CLIP CDN that ultralytics/YOLO-World depended on).
# transformers provides the model + processor; huggingface_hub provides the CLI
# used to bake the weights below.
RUN pip install --no-cache-dir -U transformers huggingface_hub

# ---- Python: LLM — Qwen 2.5 via llama-cpp-python ----------------------------
# CPU wheel (runs on the laptop). For the CUDA server, comment this out and use
# the cu124 wheel below for GPU offload.
RUN pip install --no-cache-dir llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# RUN pip install --no-cache-dir llama-cpp-python \
#       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# ---- Pin numpy < 2 ----------------------------------------------------------
# MUST be the LAST pip step so no transitive dependency can bump numpy back to
# 2.x (which breaks ROS2 and the system matplotlib).
RUN pip install --no-cache-dir --force-reinstall "numpy<2"

# ---- Model weights (baked in for an offline, self-contained image) ----------
# Heavy and rarely-changing, so placed BEFORE the source COPY: editing nodes
# won't invalidate this layer and trigger a re-download. No HF_HUB_OFFLINE is
# set yet, so this download has network; runtime is forced offline lower down.
# RUN mkdir -p /models && \
#     hf download IDEA-Research/grounding-dino-base \
#       --local-dir /models/grounding-dino-base
ENV GDINO_DIR=/models/grounding-dino-base

# Qwen GGUF: bake it the same way for the final image, or keep mounting /models
# as a volume during dev. NOTE: the official Qwen repo ships q4_k_m SPLIT into
# two shards; for a single self-contained file use bartowski's merged copy.
# RUN hf download bartowski/Qwen2.5-7B-Instruct-GGUF \
#       --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" --local-dir /models
ENV QWEN_GGUF=/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# ---- Workspace: source + build ----------------------------------------------
WORKDIR /ros2_ws
RUN mkdir -p src
COPY manip_challenge               /ros2_ws/src/manip_challenge
COPY utils/riro-kdl                /ros2_ws/src/riro-kdl
COPY utils/riro-msgs               /ros2_ws/src/riro-msgs
COPY platforms/ur5_ros2_gazebo     /ros2_ws/src/ur5_ros2_gazebo
COPY platforms/robotiq_85_gripper  /ros2_ws/src/robotiq_85_gripper
COPY assignment_1                      /ros2_ws/src/assignment_1
COPY assignment_2                      /ros2_ws/src/assignment_2
COPY team_2                        /ros2_ws/src/team_2
COPY team_2_interfaces              /ros2_ws/src/team_2_interfaces

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install && \
    rm -rf log

# ---- Runtime config ---------------------------------------------------------
# FastDDS profile that disables shared-memory transport (root-in-container vs.
# regular-user-on-host SHM permission clash over --net=host).
COPY manip_challenge/disable_shm.xml /root/.fastdds_disable_shm.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/root/.fastdds_disable_shm.xml
ENV ROS_DOMAIN_ID=11
# Force transformers/HF to load only the baked weights — never hit the network
# at runtime (the competition server may be fully offline).
ENV HF_HUB_OFFLINE=1

# Auto-source ROS + workspace in every interactive shell.
RUN echo "source /opt/ros/humble/setup.bash"  >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> /root/.bashrc

# Standby default — drop into bash. For submission, replace with the launch that
# sits idle on the instruction topic:
#   CMD ["ros2", "launch", "team_2", "contest_run.launch.py"]
CMD ["bash"]