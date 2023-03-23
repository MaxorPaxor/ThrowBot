# ThrowBot

## Learning to Throw With a Handful of Samples Using Decision Transformers

### [Paper](https://ieeexplore.ieee.org/document/9984828?source=authoralert) | [Video](https://www.youtube.com/watch?v=5_G6o_H3HeE)

Throwing objects by a robot extends its reach and has many industrial applications. While analytical models can provide efficient performance, they require accurate estimation of system parameters. Reinforcement Learning (RL) algorithms can provide an accurate throwing policy without prior knowledge. However, they require an extensive amount of real world samples which may be time consuming and, most importantly, pose danger. Training in simulation, on the other hand, would most likely result in poor performance on the real robot. In this letter, we explore the use of Decision Transformers (DT) and their ability to transfer from a simulation-based policy into the real-world. Contrary to RL, we re-frame the problem as sequence modelling and train a DT by supervised learning. The DT is trained off-line on data collected from a far-from-reality simulation through random actions without any prior knowledge on how to throw. Then, the DT is fine-tuned on an handful ( ∼5 ) of real throws. Results on various objects show accurate throws reaching an error of approximately 4 cm. Also, the DT can extrapolate and accurately throw to goals that are out-of-distribution to the training data. We additionally show that few expert throw samples, and no pre-training in simulation, are sufficient for training an accurate policy.

<div align="center">

</div>

## Installation
### ROS Dependencies
1. **Motoman**
fixed motoman point-streaming:
https://github.com/MaxorPaxor/motoman_ps

    Motoman original repo: 
    https://github.com/ros-industrial/motoman/tree/kinetic-devel

2. **Robotiq gripper - 2f_140_gripper:**
https://github.com/ros-industrial/robotiq

    **Note**

    Two URDF files must be changed for the gripper to work:

    - robotiq_arg2f_transmission.xacro
    - robotiq_arg2f_140_model_macro.xacro
   
    must be replaced with the files located in robotiq_2f_140_gripper_changed_urdf folder.

3. **Mimic joints plugin:**
https://github.com/roboticsgroup/roboticsgroup_upatras_gazebo_plugins

4. **The-Gazebo-grasp-fix-plugin:**
The gripper has problems with picking objects.
The problems are fixed with Gazebo grasping plugin
https://github.com/JenniferBuehler/gazebo-pkgs/wiki/The-Gazebo-grasp-fix-plugin
