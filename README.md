# 10701-Project

## Project Setup
Stage the virtual environment with `chmod +x create_env.sh` and run with `./create_env.sh`.
This installs all requirements and creates a properly named virtual environment.

## Project Structure 

### /mujoco environments/
This folder contains `.xml` files corresponding to certain environments used in the training/testing processes. The format of these environment files follows the [Mujoco XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

## Running The Project
You can run an implementation as folllows

Run the bash script create_env.sh to activate the python venv

To run Dynamics-Based Action Projection: run $python3 

To run Lyapunov-Based Action Projection trained on entire trajectories: run $python3 train-and-eval-trajectory-based.py to run the Trajectory Based

To run Lyapunov-Based Action Projection trained on individual trajectory steps: run $python3 
