# SCRIMP
This is the code for implementing the SCRIMP algorithm :`SCRIMP: Scalable Communication for Reinforcement- and Imitation-Learning-Based Multi-Agent Pathfinding`

## Requirements

Python == 3.7
   ```
    pip install -r requirements.txt
   ```
    

## Setting up Code

* cd into the od_mstar3 folder.
* python3 setup.py build_ext --inplace.
* Check by going back to the root of the git folder, running python3 and `import od_mstar3.cpp_mstar`.
    
## Running Code

* Modify the parameters in `alg_parameters.py` to set the desired training setting and recording methods.
* Call python `driver.py`.
    
## Key Files

`alg_parameters.py` - Training parameters.

`driver.py` - Driver of program. Holds global training network for PPO.

`episodic_buffer.py` - Defines the episodic buffer used to generate intrinsic rewards.

`eval_model.py` - Evaluates trained model.

`mapf_gym.py` - Defines the classical Reinforcement Learning environment of Multi-Agent Pathfinding.

`model.py` - Defines the neural network-based operation model. 

`net.py` - Defines network architecture.

`runner.py` - A single process for collecting training data. 


## Other Links

Fully trained SCRIMP model - https://www.dropbox.com/scl/fo/ekhxyt7gm575kfwaerwb5/h?rlkey=j3cdikwofz0zelj2oci9q97k8&dl=0


## Authors

Yutong Wang

Bairan Xiang

Shinan Huang

Guillaume Sartoretti
