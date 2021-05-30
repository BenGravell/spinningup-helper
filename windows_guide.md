# OpenAI Spinning Up Install Guide for Windows 10 + Nvidia GTX 1070

## Global notes
- These instructions roughly follow https://github.com/openai/spinningup/issues/215 and let you run everything in Windows natively.
- This is in contrast to the earlier https://github.com/openai/spinningup/issues/23 which requires you to use WSL to mimic Linux functionality.


ALWAYS RUN ANACONDA PROMPT AS ADMINSTRATOR 
- Right click the app icon and use menu to 'Run as administrator'
- Alternately, so you never forget, set your Anaconda Prompt to always run as administrator following https://www.cnet.com/how-to/always-run-a-program-in-administrator-mode-in-windows-10/

ALWAYS ACTIVATE YOUR ENVIRONMENT BEFORE INSTALLING PACKAGES OR RUNNING PYTHON FROM PROMPT
- From an Anaconda Prompt run the command
- `conda activate spinningup`


### Anaconda commands cheat sheet
- `conda env list`    
  - Show a list of all environments on the system
- `conda list`        
  - Show a list of all installed packages in the currently activated environment
- `conda config --show channels'
  - Show a list of all active high-level channel names, in priority order (highest priority at top to lowest priority at the bottom)

## Guide
### Create a new environment
- From an Anaconda prompt run the commands
- `conda create --name spinningup python=3.6`
- `conda activate spinningup`

### Add channels
- Add the `conda-forge` channel with the command
- `conda config --append channels conda-forge`
- In case you have other channels already, you can manually re-arrange the channel priority order by editing your `.condarc` file
  - My `.condarc` file is in `C:\Users\bjgra`


### Clone the [spinningup Git repo](https://github.com/openai/spinningup) using Git or GitHub Desktop
- The location where the spinningup repo is cloned on your computer can be chosen as desired
  - I prefer to install them in the default location at `C:\Users\bjgra\Documents\GitHub`

### Stop spinningup from automatically installing all packages at once
- In File Explorer, navigate into the spinningup Git repo folder and edit `setup.py` and remove the following lines from the `install_requires` block:
  - `'numpy'`
  - `'scipy'`
  - `'matplotlib==3.1.1'`
  - `'pandas'`
  - `'joblib'`
  - `'mpi4py'`
  - `'pytest'`
  - `'psutil'`
  - `'tqdm'`
  - `'cloud_pickle==1.2.1'`
  - `'torch==1.3.1'`
  - `'tensorflow>=1.8.0,<2.0'`
- This should leave only the following:
- `install_requires=['gym[atari,box2d,classic_control]~=0.15.3'],`
- Compare against [`copy_this_over___setup.py`](copy_this_over___setup.py) file.

### Install PyTorch with CUDA
- `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge`
- Official PyTorch instructions: https://pytorch.org/get-started/locally/

To make sure PyTorch can use CPU and GPU without failing, run [`cpu_gpu_test.py`](cpu_gpu_test.py)

### Install TensorFlow
- `conda install tensorflow">=1.8.0,<2.0"`
- NOTE: Spinning Up uses old versions of TensorFlow that do not match the current docs in some ways. 
- For example when doing the basics at https://www.tensorflow.org/tutorials/customization/basics you will find that 
  - `ndarray = np.ones([3, 3])`
  - `tensor = tf.multiply(ndarray, 42)`
  - `print(tensor)`
- yields a different result than the official docs.
- This is because TensorFlow changed how it handles evaluation of the computational graph. 
- For the old version we have to start a session and run it to evaluate the result manually.
  - `sess = tf.Session()`
  - `sess.run(tensor)`
- See https://stackoverflow.com/questions/46548339/tensorflow-tensormul0-shape-dtype-int32

### Install standard packages
- We are doing this now to isolate install potential problems from the more complicated packages later.
- These packages are well-developed so you should not have any trouble installing these.
- IMPORTANT: Install these in two separate groups to avoid issues with
- > LinkError: post-link script failed for package ...
- `conda install ipython numpy scipy pandas matplotlib=3.1.1 seaborn=0.8.1`
- `conda install pytest psutil joblib tqdm mpi4py cloudpickle=1.2.1`
<!-- - Nevertheless, it will be useful to install these one-at-a-time to isolate any issues.
  - `conda install ipython`
  - `conda install numpy`
  - `conda install scipy`
  - `conda install pandas`
  - `conda install matplotlib=3.1.1`
  - `conda install seaborn=0.8.1`
  - `conda install pytest`
  - `conda install psutil`
  - `conda install joblib`
  - `conda install tqdm`
  - `conda install mpi4py`
  - `conda install cloudpickle=1.2.1`
- Alternatively, if you trust these standard packages to install correctly you can install all at once with
- `conda install ipython numpy scipy pandas matplotlib=3.1.1 seaborn=0.8.1 pytest psutil joblib tqdm mpi4py cloudpickle=1.2.1`  -->

### Set up OpenAI Gym
Now we need to prevent an error with OpenAI Gym, specifically with the Box2D environment which needs `swig` in order to build properly.
- `conda install -c anaconda swig`
- `conda install -c conda-forge pybox2d`

### Install OpenAI Spinning Up
From here we follow the official OpenAI Spinning Up instructions https://spinningup.openai.com/en/latest/user/installation.html
- In an Anaconda Prompt, change directories to the spinningup Git repo
- `cd [YOUR_PATH_HERE]\spinningup`
- Now run the command to install spinningup and its dependencies
- `pip install -e .`

### Test Spinning Up
Throughout this section compare with the screenshot collage [`spinningup_install_check.jpg`](spinningup_install_check.jpg).

Run the test script command to train an RL agent using the PPO algorithm.
- `python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999`
- NOTE: This will take several minutes to complete.
- NOTE: If it works properly you should see real time update tables of values like
  - 'Epoch ...'
  - 'AverageEpRet ...' 
  - ...and lines like...
  - 'Warning: trajectory cut off by epoch at xxx steps.'
- NOTE: This test script uses TensorFlow, not PyTorch.

Then you can watch a video of the trained policy.
- `python -m spinup.run test_policy data\installtest\installtest_s0`
- NOTE: By default test_policy.py will run 100 episodes.
- Edit the __main__ of test_policy.py to use fewer episodes (perhaps just 3) if you dont want to wait around so long.
- This will also take several minutes to complete (there are multiple episodes).

And plot the results.
- `python -m spinup.run plot data\installtest\installtest_s0`
- NOTE: You MUST use backslashes in the path, which is the Windows convention.
- This contrasts with the Linux convention of forward slashes, which is used throughout Spinning Up.
  - Do not just blindly copy code snippets that involve paths from the Spinning Up docs!
- Otherwise `plot.py` will not build the `logdirs` list of strings properly.


## (Optional) MuJoCo and ToyText Gym Environments
These instructions are optional and only for getting the additional MuJoCo and ToyText environments in Gym.

These instructions roughly follow https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5 but with some small critical changes.

Note that MuJoCo requires a license. Student licenses are free and can be requested from the MuJoCo developers.

### Install Microsoft Visual C++ build tools 2017/2019
- If you do not have it installed in your system, you can download from https://visualstudio.microsoft.com/downloads/.
- Go to the link and scroll down until you reach “Tools for Visual Studio 2019” and download the "Build Tools for Visual Studio 2019"
- Once you’ve downloaded it, open the application and you should land in “Workloads” tab. Make sure you have `C++ Build Tools` ticked and click install. The installation file will be around 4.59GB and will need you to restart your system once prompted.

### Install pystan for ToyText environments in Gym
- `conda install -c conda-forge pystan`

### Prepare MuJoCo
Since we are using Windows, we must use an older version of MuJoCo.
- Go to https://www.roboti.us/index.html
- Download `mjpro150 win64` which is a .zip file
- Create a folder in your %userprofile% called “.mujoco”.
  - From here on I will refer to %userprofile% with my own username which is "bjgra"
- Once you have downloaded the zip for mujoco, extract it to "~bjgra\.mujoco"
  - The file structure should be like "C:\Users\bjgra\.mujoco\mjpro150" and mjpro150 should have 5 folders
    - bin
    - doc
    - include
    - model
    - sample
- After registering your computer id, Roboti LLC will email you a mjkey.txt which you need to download and copy to "~\bjgra\.mujoco\mjkey.txt"
- Now you must set the environment variables that will tell MuJoCo and mujoco-py where the MuJoCo files and license are located.
  - From the Windows start menu (press the Windows key) search for "environment variables" and click the link to "Edit the system environment variables" in the Control Panel
  - This will open the "System Properties" window. Click on "Environment Variables..." at the bottom.
  - First, save your existing "Path" or "PATH" environment variable to a safe location since we are about to overwrite it.
    - Highlight the "PATH" environment variable and click "Edit..."
    - This will pop out a new window. At the bottom click "Edit text..."
    - This will pop out yet another window. Now you can copy the text from the "Variable value:" field and save it somewhere safe to recover later.
    - While we are at it, replace the "Variable value:" field with our new value "C:\Users\bjgra\.mujoco\mjpro150\bin;%PATH%;"
  - Under the "User variables for user" box at the top, click "New..." to add a new environment variable
    1. Environment variable for the license
    - Variable name: MUJOCO_PY_MJKEY_PATH 
    - Variable value: C:\Users\bjgra\.mujoco\mjkey.txt
    2. Environment variable for the MuJoCo files
    - Variable name: MUJOCO_PY_MUJOCO_PATH 
    - Variable value: C:\Users\bjgra\.mujoco\mjpro150
- NOTE: Every time you change an environment variable you must start a new Anaconda Prompt in order to register the change.

### Install mujoco-py
Since we are using Windows, we must use an older version of mujoco-py.
- Go to the old release `mujoco 1.50.1.68` at https://github.com/openai/mujoco-py/tree/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191
- Clone or simply download and extract the repo to a folder mujoco-py.
- Change directories in Anaconda Prompt to folder mujoco-py.
- Install the following dependencies:
  - `pip install cffi`
  - `pip install pygit2`
  - `python -m pip install --upgrade setuptools`
  - `pip install -r requirements.txt`
  - `pip install -r requirements.dev.txt`
- Now we must apply a simple fix to correct a bug in the code. 
  - Open the following scripts in an editor of your choice.
    - \path\to\mujoco-py\scripts\gen_wrappers.py
    - \path\to\mujoco-py\mujoco_py\generated\wrappers.pxi
  - Replace all the instances of
    - `isinstance(addr, (int, np.int32, np.int64))`
  - with
    - `hasattr(addr, '__int__')`
- Return to your Anaconda Prompt in the folder mujoco-py and update files by force compilation:
  - `python -c "import mujoco_py"`
- Now you can install mujoco-py with
  - `python setup.py install`

### Install Gym
To do a full installation of gym with ToyText, MujoCo etc. run the following code:
- `pip install gym[all]`

### Testing Gym
Throughout this section compare with the screenshot collage [`gym_test.jpg`](gym_test.jpg).

Run the test scripts to ensure Gym runs properly
- [`gym_test_classic.py`](gym_test_classic.py)
- [`gym_test_mujoco.py`](gym_test_mujoco.py)

Run the test command to ensure Spinning Up runs on MuJoCo Gym environment properly
- `python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v2 --exp_name mujocotest`
