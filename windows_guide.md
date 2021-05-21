# OpenAI Spinning Up Install Guide
# For Windows 10 + Nvidia GTX 1070


These instructions roughly follow the Git issues
https://github.com/openai/spinningup/issues/215
This is in contrast to the earlier Git issue 
https://github.com/openai/spinningup/issues/23
which requires you to use WSL to mimic Linux functionality.


ALWAYS RUN ANACONDA PROMPT AS ADMINSTRATOR 
Right click the app icon and using menu to 'Run as Administrator'

ALWAYS ACTIVATE YOUR ENVIRONMENT BEFORE INSTALLING OR USING PYTHON
From an Anaconda prompt run the command
conda activate spinningup


Anaconda commands cheat sheet
conda env list    ## Show a list of all environments on the systemconda 
conda list        ## Show a list of all installed packages in the currently activated environment


Create a new environment
From an Anaconda prompt run the command
conda create --name spinningup python=3.6
conda activate spinningup


Clone the spinningup Git repo using Git or GitHub Desktop
In File Explorer, navigate into the spinningup Git repo folder and edit setup.py and remove the following lines from the install_requires block
'numpy'
'scipy'
'matplotlib==3.1.1'
'pandas'
'joblib'
'mpi4py'
'pytest'
'psutil'
'tqdm'
'cloud_pickle==1.2.1'
'torch==1.3.1'
'tensorflow>=1.8.0,<2.0'

This should leave only the following:
    install_requires=[
        'gym[atari,box2d,classic_control]~=0.15.3'
    ],
Compare against my copy_this_over___setup.py file in the spinningup-helper repo.


Install standard packages 
We are doing this now to isolate install potential problems from the more complicated packages later
conda install ipython
conda install numpy
conda install scipy
conda install pandas
conda install matplotlib=3.1.1
conda install seaborn=0.8.1


conda install pytest
conda install psutil
conda install joblib
conda install tqdm
conda install mpi4py
conda install cloudpickle=1.2.1



Install TensorFlow
conda install tensorflow">=1.8.0,<2.0"

NOTE: Spinning Up uses old versions of TensorFlow that do not match the current docs in some ways. 
For example when doing the basics at https://www.tensorflow.org/tutorials/customization/basics
you will find that 
ndarray = np.ones([3, 3])
tensor = tf.multiply(ndarray, 42)
print(tensor) 
yields different result - this is because TensorFlow changed how it handles evaluation of the computational graph. 
For the old version we have to start a session and run it to evaluate the result manually
https://stackoverflow.com/questions/46548339/tensorflow-tensormul0-shape-dtype-int32
sess = tf.Session()
sess.run(tensor)


Install PyTorch with CUDA using the official PyTorch instructions at https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

To make sure PyTorch can use CPU and GPU without failing, run my cpu_gpu_test.py script in the spinningup-helper repo.

Now we need to prevent an error with OpenAI Gym, specifically with the Box2D environment which needs swig in order to build properly.
conda install -c anaconda swig
conda install -c conda-forge pybox2d


From here we follow the official OpenAI Spinning Up instructions https://spinningup.openai.com/en/latest/user/installation.html
In an Anaconda Prompt, change directories to the spinningup Git repo
cd [...\spinningup]
Now run the command to install spinningup and its dependencies
pip install -e .

Now run the test script command to train an RL agent using the PPO algorithm.
This will take several minutes to complete.
If it works properly you should see real time update tables of values like 'Epoch ...' 'AverageEpRet ...' and lines like 'Warning: trajectory cut off by epoch at xxx steps.'
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999

Then you can watch a video of the trained policy.
python -m spinup.run test_policy data/installtest/installtest_s0
NOTE: By default test_policy.py will run 100 episodes.
Edit the __main__ of test_policy.py if you dont want to wait around so long.
This will also take severeal minutes to complete (there are multiple episodes).
python -m spinup.run test_policy data/installtest/installtest_s0

And plot the results.
python -m spinup.run plot data\installtest\installtest_s0
NOTE: You MUST use backslashes in the path for the plot.py to run properly, which is the Windows convention.
This contrasts with the Linux convention of forward slashes (which is used throughout Spinning Up).
Otherwise plot.py will get screwed up particularly on the line
logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

