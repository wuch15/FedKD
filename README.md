# FedKD

1. Environment Requirements
* Ubuntu 16.04
* Anaconda with Python 3.6.9
* CUDA 10.0
* JAVA jdk1.8.0_121 
* Hadoop 2.9.2-SNAPSHOT
* Horovod 0.19.5

Note: The complete python package list of our environment is included in the requirements.txt.
The installation may need several minutes if there is no environmental conflicts.

2. Hardware requirements
Needs a server with at least one Tesla V100 GPU, while a larger number of GPUs (e.g., 4) is preferred.

3. Training and Testing
* Download datasets and pretrained language models from their original sources
* Change the path names and data file names. If you have K GPUs, need to split the entire training data into K folds.
* Execute "sh run.sh"

Note: The logs at the training stage will show the training loss and accuracy. Logs at the test stage will show the test results. The sample codes usually run for a few minutes.

