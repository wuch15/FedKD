# FedKD

1. Environment Requirements
* Ubuntu 16.04
* Anaconda with Python 3.6.9
* CUDA 10.0
* JAVA jdk1.8.0_121 
* Hadoop 2.9.2-SNAPSHOT
* Horovod 0.19.5

2. Additional Python Dependencies
* datasets==1.18.3
* torch==1.5.0
* transformers==4.9.0
The pip installation may need several minutes if there is no environmental conflicts.

2. Hardware requirements
Needs a server with at least one Tesla V100 GPU, while a larger number of GPUs (e.g., 4) is preferred.

3. Training and Testing
* For the MIND dataset, need to download it from the official site https://msnews.github.io/. The other datasets can be automatically downloaded by the datasets library. The pretrained language models can be downloaded from huggingface automatically.

Note: The logs at the training stage will show the training loss and accuracy. Logs at the test stage will show the test results. 

