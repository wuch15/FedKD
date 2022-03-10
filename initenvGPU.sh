#!/bin/bash
export JAVA_HOME=/usr/jdk/jdk1.8.0_121 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/lib/server
export HADOOP_HOME=/usr/local/hadoop
export PATH=${PATH}:${HADOOP_HOME}/bin:${JAVA_HOME}/bin
export LIBRARY_PATH=${LIBRARY_PATH}:${HADOOP_HOME}/lib/native
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native:/usr/local/cuda-10.0/extras/CUPTI/lib64
source $HADOOP_HOME/libexec/hadoop-config.sh
export PYTHONPATH="/opt/conda/lib/python3.6/site-packages"
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)