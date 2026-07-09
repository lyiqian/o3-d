
# clone the repo
git clone https://github.com/haotian-liu/LLaVA.git

# build the docker
docker/build_docker.sh 

# run the docker on two GPUs (one is not enough)
docker/run_docker.sh -gd 2,3

# inside the docker run
python3 run_llava.py