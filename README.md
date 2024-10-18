# S-LoRA

[Google doc link](https://github.com/S-LoRA/S-LoRA)

If you have a fresh environment (ubuntu) do the following to start docker:

Install docker with [link](https://docs.docker.com/engine/install/ubuntu/)

Install NVIDIA driver with [link](https://ubuntu.com/server/docs/nvidia-drivers-installation)

Install NVIDIA container tool with [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Start docker:
```
sudo systemctl start docker
```
Then go to docker dir, run the following command to start docker 
```
sudo docker run -it --entrypoint /bin/bash --gpus all slora
```
If you want to start another shell inside docker, first check the name of the container with:
```
sudo docker ps
```
It will show the container names, copy the name and do:
```
sudo docker exec -it <container_name> bash
```