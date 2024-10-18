FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openssl \
    bash \
    curl \
    vim \
    sudo \
    iproute2 \
    wget \
    git \
    tmux \
    htop \
    psmisc \
    openssh-server \
    zsh \
    git-lfs \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install \
    net-tools

SHELL ["/bin/zsh", "-c"]
WORKDIR /

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -rm -d /home/${USER_NAME} -s /bin/zsh -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} && \
    chown ${USER_ID} -R /home/${USER_NAME} && \
    usermod -a -G ${GROUP_NAME} ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    openssl rand -base64 16 > /home/${USER_NAME}/password.txt && \
    echo "${USER_NAME}:$(cat /home/${USER_NAME}/password.txt)" | chpasswd

ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/miniconda3
ENV CONDA=${CONDA_PREFIX}/condabin/conda

WORKDIR ${HOME}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}

RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN rm miniconda.sh

RUN ${CONDA} update --all -y

RUN ${CONDA} create -n slora python=3.9

# RUN ${CONDA} install cuda -c nvidia/label/cuda-11.8.0 -y
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"

RUN ${CONDA_PREFIX}/envs/slora/bin/pip install torch==2.0.1

RUN git clone https://github.com/S-LoRA/S-LoRA.git
WORKDIR ${HOME}/S-LoRA

RUN ${CONDA_PREFIX}/envs/slora/bin/pip install -e .
RUN ${CONDA_PREFIX}/envs/slora/bin/pip install numpy==1.23.5

CMD ["sleep", "infinity"]
