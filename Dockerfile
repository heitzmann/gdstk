FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libqhull-dev \
    git \
    wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.26
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.0/cmake-3.26.0-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-3.26.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-3.26.0-linux-x86_64.sh

RUN python3 -m pip install --upgrade pip \
    && pip3 install numpy pytest build

WORKDIR /usr/src/app

COPY . .

CMD ["bash"]
