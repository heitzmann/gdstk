FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    zlib1g-dev \
    libqhull-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && pip3 install numpy pytest build setuptools wheel

WORKDIR /usr/src/app

COPY . .

CMD ["bash"]