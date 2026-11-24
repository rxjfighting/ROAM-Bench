FROM nvcr.io/nvidia/isaac-sim:5.1.0
WORKDIR /workspace/ROAM-Bench
COPY code/ code/
COPY dataset/ dataset/
RUN /isaac-sim/python.sh -m pip install -e './code[geometry,media]'
WORKDIR /workspace/ROAM-Bench/code
ENTRYPOINT ["/isaac-sim/python.sh", "-m", "roam_bench.cli"]
