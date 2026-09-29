# Base image (as in masif_seed instructions)
FROM pymesh/pymesh:latest

ARG MASIF_SEED_COMMIT=77153911d8252d02193a6d1d6d0539bf02c5faba

# ---- Fix apt repos (base image uses Debian buster which is EOL -> 404 on normal mirrors) ----
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list \
 && sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list \
 && sed -i 's|buster-updates|buster|g' /etc/apt/sources.list \
 && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99no-check-valid-until

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget git unzip cmake vim libgl1-mesa-glx dssp curl \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Ensure we consistently use Python 3.6 from /usr/local (this image has multiple Pythons)
# We make /usr/local/bin/python be python3.6 so "python" is stable for scripts.
RUN ln -sf /usr/local/bin/python3.6 /usr/local/bin/python

# Install pip for Python 3.6 (and only use python -m pip afterwards)
RUN curl -fsSL https://bootstrap.pypa.io/pip/3.6/get-pip.py -o /tmp/get-pip.py \
 && /usr/local/bin/python3.6 /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py \
 && /usr/local/bin/python3.6 -m pip --version

# ---- APBS / PDB2PQR build ----
RUN mkdir -p /install
WORKDIR /install
RUN git clone https://github.com/Electrostatics/apbs-pdb2pqr

WORKDIR /install/apbs-pdb2pqr
RUN git checkout b3bfeec \
 && git submodule init \
 && git submodule update

# Build APBS (+ msms download via CMake option)
RUN cmake -DGET_MSMS=ON apbs \
 && make -j"$(nproc)" \
 && make install

# Copy msms binary bundle where masif_seed expects it
RUN mkdir -p /root/msms \
 && cp -r /install/apbs-pdb2pqr/apbs/externals/mesh_routines/msms/msms_i86_64Linux2_2.6.1 /root/msms/

# INSTALL PDB2PQR (python2.7)
WORKDIR /install/apbs-pdb2pqr/pdb2pqr
RUN git checkout b3bfeec \
 && python2.7 scons/scons.py install

# Setup environment variables
ENV MSMS_BIN=/usr/local/bin/msms
ENV APBS_BIN=/usr/local/bin/apbs
ENV MULTIVALUE_BIN=/usr/local/share/apbs/tools/bin/multivalue
ENV PDB2PQR_BIN=/root/pdb2pqr/pdb2pqr.py

# ---- Reduce (for protonation) ----
WORKDIR /install
RUN git clone https://github.com/rlabduke/reduce.git

WORKDIR /install/reduce
RUN make install

RUN mkdir -p /install/reduce/build/reduce
WORKDIR /install/reduce/build/reduce
RUN cmake /install/reduce/reduce_src

WORKDIR /install/reduce/reduce_src
RUN make -j"$(nproc)" \
 && make install

# ---- Python libs (install into the SAME interpreter used by scripts) ----
RUN /usr/local/bin/python3.6 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN /usr/local/bin/python3.6 -m pip install --no-cache-dir \
    requests \
    matplotlib \
    ipython \
    biopython \
    scikit-learn \
    tensorflow==1.12 \
    networkx \
    open3d==0.8.0.0 \
    dask==1.2.2 \
    packaging

COPY scripts/fetch-pinned-source.sh /usr/local/bin/mn-fetch-source
COPY patches/masif-seed.patch /opt/mn-tool-patches/masif-seed.patch
RUN chmod +x /usr/local/bin/mn-fetch-source \
 && mn-fetch-source https://github.com/LPDI-EPFL/masif_seed.git "${MASIF_SEED_COMMIT}" /opt/masif_seed masif-seed.patch

# Optional sanity check (fails build early if env is inconsistent)
RUN /usr/local/bin/python3.6 -c "import sys; import requests; import Bio; print(sys.version); print('requests', requests.__version__); print('biopython', Bio.__version__)"

WORKDIR /opt/masif_seed

CMD ["bash"]
