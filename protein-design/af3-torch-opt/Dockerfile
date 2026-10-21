ARG BASE_IMAGE=mn-af3-torch-opt-kit:22bdeed-cu130
FROM ${BASE_IMAGE}

# Portable preprocessing for apps that need to turn per-chain PDB/mmCIF uploads
# into the AF3 JSON v2 template fields consumed by the kit runner.
COPY protein-design/af3-torch-opt/prepare_templates.py /usr/local/bin/af3-torch-prepare-templates
RUN chmod 0555 /usr/local/bin/af3-torch-prepare-templates
