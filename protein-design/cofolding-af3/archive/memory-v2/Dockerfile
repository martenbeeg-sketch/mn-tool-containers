# Archived, versioned follow-up to the active AF3 memory recipe. It is retained
# for provenance and is not part of the active image catalog.
FROM mn-cofolding-af3:3.1.14-cu13

RUN mkdir -p /opt/mn-cofolding-af3-opt/opt_core/kernels/fpf_pallas

COPY --from=af3jax /opt/forward/flashpairformer/af3_flashpairformer \
    /opt/mn-cofolding-af3-opt/flashpairformer/af3_flashpairformer
COPY --from=optcore /opt_core/opt_core/__init__.py \
    /opt/mn-cofolding-af3-opt/opt_core/__init__.py
COPY --from=optcore /opt_core/opt_core/oom.py /opt_core/opt_core/gates.py \
    /opt/mn-cofolding-af3-opt/opt_core/
COPY --from=optcore /opt_core/opt_core/kernels/__init__.py \
    /opt_core/opt_core/kernels/fpf_pallas_serve.py \
    /opt_core/opt_core/kernels/safe_settings.py \
    /opt/mn-cofolding-af3-opt/opt_core/kernels/
COPY --from=optcore /opt_core/opt_core/kernels/fpf_pallas_bias.py \
    /opt_core/opt_core/kernels/fpf_pallas_f32.py \
    /opt/mn-cofolding-af3-opt/opt_core/kernels/
COPY --from=optcore /opt_core/opt_core/kernels/fpf_pallas \
    /opt/mn-cofolding-af3-opt/opt_core/kernels/fpf_pallas

RUN cp /usr/local/bin/cofolding-af3-runner.py /usr/local/bin/cofolding-af3-runner-stock.py
COPY protein-design/cofolding-af3/archive/memory-v2/cofolding-af3-memory-v2-runner.py \
    /usr/local/bin/cofolding-af3-runner.py
RUN chmod +x /usr/local/bin/cofolding-af3-runner.py

ENV MN_COFOLDING_AF3_OPT_MODE=off

LABEL org.mn-cofolding.base-image="mn-cofolding-af3:3.1.14-cu13" \
      org.mn-cofolding.af3-optimization="off-fast-big-3.1.14-memory-v2-trimul-chunk"
