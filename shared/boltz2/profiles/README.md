# Boltz-2 hardware profiles

The stock `Dockerfile` and the single active optimized recipe use the same
image layout for both GPU generations. These files are host-side launch
profiles, not separate images:

```bash
set -a
source shared/boltz2/profiles/rtx5090.env
set +a
```

The RTX 4090 profile is the measured Ada baseline. The RTX 5090 profile
targets Blackwell `sm_120` but remains unvalidated until the 5090 benchmark is
run. `MN_BOLTZ2_N_GPU=1` is the safe default; increase it only for the kit's
validated `big` mode on multiple GPUs in the same host. Networked hosts are
separate inference targets and do not form one shared GPU pool.
