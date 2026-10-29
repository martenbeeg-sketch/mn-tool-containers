"""Small GPU kernel diagnostic, not molecular inference or a scientific result."""
import json
import torch
import torch.nn.functional as F
from protenix.model.layer_norm.layer_norm import FusedLayerNorm, fast_layer_norm_cuda_v2

torch.manual_seed(42)
results = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for width in (128, 256, 384, 768):
        for scale, bias in ((False, False), (True, False), (False, True), (True, True)):
            layer = FusedLayerNorm(width, create_scale=scale, create_offset=bias).cuda()
            x = torch.randn(2, 32, width, device='cuda', dtype=dtype)
            # Separate synchronizations attribute the original failure correctly.
            torch.cuda.synchronize()
            actual = layer(x)
            torch.cuda.synchronize()
            weight = layer.weight.to(dtype) if scale else None
            offset = layer.bias.to(dtype) if bias else None
            expected = F.layer_norm(x, (width,), weight, offset, layer.eps)
            torch.cuda.synchronize()
            tolerance = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            results.append(dict(dtype=str(dtype), width=width, scale=scale, bias=bias,
                                max_abs_error=float((actual - expected).abs().max())))
print(json.dumps(dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                     torch=torch.__version__, extension=fast_layer_norm_cuda_v2.__file__,
                     kernel_cases=len(results), results=results), indent=2))
