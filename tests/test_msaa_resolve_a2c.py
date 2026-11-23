import pytest
import graphics.msaa_resolve_a2c as mod

CASES = [
    ("sample_count", (1)),
    ("a2c_compat", (1)),
    ("depth_color_order", (1)),
    ("per_eye", (1)),
    ("resolve_sync", (1)),
    ("edge_weight", (1)),
    ("resolve", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("alpha_edge", (1)),
    ("metrics", (1)),
    ("format", (1)),
    ("flicker", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
