"""Hardware integration tests — run only on the real AMD + gfx1100 machine.

Skipped automatically when the environment (venv, GPU, weights) is absent,
e.g. in GitHub Actions CI.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.timeout(600)


def _hw_ready() -> bool:
    try:
        import torch  # noqa: F401
        from course.model import OFFICIAL_PT
        return torch.cuda.is_available() and OFFICIAL_PT.exists()
    except Exception:
        return False


@pytest.mark.skipif(not _hw_ready(), reason="requires AMD gfx1100 lab machine")
class TestLabMachine:
    def test_cpu_is_amd(self):
        from course.setup_checks import cpu_info
        assert cpu_info()["vendor"] == "AuthenticAMD"

    def test_gpu_is_gfx1100(self):
        from course.setup_checks import rocm_info
        assert "gfx1100" in (rocm_info()["gfx"] or [])

    def test_torch_pins(self):
        import torch, torchvision, torchaudio
        assert torch.__version__ == "2.12.0+rocm7.14.1"
        assert torchvision.__version__ == "0.27.0+rocm7.14.1"
        assert torchaudio.__version__ == "2.11.0+rocm7.14.1"

    def test_gpu_real_compute(self):
        import torch
        a = torch.randn(1024, 1024, device="cuda")
        assert torch.isfinite(a @ a).all().item()
        from torchvision.ops import nms
        keep = nms(torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11],
                                 [50, 50, 60, 60]], dtype=torch.float32,
                                device="cuda"),
                   torch.tensor([.9, .8, .7], device="cuda"), 0.5)
        assert keep.tolist() == [0, 2]

    def test_official_model_predicts_80_classes(self):
        from course.model import load_official_yolo26l
        m = load_official_yolo26l()
        assert len(m.names) == 80

    def test_openvino_cpu_plugin_available(self):
        import openvino as ov
        assert "CPU" in ov.Core().available_devices
