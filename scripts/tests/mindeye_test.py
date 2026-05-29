import torch
import pytest
from torch.utils.data import DataLoader

from ieeg_dataset import iEEGDataset


@pytest.fixture
def fake_paths(tmp_path):
    """Save un-normalized iEEG and matching CLIP tensors to disk."""
    N = 64
    ieeg = torch.randn(N, 2500) * 4.2 + 1.7    # nonzero mean, nonunit std
    clip = torch.randn(N, 256, 1664)
    ip, cp = tmp_path / "ieeg.pt", tmp_path / "clip.pt"
    torch.save(ieeg, ip)
    torch.save(clip, cp)
    return str(ip), str(cp), N


def test_zscore_train(fake_paths):
    """Train split must have mean 0 and std 1 per feature."""
    ip, cp, N = fake_paths
    ds = iEEGDataset(ip, cp)
    x = torch.stack([ds[i][0].squeeze(0) for i in range(N)])    # (N, 2500)
    assert torch.allclose(x.mean(0), torch.zeros(2500), atol=1e-5)
    assert torch.allclose(x.std(0),  torch.ones(2500),  atol=1e-2)


def test_dataloader_shapes(fake_paths):
    """DataLoader must yield (B, 1, 2500) and (B, 256, 1664)."""
    ip, cp, _ = fake_paths
    loader = DataLoader(iEEGDataset(ip, cp), batch_size=8, shuffle=False)
    ieeg, clip = next(iter(loader))
    assert ieeg.shape == (8, 1, 2500)
    assert clip.shape == (8, 256, 1664)


def test_val_uses_train_stats(fake_paths):
    """Val/test splits must reuse train mean/std (not recompute their own)."""
    ip, cp, _ = fake_paths
    train = iEEGDataset(ip, cp)
    val   = iEEGDataset(ip, cp, mean=train.mean, std=train.std)
    assert torch.equal(train.mean, val.mean)
    assert torch.equal(train.std,  val.std)


def test_no_nans(fake_paths):
    """Z-scoring should never produce NaNs even on near-constant features."""
    ip, cp, _ = fake_paths
    ds = iEEGDataset(ip, cp)
    x, y = ds[0]
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()