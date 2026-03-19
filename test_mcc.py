"""Tests for MCCLoss (loss.py)."""

import pytest
import torch

from loss import MCCLoss


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_perfect_mixed_batch():
    """Both classes present, all predicted correctly (batch reduction)."""
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(2, 1, 1, 2)
    y_true = torch.tensor([[1, 0], [0, 1]]).view(2, 1, 1, 2)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_perfect_mixed_sample():
    """Both classes present, all predicted correctly (sample reduction)."""
    criterion = MCCLoss(reduction="sample")
    y_pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(2, 1, 1, 2)
    y_true = torch.tensor([[1, 0], [0, 1]]).view(2, 1, 1, 2)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_perfect_all_ones():
    """All-foreground GT predicted correctly (tp > 0, tn = 0)."""
    criterion = MCCLoss(reduction="sample")
    y_pred = torch.ones(1, 1, 3, 3)
    y_true = torch.ones(1, 1, 3, 3)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_perfect_all_zeros():
    """All-background GT predicted correctly (tp = 0, tn > 0)."""
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.zeros(1, 1, 3, 3)
    y_true = torch.zeros(1, 1, 3, 3)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Degenerate per-sample case
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_degenerate_all_zero_sample_in_mixed_batch():
    """One sample is all-zero GT + pred (degenerate), another has mixed GT.

    The degenerate sample should contribute loss = 0, not loss ≈ 1.
    """
    criterion = MCCLoss(reduction="sample")
    y_pred = torch.tensor([[0.0, 0.0], [1.0, 0.0]]).view(2, 1, 1, 2)
    y_true = torch.tensor([[0.0, 0.0], [1.0, 0.0]]).view(2, 1, 1, 2)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_degenerate_all_one_sample_in_mixed_batch():
    """One sample is all-one GT + pred (degenerate), another has mixed GT."""
    criterion = MCCLoss(reduction="sample")
    y_pred = torch.tensor([[1.0, 1.0], [1.0, 0.0]]).view(2, 1, 1, 2)
    y_true = torch.tensor([[1.0, 1.0], [1.0, 0.0]]).view(2, 1, 1, 2)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Worst-case predictions
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_worst_case():
    """Completely inverted predictions → MCC = -1, loss = 2."""
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 2, 2)
    y_true = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 2, 2)
    assert criterion(y_pred, y_true).item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Logits
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_from_logits_perfect():
    """High-magnitude logits matching GT should give loss ≈ 0."""
    criterion = MCCLoss(from_logits=True)
    y_pred = torch.tensor([10.0, -10.0, 10.0]).view(1, 1, 1, 3)
    y_true = torch.tensor([1, 0, 1]).view(1, 1, 1, 3)
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-3)


@torch.no_grad()
def test_from_logits_inverted():
    """High-magnitude logits inverted from GT should give loss ≈ 2."""
    criterion = MCCLoss(from_logits=True)
    y_pred = torch.tensor([-10.0, 10.0, -10.0]).view(1, 1, 1, 3)
    y_true = torch.tensor([1, 0, 1]).view(1, 1, 1, 3)
    assert criterion(y_pred, y_true).item() == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_input_2d():
    """2D inputs (N, L) should be accepted."""
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    y_true = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


@torch.no_grad()
def test_input_3d():
    """3D inputs (N, H, W) should be accepted."""
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert criterion(y_pred, y_true).item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Reduction equivalence
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_single_sample_reduction_equivalence():
    """With one sample, batch and sample reductions must agree."""
    torch.manual_seed(42)
    y_pred = torch.rand(1, 1, 8, 8)
    y_true = torch.bernoulli(torch.rand(1, 1, 8, 8))
    loss_batch = MCCLoss(reduction="batch")(y_pred, y_true).item()
    loss_sample = MCCLoss(reduction="sample")(y_pred, y_true).item()
    assert loss_batch == pytest.approx(loss_sample, abs=1e-5)


# ---------------------------------------------------------------------------
# Random inputs — sanity-check the loss range
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_random_loss_range_sample():
    criterion = MCCLoss(reduction="sample")
    y_pred = torch.bernoulli(torch.rand(4, 1, 5, 5))
    y_true = torch.bernoulli(torch.rand(4, 1, 5, 5))
    loss = criterion(y_pred, y_true).item()
    assert 0.0 <= loss <= 2.0


@torch.no_grad()
def test_random_loss_range_batch():
    criterion = MCCLoss(reduction="batch")
    y_pred = torch.bernoulli(torch.rand(4, 1, 5, 5))
    y_true = torch.bernoulli(torch.rand(4, 1, 5, 5))
    loss = criterion(y_pred, y_true).item()
    assert 0.0 <= loss <= 2.0


# ---------------------------------------------------------------------------
# Soft (non-binary) predictions
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_soft_predictions():
    """Soft probabilities should give a loss strictly between 0 and 2."""
    criterion = MCCLoss(reduction="batch")
    torch.manual_seed(0)
    y_pred = torch.rand(2, 1, 8, 8)
    y_true = torch.bernoulli(torch.rand(2, 1, 8, 8))
    loss = criterion(y_pred, y_true).item()
    assert 0.0 < loss < 2.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flow():
    """Backward pass should produce finite, non-NaN gradients."""
    criterion = MCCLoss(from_logits=True)
    y_pred = torch.randn(2, 1, 8, 8, requires_grad=True)
    y_true = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = criterion(y_pred, y_true)
    loss.backward()
    assert y_pred.grad is not None
    assert not torch.isnan(y_pred.grad).any()
    assert not torch.isinf(y_pred.grad).any()


def test_gradient_degenerate_all_ones():
    """Gradient must not be NaN when GT and pred are all-foreground."""
    criterion = MCCLoss(from_logits=True)
    y_pred = torch.full((1, 1, 4, 4), 10.0, requires_grad=True)
    y_true = torch.ones(1, 1, 4, 4)
    loss = criterion(y_pred, y_true)
    loss.backward()
    assert not torch.isnan(y_pred.grad).any()


def test_gradient_degenerate_all_zeros():
    """Gradient must not be NaN when GT and pred are all-background."""
    criterion = MCCLoss(from_logits=True)
    y_pred = torch.full((1, 1, 4, 4), -10.0, requires_grad=True)
    y_true = torch.zeros(1, 1, 4, 4)
    loss = criterion(y_pred, y_true)
    loss.backward()
    assert not torch.isnan(y_pred.grad).any()


# ---------------------------------------------------------------------------
# sklearn comparison
# ---------------------------------------------------------------------------


def test_matches_sklearn():
    """Loss should match 1 - sklearn.matthews_corrcoef for hard predictions."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    criterion = MCCLoss(reduction="batch")
    y_true_np = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred_np = [1, 0, 0, 1, 0, 1, 1, 0]

    y_true = torch.tensor(y_true_np, dtype=torch.float).unsqueeze(0)
    y_pred = torch.tensor(y_pred_np, dtype=torch.float).unsqueeze(0)

    our_mcc = 1.0 - criterion(y_pred, y_true).item()
    ref_mcc = sklearn_metrics.matthews_corrcoef(y_true_np, y_pred_np)
    assert our_mcc == pytest.approx(ref_mcc, abs=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_invalid_reduction():
    with pytest.raises(ValueError, match="reduction"):
        MCCLoss(reduction="mean")


@torch.no_grad()
def test_shape_mismatch():
    with pytest.raises(ValueError, match="[Ss]hape"):
        MCCLoss()(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 5))


@torch.no_grad()
def test_1d_input_rejected():
    with pytest.raises(ValueError, match="2D"):
        MCCLoss()(torch.rand(4), torch.rand(4))
