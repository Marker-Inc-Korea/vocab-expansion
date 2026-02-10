import torch
from adaptivocab_expand.init import softmax_weights


def test_softmax_increasing():
    w = softmax_weights(4, alpha=2.0, sign=+1.0, device="cpu")
    assert torch.all(w[1:] >= w[:-1])


def test_softmax_decreasing():
    w = softmax_weights(4, alpha=2.0, sign=-1.0, device="cpu")
    assert torch.all(w[1:] <= w[:-1])
