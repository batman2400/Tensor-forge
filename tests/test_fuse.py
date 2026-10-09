"""Fusion helpers: mix endpoints and temperature invariance of the category argmax."""

from __future__ import annotations

import numpy as np

from ml.fuse import mix, scale_binary, scale_multiclass


def test_mix_endpoints_match_each_branch():
    classical = np.array([[0.2, 0.8], [1.0, 0.0]])
    encoder = np.array([[0.6, 0.4], [0.0, 1.0]])
    assert np.allclose(mix(classical, encoder, 0.0), classical)
    assert np.allclose(mix(classical, encoder, 1.0), encoder)


def test_category_temperature_keeps_argmax():
    proba = np.array([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1]])
    for temperature in (0.5, 1.0, 2.0):
        scaled = scale_multiclass(proba, temperature)
        assert np.array_equal(scaled.argmax(axis=1), proba.argmax(axis=1))
        assert np.allclose(scaled.sum(axis=1), 1.0)


def test_binary_temperature_keeps_the_side_of_one_half():
    proba = np.array([0.2, 0.8])
    for temperature in (0.5, 1.0, 2.0):
        scaled = scale_binary(proba, temperature)
        assert np.array_equal(scaled > 0.5, proba > 0.5)
