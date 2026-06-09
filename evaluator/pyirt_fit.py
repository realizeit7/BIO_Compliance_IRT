"""
evaluator/pyirt_fit.py — 1PL (Rasch) fit via py-irt, with posterior uncertainty.

Companion to evaluator/rasch.py. Where girth's `rasch_mml` returns point
estimates of item difficulty only, py-irt fits the same 1PL model with
black-box variational inference (Pyro) and exposes a *posterior standard
deviation* per item — the uncertainty the paper reports as b ± SE.

  * loc_diff   (Pyro param store) -> posterior mean difficulty  -> b_mean
  * scale_diff (Pyro param store) -> posterior SD of difficulty  -> b_se

Baseline anchoring matches rasch.py exactly: in the 1PL model
P(correct | theta, b) = sigma(theta - b), so shifting every b by a constant c
shifts every theta by -c. We pick c = theta_baseline so the baseline
examinee's ability is 0 by definition; b_se is invariant to this shift.

DATA DEPENDENCY: this needs the raw binary ResponseMatrix, i.e.
logs/arena_runs/phase1_baseline/responses.jsonl must exist (regenerate via
run_phase1.py if absent). It cannot be reconstructed from the per-item
summary outputs in evaluator/output/.

No API calls. Fit on the 1,284 x 45 matrix is a few seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluator.response_matrix import ResponseMatrix


@dataclass
class PyIrtFit:
    """1PL calibration result with posterior uncertainty, aligned to a ResponseMatrix."""

    item_ids: list[str]
    b_mean: np.ndarray  # shape (n_items,), posterior mean difficulty (anchored if anchored=True)
    b_se: np.ndarray  # shape (n_items,), posterior SD of difficulty (scale_diff)
    baseline_examinee_id: str | None = None
    theta_baseline_raw: float | None = None
    anchored: bool = False


def _prepare_matrix(rm: ResponseMatrix) -> np.ndarray:
    """py-irt needs a complete 0/1 matrix; map the -1 'missing' sentinel to 0."""
    mat = rm.matrix.copy()
    mat[mat < 0] = 0
    return mat.astype(np.int8)


def fit_pyirt(
    rm: ResponseMatrix,
    *,
    epochs: int = 2000,
    seed: int = 42,
    device: str = "cpu",
) -> PyIrtFit:
    """
    Fit a 1PL model via py-irt and return per-item posterior mean + SD.

    Imports of py_irt/pyro are deferred so that importing this module is cheap
    and does not pull in torch unless a fit is actually requested.
    """
    import pandas as pd
    import pyro
    from py_irt.config import IrtConfig
    from py_irt.dataset import Dataset
    from py_irt.training import IrtModelTrainer

    mat = _prepare_matrix(rm)  # (n_items, n_examinees)

    # py-irt wants subjects (examinees) as rows, items as columns.
    wide = pd.DataFrame(
        mat.T,
        index=list(rm.examinee_ids),
        columns=list(rm.item_ids),
    )
    wide.index.name = "subject_id"
    wide = wide.reset_index()
    item_cols = list(rm.item_ids)

    dataset = Dataset.from_pandas(
        wide, subject_column="subject_id", item_columns=item_cols
    )

    pyro.clear_param_store()
    config = IrtConfig(model_type="1pl", epochs=epochs, seed=seed, log_every=epochs + 1)
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(device=device)

    params = trainer.best_params
    # params["item_ids"] is a dict {column_index: item_id_string}; diff and
    # scale_diff are aligned to that integer index. Map index -> id, then id -> value.
    ix_to_id = params["item_ids"]
    diff = np.asarray(params["diff"]).ravel()
    b_by_id = {ix_to_id[ix]: float(diff[ix]) for ix in ix_to_id}

    store = pyro.get_param_store()
    scale_diff = np.asarray(store["scale_diff"].detach().cpu().numpy()).ravel()
    se_by_id = {ix_to_id[ix]: float(scale_diff[ix]) for ix in ix_to_id}

    ordered_ids = list(rm.item_ids)
    b_mean = np.array([b_by_id[i] for i in ordered_ids], dtype=float)
    b_se = np.array([se_by_id[i] for i in ordered_ids], dtype=float)

    return PyIrtFit(item_ids=ordered_ids, b_mean=b_mean, b_se=b_se)


def anchor_baseline_to_zero(
    fit: PyIrtFit,
    rm: ResponseMatrix,
    baseline_examinee_id: str,
) -> PyIrtFit:
    """
    Shift b_mean so the baseline examinee's MLE theta equals 0 (b_se unchanged).

    Uses girth.ability_mle on the baseline response vector against the fitted
    b_mean, exactly mirroring evaluator/rasch.anchor_baseline_to_zero.
    """
    from girth import ability_mle

    baseline_responses = rm.baseline_responses(baseline_examinee_id)
    single = baseline_responses.reshape(-1, 1)
    discrimination = np.ones_like(fit.b_mean)
    theta = ability_mle(single, fit.b_mean, discrimination)
    theta_scalar = float(np.atleast_1d(theta)[0])

    if not np.isfinite(theta_scalar):
        return PyIrtFit(
            item_ids=fit.item_ids,
            b_mean=fit.b_mean.copy(),
            b_se=fit.b_se.copy(),
            baseline_examinee_id=baseline_examinee_id,
            theta_baseline_raw=theta_scalar,
            anchored=False,
        )

    return PyIrtFit(
        item_ids=fit.item_ids,
        b_mean=fit.b_mean - theta_scalar,
        b_se=fit.b_se.copy(),
        baseline_examinee_id=baseline_examinee_id,
        theta_baseline_raw=theta_scalar,
        anchored=True,
    )
