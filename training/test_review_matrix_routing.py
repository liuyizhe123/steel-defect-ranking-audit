# -*- coding: utf-8 -*-
"""Regression test for review-matrix metadata versus FWN overrides."""

from run_review_matrix import JOBS, configure_fwn, fwn_overrides_for_job


def main() -> None:
    interaction_fwn = [
        job
        for job in JOBS
        if job["group"] == "interaction_seed_replication" and job["loss"] == "fwn"
    ]
    sensitivity = [job for job in JOBS if job["group"] == "fwn_sensitivity"]
    assert len(interaction_fwn) == 4
    assert len(sensitivity) == 4

    for job in JOBS:
        config = configure_fwn(fwn_overrides_for_job(job))
        assert config["k"] > 0
        assert 0 < config["s0"] < 1

    for job in sensitivity:
        config = configure_fwn(fwn_overrides_for_job(job))
        assert job["params"]
        assert set(job["params"]).issubset(
            {"FWN_K", "FWN_S0", "FWN_U", "FWN_V", "FWN_WIOU_ALPHA", "FWN_WIOU_DELTA", "FWN_NWD_C"}
        )

    assert all(fwn_overrides_for_job(job) == {} for job in interaction_fwn)
    print("REVIEW_MATRIX_ROUTING_OK")


if __name__ == "__main__":
    main()
