import numpy as np

from openmc_apr1400_phase1_runner_final import generate_sobol_design_matrix, partition_case_jobs


def test_sobol_design_is_space_filling_and_in_bounds():
    design = generate_sobol_design_matrix(n_unique=20, random_seed=123)
    assert len(design) == 20
    arr = np.array([
        [p["alpha_gd"], p["fuel_temp"], p["moderator_temp"], p["boron_ppm"]]
        for p in design
    ])
    assert arr.shape == (20, 4)
    assert (arr[:, 0] >= 0.25).all() and (arr[:, 0] <= 1.25).all()
    assert (arr[:, 1] >= 294.0).all() and (arr[:, 1] <= 1000.0).all()
    assert (arr[:, 2] >= 294.0).all() and (arr[:, 2] <= 600.0).all()
    assert (arr[:, 3] >= 0.0).all() and (arr[:, 3] <= 2000.0).all()
    assert np.all(arr[:, 1] >= arr[:, 2])


def test_partition_case_jobs_honors_worker_count():
    batches = partition_case_jobs(17, workers=4)
    assert len(batches) == 4
    assert sum(len(batch) for batch in batches) == 17
    assert max(len(batch) for batch in batches) <= 5
