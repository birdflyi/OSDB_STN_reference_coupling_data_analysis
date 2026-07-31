import pandas as pd

from script.ch5_direct_reference_coupling.seed_selection import select_analysis_seed_repos


def test_activity_threshold_keeps_10_and_11_but_drops_9():
    repos = pd.DataFrame(
        {
            "repo_id": [1, 2, 3],
            "repo_name": ["a/db", "b/db", "c/db"],
            "i_pr_rec_cnt": [9, 10, 11],
        }
    )

    selected = select_analysis_seed_repos(repos, threshold=10)

    assert selected["repo_name"].tolist() == ["b/db", "c/db"]
    assert selected["repo_id"].tolist() == ["2", "3"]
