# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


## Thread setting
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import torch
# torch.set_num_interop_threads(3)
# torch.set_num_threads(3)
# multi-socket friendly args
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_utils import run_experiment, load_csf_data
import pandas as pd


if __name__ == "__main__":
    
    num_workers = min(os.cpu_count()-2, 4)
    futures = []
    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for fold in range(20):
            futures.append(executor.submit(run_experiment, fold=fold, data_loader=load_csf_data))

        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as exc:
                print(f"Got error {exc}")


    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results_xval_csf_final_uniformpriors.csv', index=False)
    print(results_df)
