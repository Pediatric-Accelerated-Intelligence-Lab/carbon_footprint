# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVFlare job.py (SLOW via extra iterations + sleep):

- extra_no_update_iters=100
- sleep_ms_mean=500, sleep_ms_std=250  (Gaussian per-step delay; clipped at 0)
This slows ALL clients using both mechanisms.
"""

import os
from cifar10_data import cifar10_split

from cifar10_pt_fl import Net
from fedavg_carbon import FedAvg

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_pt_fl.py")


if __name__ == "__main__":
    n_clients = 6
    num_rounds = 10
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_pt_fl.py")

    # split the data into n_clients

    data_path = "/data"
    data_split_dir = os.path.join(data_path, "cifar10_splits")
    cifar10_split(num_sites=n_clients, split_dir=data_split_dir, alpha=1.0, seed=0)
    print(f"Data split directory: {data_split_dir}")

    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="carbon_footprint",
      initial_model=Net(),
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        sim = "--sleep_ms_mean=500 --sleep_ms_std=250 --extra_no_update_iters=100"
        runner = ScriptRunner(script=train_script, script_args=f"--country_iso_code=USA {sim} --split_data_root={data_split_dir} --allow_download")
        job.to(runner, f"site-{i}")

    job.export_job("./job_configs")
    job.simulator_run("runLow", gpu="0")
    
