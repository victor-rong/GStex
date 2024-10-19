import subprocess
import json
from datetime import datetime
import torch
from pathlib import Path

def get_formatted_time():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

class ExperimentRunner():
    def __init__(self, log_dir, log_name=None):
        if log_name is None:
            self.log_name = get_formatted_time()
        else:
            self.log_name = log_name
        self.log_dir = Path(log_dir) / self.log_name
        self.detailed_log_dir = self.log_dir / "logs"
        self.detailed_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_info = []
        self.cmds = []

    def update_log(self):
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        with open(self.log_dir / "log.json", "w") as f:
            json.dump(self.log_info, f)

    def update_cmds(self):
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        with open(self.log_dir / "cmds.sh", "w") as f:
            f.write("\n\n".join(self.cmds))

    def train_experiment(self, method, experiment_name, instance_name, data_dir, params):
        data_dir = str(data_dir)
        cmd_str = f"ns-train {method} --experiment-name {experiment_name} --timestamp {instance_name}"
        for k in params:
            cmd_str += f" --{k} {params[k]}"
        cmd_str += f" --data {data_dir}"

        log_location = self.detailed_log_dir / f"{experiment_name}__{instance_name}_train.log"
        cmd_str += f" > {log_location}"
        self.cmds.append(cmd_str)
        self.update_cmds()

        experiment_info = {
            "experiment_name": experiment_name,
            "method": method,
            "instance_name": instance_name,
            "data_dir": data_dir,
        }
        experiment_info.update(params)
        self.log_info.append(experiment_info)
        self.update_log()
        subprocess.call(cmd_str, shell=True)
        torch.cuda.empty_cache()

    def eval_experiment(self, method, experiment_name, instance_name):
        instance_location = Path("./outputs") / experiment_name / method / instance_name / "config.yml"
        output_dir = Path("./renders") / experiment_name / method / instance_name
        render_location = output_dir / "images"
        output_location = output_dir / "output.json"
        cmd_str = f"ns-eval --load-config {instance_location} --output-path {output_location} --render-output-path {render_location}"
        log_location = self.detailed_log_dir / f"{experiment_name}__{instance_name}_eval.log"
        cmd_str += f" > {log_location}"
        self.cmds.append(cmd_str)
        self.update_cmds()

        subprocess.call(cmd_str, shell=True)
        torch.cuda.empty_cache()

        with open(output_location, "r") as f:
            eval_info = json.load(f)
        abridged_keys = ["psnr", "ssim", "lpips", "gaussian_count", "texel_count", "pixel_scale", "fps"]
        abridged_eval_info = {}
        for k in abridged_keys:
            if k in eval_info["results"]:
                abridged_eval_info[k] = eval_info["results"][k]
        self.log_info[-1]["eval"] = abridged_eval_info
        self.update_log()