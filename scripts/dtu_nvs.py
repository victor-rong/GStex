import argparse
from pathlib import Path
from utils import ExperimentRunner

def run_dtu_experiment(runner, exp_name, inst_name, data_name, data_dir, point_cloud_dir):
    experiment_name = exp_name
    instance_name = inst_name

    data_location = Path(data_dir) / data_name
    ply2d = Path(point_cloud_dir) / data_name / "init_nvs/point_cloud.ply"

    base_params = {
        "pipeline.model.num-random": -1,
        "pipeline.model.init-ply": str(ply2d),
    }
    cur_instance_name = instance_name
    runner.train_experiment("gstex-dtu-nvs", experiment_name, cur_instance_name, data_location, base_params)
    runner.eval_experiment("gstex", experiment_name, cur_instance_name)

def run_dtu(runner, args):
    data_names = [
        "scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69",
        "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122"
    ]
    inst_name = runner.log_name
    for i in range(args.start, args.end):
        data_name = data_names[i]
        exp_name = f"{data_name}_nvs"
        run_dtu_experiment(runner, exp_name, inst_name, data_name, args.data_dir, args.point_cloud_dir)

if __name__ == "__main__":
    # python scripts/dtu_nvs.py --data_dir ./data/dtu --data_dir ./pcd_data/dtu --start 0 --end 15
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=15, type=int)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--point_cloud_dir", required=True, type=str)
    parser.add_argument("--log_dir", default="./results/dtu_nvs", type=str)
    args = parser.parse_args()
    runner = ExperimentRunner(log_dir=args.log_dir)    
    run_dtu(runner, args)
