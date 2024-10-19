import argparse
from pathlib import Path
from utils import ExperimentRunner

def run_blender_experiment(runner, exp_name, inst_name, data_name, data_dir, point_cloud_dir, num_pts):
    experiment_name = exp_name
    instance_name = inst_name

    data_location = Path(data_dir) / data_name
    ply_lod = Path(point_cloud_dir) / data_name / f"init_lod/pcd_{num_pts}.ply"

    base_params = {
        "pipeline.model.num-random": -1,
        "pipeline.model.init-lod-ply": str(ply_lod),
    }

    cur_instance_name = instance_name
    runner.train_experiment("gstex-blender-lod", experiment_name, cur_instance_name, data_location, base_params)
    runner.eval_experiment("gstex", experiment_name, cur_instance_name)

def run_blender(runner, args):
    data_names = [
        "chair", "drums", "ficus", "hotdog",
        "lego", "materials", "mic", "ship",
    ]
    nums = [128, 512, 2048, 8192, 32768]
    inst_name = runner.log_name
    for i in range(args.start, args.end):
        for num_pts in nums:
            data_name = data_names[i]
            exp_name = f"{data_name}_lod_{num_pts}"
            run_blender_experiment(runner, exp_name, inst_name, data_name, args.data_dir, args.point_cloud_dir, num_pts)


if __name__ == "__main__":
    # python scripts/blender_lod.py --data_dir ./data/blender --point_cloud_dir ./pcd_data/blender --start 0 --end 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=8, type=int)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--point_cloud_dir", required=True, type=str)
    parser.add_argument("--log_dir", default="./results/blender_lod", type=str)
    args = parser.parse_args()
    runner = ExperimentRunner(log_dir=args.log_dir)
    run_blender(runner, args)