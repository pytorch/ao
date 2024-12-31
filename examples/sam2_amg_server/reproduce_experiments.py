import subprocess
import os
import fire
import json
import pandas as pd
from pathlib import Path
from compare_rle_lists import compare as compare_folders

def run_script_with_args(positional_args, keyword_args, dry=False, environ=None):
    assert isinstance(positional_args, list)
    assert isinstance(keyword_args, dict)
    # Construct the command with the base arguments
    command = ['python'] + positional_args
    
    # Add any additional arguments in the '--arg value' style
    for arg, value in keyword_args.items():
        if value is None:
            command.extend([f'--{arg}'])
        else:
            command.extend([f'--{arg}', str(value)])
    try:
        env_vars = os.environ.copy()
        if environ is None:
            print(" ".join(command))
        else:
            environ = environ | {'TORCH_LOGS': "recompiles"}
            print(" ".join([f"{k}={environ[k]}" for k in environ] + command))
            env_vars = env_vars | environ
        if dry:
            return None, None
        # Run the command
        result = subprocess.run(command, env=env_vars, check=True, capture_output=True, text=True)
        # Print the output
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        import pdb
        pdb.set_trace()
        return None, None

# TODO: Need experiments to measure time it takes to max-autotune

# python generate_data.py 
# ~/checkpoints/sam2
# large
# amg 
# ~/blogs/sam2_amg_example/sav_val_image_paths_shuf_1000 
# output_data_ao_sps  
# --meta-folder ~/blogs/sam2_amg_example/annotated_images_baseline 
# --overwrite 
# --furious 
# --num-images 1000 
# --points-per-batch 1024


def main(image_paths, output_base_path, dry=False, overwrite=False):
    output_base_path = Path(output_base_path)
    print("output_base_path: ", output_base_path)
    result_csv_path = output_base_path / "result.csv"
    if not dry and not overwrite and result_csv_path.exists():
        raise ValueError(f"Expected {result_csv_path} to not exist. Use --overwrite to overwrite.")
    # output_base_path = "~/blogs/sam2_amg_example"
    # image_paths = f"{output_base_path}/sav_val_image_paths_shuf_1000"

    results = []

    def run(task, output_path: Path, kwargs, baseline_folder=None, environ=None):
        if baseline_folder is not None:
            baseline_folder = output_base_path / baseline_folder
        output_path = output_base_path / output_path
        all_stats_file = Path(str(output_path) + "_stats.json")
        if not dry:
            output_path.mkdir(exist_ok=overwrite)
        if overwrite:
            kwargs = kwargs | {"overwrite": None}
        stdout, stderr = run_script_with_args(["generate_data.py",
                                               "~/checkpoints/sam2",
                                               "large",
                                               task,
                                               image_paths,
                                               str(output_path)],
                                              kwargs | {"quiet": None},
                                              dry=dry,
                                              environ=environ)
        if stdout is not None:
            with open(str(output_path) + ".stdout", 'w') as file:
                file.write(stdout)
        if stderr is not None:
            with open(str(output_path) + ".stderr", 'w') as file:
                file.write(stderr)
        if dry and all_stats_file.exists():
            with open(str(all_stats_file), 'r') as file:
                all_stats = json.load(file)
            results.append(all_stats)
            return all_stats
        if dry:
            return {}

        all_stats = json.loads(stdout.split("\n")[-2])
        if baseline_folder is not None:
            miou_count, miou_sum, fail_count = compare_folders(str(output_path),
                                                               str(baseline_folder),
                                                               strict=True,
                                                               compare_folders=True)
            all_stats["miou"] = miou_sum / miou_count
            all_stats["fail_count"] = fail_count
        all_stats["task"] = task
        all_stats["experiment_name"] = output_path.name
        all_stats = all_stats | {key: str(kwargs[key]) for key in kwargs}
        if not overwrite and all_stats_file.exists():
            raise ValueError(f"{all_stats_file} already exists. Use --overwrite to overwrite.")
        with open(all_stats_file, 'w') as file:
            file.write(json.dumps(all_stats, indent=4))

        # TODO:: Save this and use overwrite to check for it before writing
        results.append(all_stats)
        return all_stats

    def run_annotate(image_paths, baseline_folder: Path, output_path: Path, overwrite, dry):
        output_path  = output_base_path / output_path
        baseline_folder = output_base_path / output_base_path
        if not dry:
            output_path.mkdir(exist_ok=overwrite)
        stdout, stderr = run_script_with_args(["annotate_with_rle.py",
                                               "~/checkpoints/sam2",
                                               "large",
                                               image_paths,
                                               str(baseline_folder),
                                               str(output_path)],
                                              {},
                                              dry=dry)

    # TODO: Something about sps + torch.compile is messed up

    for ttype in ["amg", "sps", "mps"]:
        meta_kwarg = {} if ttype == "amg" else {"meta-folder": output_base_path / "amg_baseline_annotations"}
        # Generate baseline data
        ppb = {'amg': 64, 'sps': 1, 'mps': None}[ttype]
        ppb_kwarg = {} if ppb is None else {"points-per-batch": ppb}
        run(ttype, f"baseline_{ttype}", {'baseline': None} | ppb_kwarg | meta_kwarg)

        def run_with_compare(task, output_path: Path, kwargs, environ=None):
            return run(task, output_path, kwargs, baseline_folder=f"baseline_{ttype}", environ=environ)

        # AO version of baseline for sanity check
        run_with_compare(ttype, f"{ttype}_ao", ppb_kwarg | meta_kwarg)

        ppb = {'amg': 1024, 'sps': 1, 'mps': None}[ttype]
        ppb_kwarg = {} if ppb is None else {"points-per-batch": ppb}
        run_with_compare(ttype, f"{ttype}_ao_ppb_1024",                     ppb_kwarg | meta_kwarg)

        environ = {"TORCHINDUCTOR_CACHE_DIR": str(output_base_path / f"{ttype}_inductor_cache_dir")}
        # fast
        run(ttype,              f"{ttype}_ao_ppb_{ppb}_fast_cold",                      {"fast": None} | ppb_kwarg | meta_kwarg, environ=environ)
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast",                           {"fast": None} | ppb_kwarg | meta_kwarg, environ=environ)
        # TODO: Set num images to 0 for export job
        export_model_path = output_base_path / "exported_models" / f"{ttype}_ao_fast"
        if not dry:
            export_model_path.mkdir(exist_ok=overwrite, parents=True)
        run(ttype,              f"{ttype}_ao_ppb_{ppb}_save_export",                    {"num-images": 0,                  "export-model":        str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ)
        environ_load = {"TORCHINDUCTOR_CACHE_DIR": str(output_base_path / f"{ttype}_load_export_inductor_cache_dir")}
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_load_export",                    {                                  "load-exported-model": str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ_load)
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast_export",                    {"fast": None,                     "load-exported-model": str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ)

        # fast and furious
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast_furious_cold",              {"fast": None,    "furious": None} | ppb_kwarg | meta_kwarg, environ=environ)
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast_furious",                   {"fast": None,    "furious": None} | ppb_kwarg | meta_kwarg, environ=environ)
        # TODO: Set num images to 0 for export job
        export_model_path = output_base_path / "exported_models" / f"{ttype}_ao_fast_furious"
        if not dry:
            export_model_path.mkdir(exist_ok=overwrite, parents=True)
        run(ttype,              f"{ttype}_ao_ppb_{ppb}_save_export_furious",            {"num-images": 0, "furious": None, "export-model":        str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ)
        environ_load = {"TORCHINDUCTOR_CACHE_DIR": str(output_base_path / f"{ttype}_load_export_furious_inductor_cache_dir")}
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_load_export_furious",            {                 "furious": None, "load-exported-model": str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ_load)
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast_export_furious",            {"fast": None,    "furious": None, "load-exported-model": str(export_model_path)} | ppb_kwarg | meta_kwarg, environ=environ)
        run_with_compare(ttype, f"{ttype}_ao_ppb_{ppb}_fast_export_furious_recompiles", {"fast": None,    "furious": None, "load-exported-model": str(export_model_path),  "allow-recompiles": None} | ppb_kwarg | meta_kwarg, environ=environ)

        # TODO: Add a job that uses torchvision for I/O

        # Annotations for baseline folder
        run_annotate(image_paths,  f"baseline_{ttype}", f"{ttype}_baseline_annotations", overwrite, dry)

    all_keys = set().union(*(d.keys() for d in results))
    normalized_data = [{key: d.get(key, None) for key in all_keys} for d in results]
    df = pd.DataFrame(normalized_data)
    if not dry:
        df.to_csv(result_csv_path, index=False)

if __name__ == "__main__":
    fire.Fire(main)
