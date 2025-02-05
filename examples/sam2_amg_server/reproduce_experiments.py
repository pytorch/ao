import json
import os
import subprocess
import time
from pathlib import Path

import fire
import pandas as pd
import torch
import torchvision
from compare_rle_lists import compare as compare_folders


def cond_mkdir(path, overwrite, resume, dry):
    if not dry:
        path.mkdir(exist_ok=(overwrite or resume), parents=True)


def run_script_with_args(positional_args, keyword_args, dry=False, environ=None):
    assert isinstance(positional_args, list)
    assert isinstance(keyword_args, dict)
    # Construct the command with the base arguments
    command = ["python"] + positional_args

    # Add any additional arguments in the '--arg value' style
    for arg, value in keyword_args.items():
        if value is None:
            command.extend([f"--{arg}"])
        else:
            command.extend([f"--{arg}", str(value)])
    try:
        env_vars = os.environ.copy()
        if environ is None:
            print(" ".join(command))
        else:
            environ = environ | {"TORCH_LOGS": "recompiles"}
            print(" ".join([f"{k}={environ[k]}" for k in environ] + command))
            env_vars = env_vars | environ
        if dry:
            return None, None
        # Run the command
        result = subprocess.run(
            command, env=env_vars, check=True, capture_output=True, text=True
        )
        # Print the output
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None, e.stderr


def main(
    image_paths,
    output_base_path,
    dry=False,
    overwrite=False,
    resume=False,
    batch_size=1,
):
    assert batch_size > 0
    print("batch_size: ", batch_size)
    output_base_path = Path(output_base_path)
    print("output_base_path: ", output_base_path)
    result_csv_path = output_base_path / "result.csv"
    if not dry and not overwrite and result_csv_path.exists():
        raise ValueError(
            f"Expected {result_csv_path} to not exist. Use --overwrite to overwrite."
        )
    # output_base_path = "~/blogs/sam2_amg_example"
    # image_paths = f"{output_base_path}/sav_val_image_paths_shuf_1000"

    results = []

    def run(task, output_path: Path, kwargs, baseline_folder=None, environ=None):
        if baseline_folder is not None:
            baseline_folder = output_base_path / baseline_folder
        output_path = output_base_path / output_path
        all_stats_file = Path(str(output_path) + "_stats.json")
        if all_stats_file.exists() and resume:
            with open(all_stats_file, "r") as file:
                return json.load(file)
        if not dry:
            output_path.mkdir(exist_ok=(overwrite or resume))
        if overwrite:
            kwargs = kwargs | {"overwrite": None}
        run_script_time = time.time()
        stdout, stderr = run_script_with_args(
            [
                "generate_data.py",
                f"{str(Path.home())}/checkpoints/sam2",
                "large",
                task,
                image_paths,
                str(output_path),
            ],
            kwargs | {"quiet": None},
            dry=dry,
            environ=environ,
        )
        run_script_time = time.time() - run_script_time
        if stdout is not None:
            with open(str(output_path) + ".stdout", "w") as file:
                file.write(stdout)
        if stderr is not None:
            with open(str(output_path) + ".stderr", "w") as file:
                file.write(stderr)
        if dry and all_stats_file.exists():
            with open(str(all_stats_file), "r") as file:
                all_stats = json.load(file)
            results.append(all_stats)
            return all_stats
        if dry:
            return {}

        if stdout is None:
            all_stats = {}
            all_stats["likely_failed_stderr"] = str(stderr)
        else:
            all_stats = json.loads(stdout.split("\n")[-2])
        if baseline_folder is not None and (stdout is not None):
            miou_count, miou_sum, fail_count = compare_folders(
                str(output_path),
                str(baseline_folder),
                strict=True,
                compare_folders=True,
            )
            all_stats["miou"] = miou_sum / miou_count
            all_stats["fail_count"] = fail_count
        all_stats["run_script_time"] = run_script_time
        all_stats["task"] = task
        all_stats["experiment_name"] = output_path.name
        all_stats["environ"] = str(environ)
        all_stats["torch_version"] = str(torch.__version__)
        all_stats["torchvision_version"] = str(torchvision.__version__)
        all_stats = all_stats | {key: str(kwargs[key]) for key in kwargs}
        if not overwrite and all_stats_file.exists():
            raise ValueError(
                f"{all_stats_file} already exists. Use --overwrite to overwrite."
            )
        with open(all_stats_file, "w") as file:
            print("\n".join(f"{k}: {all_stats[k]}" for k in sorted(all_stats.keys())))
            file.write(json.dumps(all_stats, indent=4))

        # TODO:: Save this and use overwrite to check for it before writing
        results.append(all_stats)
        return all_stats

    def run_annotate(
        image_paths, baseline_folder: Path, output_path: Path, overwrite, dry
    ):
        output_path = output_base_path / output_path
        baseline_folder = output_base_path / baseline_folder
        if resume and output_path.exists():
            return None, None
        if not dry:
            output_path.mkdir(exist_ok=(overwrite or resume))
        stdout, stderr = run_script_with_args(
            [
                "annotate_with_rle.py",
                "~/checkpoints/sam2",
                "large",
                image_paths,
                str(baseline_folder),
                str(output_path),
            ],
            {},
            dry=dry,
        )

    for ttype in ["amg", "sps", "mps"]:
        meta_kwarg = (
            {}
            if ttype == "amg"
            else {"meta-folder": output_base_path / "amg_baseline_annotations"}
        )
        # Generate baseline data
        ppb = {"amg": 64, "sps": 1, "mps": None}[ttype]
        ppb_kwarg = {} if ppb is None else {"points-per-batch": ppb}
        run(ttype, f"baseline_{ttype}", {"baseline": None} | ppb_kwarg | meta_kwarg)

        def run_with_compare(task, output_path: Path, kwargs, environ=None):
            return run(
                task,
                output_path,
                kwargs,
                baseline_folder=f"baseline_{ttype}",
                environ=environ,
            )

        # AO version of baseline for sanity check
        run_with_compare(ttype, f"{ttype}_ao", ppb_kwarg | meta_kwarg)

        ppb = {"amg": 1024, "sps": 1, "mps": None}[ttype]
        ppb_kwarg = {} if ppb is None else {"points-per-batch": ppb}

        batch_size_kwarg = {"batch-size": batch_size}

        def rwc_curry(output_path: Path, kwargs, environ):
            return run_with_compare(
                ttype,
                f"{ttype}_ao_ppb_{ppb}_" + output_path,
                kwargs | ppb_kwarg | meta_kwarg | batch_size_kwarg,
                environ=environ,
            )

        def r_curry(output_path: Path, kwargs, environ):
            return run(
                ttype,
                f"{ttype}_ao_ppb_{ppb}_" + output_path,
                kwargs | ppb_kwarg | meta_kwarg | batch_size_kwarg,
                environ=environ,
            )

        rwc_curry("basic", {}, None)

        environ = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_inductor_cache_dir"
            )
        }
        # fast
        r_curry("fast_cold", {"fast": None}, environ)
        rwc_curry("fast", {"fast": None}, environ)
        export_model_path = output_base_path / "exported_models" / f"{ttype}_ao_fast"
        cond_mkdir(export_model_path, overwrite, resume, dry)
        r_curry(
            "save_export",
            {"num-images": 0, "export-model": str(export_model_path)},
            environ,
        )

        environ_load = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_load_export_inductor_cache_dir"
            )
        }
        load_export_kwarg = {"load-exported-model": str(export_model_path)}
        rwc_curry("load_export_cold", load_export_kwarg, environ_load)
        rwc_curry("load_export", load_export_kwarg, environ_load)
        rwc_curry(
            "load_export_gpu_preproc",
            load_export_kwarg | {"gpu-preproc": None},
            environ_load,
        )

        environ_fast_load = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_fast_export_inductor_cache_dir"
            )
        }
        fast_load_kwarg = {"fast": None, "load-exported-model": str(export_model_path)}
        rwc_curry("fast_export_cold", fast_load_kwarg, environ_fast_load)
        rwc_curry("fast_export", fast_load_kwarg, environ_fast_load)
        rwc_curry(
            "fast_export_gpu_preproc",
            fast_load_kwarg | {"gpu-preproc": None},
            environ_fast_load,
        )

        environ_furious = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_furious_inductor_cache_dir"
            )
        }
        # fast and furious
        fast_furious_kwarg = {"fast": None, "furious": None}
        rwc_curry("fast_furious_cold", fast_furious_kwarg, environ_furious)
        rwc_curry("fast_furious", fast_furious_kwarg, environ_furious)

        export_model_path = (
            output_base_path / "exported_models" / f"{ttype}_ao_fast_furious"
        )
        cond_mkdir(export_model_path, overwrite, resume, dry)
        r_curry(
            "save_export_furious",
            {"num-images": 0, "furious": None, "export-model": str(export_model_path)},
            environ_furious,
        )

        environ_load = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_load_export_furious_inductor_cache_dir"
            )
        }
        furious_load_kwarg = {
            "furious": None,
            "load-exported-model": str(export_model_path),
        }
        rwc_curry("load_export_furious_cold", furious_load_kwarg, environ_load)
        rwc_curry("load_export_furious", furious_load_kwarg, environ_load)
        rwc_curry(
            "load_export_furious_gpu_preproc",
            furious_load_kwarg | {"gpu-preproc": None},
            environ_load,
        )

        environ_fast_load = {
            "TORCHINDUCTOR_CACHE_DIR": str(
                output_base_path / f"{ttype}_fast_export_furious_inductor_cache_dir"
            )
        }
        fast_furious_load_kwarg = furious_load_kwarg | {"fast": None}
        rwc_curry(
            "fast_export_furious_cold", fast_furious_load_kwarg, environ_fast_load
        )
        rwc_curry("fast_export_furious", fast_furious_load_kwarg, environ_fast_load)
        rwc_curry(
            "fast_export_furious_recompiles",
            fast_furious_load_kwarg | {"allow-recompiles": None},
            environ_fast_load,
        )
        rwc_curry(
            "fast_export_furious_gpu_preproc",
            fast_furious_load_kwarg
            | {
                "gpu-preproc": None,
            },
            environ_fast_load,
        )
        rwc_curry(
            "fast_export_furious_gpu_preproc_recompiles",
            fast_furious_load_kwarg | {"gpu-preproc": None, "allow-recompiles": None},
            environ_fast_load,
        )

        # TODO: Add a job that uses torchvision for I/O

        # Annotations for baseline folder
        run_annotate(
            image_paths,
            f"baseline_{ttype}",
            f"{ttype}_baseline_annotations",
            overwrite,
            dry,
        )

    all_keys = set().union(*(d.keys() for d in results))
    normalized_data = [{key: d.get(key, None) for key in all_keys} for d in results]
    df = pd.DataFrame(normalized_data)
    if not dry:
        df.to_csv(result_csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
