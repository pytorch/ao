import subprocess
import fire
import json
import pandas as pd
from pathlib import Path
from compare_rle_lists import compare as compare_folders

def run_script_with_args(positional_args, keyword_args, dry=False):
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
        print(" ".join(command))
        if dry:
            return
        # Run the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print the output
        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
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
    # output_base_path = "~/blogs/sam2_amg_example"
    # image_paths = f"{output_base_path}/sav_val_image_paths_shuf_1000"

    results = []

    def run(task, output_path: Path, kwargs, baseline_folder=None):
        output_path.mkdir(exist_ok=overwrite)
        stdout, stderr = run_script_with_args(["generate_data.py",
                                               "~/checkpoints/sam2",
                                               "large",
                                               task,
                                               image_paths,
                                               str(output_path)],
                                              kwargs,
                                              dry=dry)
        assert stdout is not None
        all_stats = json.loads(stdout.split("\n")[-2])
        if baseline_folder is not None:
            miou_count, miou_sum, fail_count = compare_folders(str(output_path),
                                                               str(baseline_folder),
                                                               strict=True,
                                                               compare_folders=True)
            all_stats["miou_count"] = miou_count
            all_stats["miou_sum"] = miou_sum
            all_stats["fail_count"] = fail_count
        all_stats["task"] = task
        all_stats = all_stats | kwargs
        # TODO:: Save this and use overwrite to check for it before writing
        results.append(all_stats)
        return all_stats


    for task_type in ["amg"]:  # , "sps", "mps"]:
        # Generate baseline data
        ppb_kwarg = {"points-per-batch":   64} if task_type == "amg" else {}
        ppb_kwarg = {"points-per-batch":    1} if task_type == "sps" else ppb_kwarg
        output_baseline_path = output_base_path / f"output_data_baseline_{task_type}"
        run("amg", output_baseline_path, {**{'baseline': None}, **ppb_kwarg})

        def run_with_compare(task, output_path: Path, kwargs):
            return run(task, output_path, kwargs, baseline_folder=output_baseline_path)

        output_ao_path = output_base_path / f"output_data_{task_type}_ao"
        # AO version of baseline for sanity check
        run_with_compare(task_type, output_ao_path,                                   ppb_kwarg)


        # TODO: Need meta folder for sps and mps
        # Postprocessing baseline AMG data for SPS and MPS tasks
        # Call into annotate_with_rle

        ppb_kwarg = {"points-per-batch":   1024} if task_type == "amg" else {}
        ppb_kwarg = {"points-per-batch":      1} if task_type == "sps" else ppb_kwarg
        run_with_compare(task_type, Path(str(output_ao_path) + "_ppb_1024"),                     ppb_kwarg)

        export_model_path = output_base_path / "exported_models" / f"{task_type}_ao_fast"
        export_model_path.mkdir(exist_ok=overwrite, parents=True)
        # TODO: Set num images to 0 for export job
        run(task_type, Path(str(output_ao_path) + "_ppb_1024_save_export"),                    {**{              "export-model":        str(export_model_path)}, **ppb_kwarg})
        run_with_compare(task_type, Path(str(output_ao_path) + "_ppb_1024_load_export"),                    {**{              "load-exported-model": str(export_model_path)}, **ppb_kwarg})
        run_with_compare(task_type, Path(str(output_ao_path) + "_ppb_1024_fast_export"),                    {**{"fast": None, "load-exported-model": str(export_model_path)}, **ppb_kwarg})

        export_model_path = output_base_path / "exported_models" / f"{task_type}_ao_fast_furious"
        export_model_path.mkdir(exist_ok=overwrite, parents=True)
        # TODO: Set num images to 0 for export job
        run(task_type, Path(str(output_ao_path) + "_ppb_1024_save_export_furious"),            {**{              "export-model":        str(export_model_path), "furious": None}, **ppb_kwarg})
        run_with_compare(task_type, Path(str(output_ao_path) + "_ppb_1024_fast_export_furious"),            {**{"fast": None, "load-exported-model": str(export_model_path), "furious": None}, **ppb_kwarg})
        run_with_compare(task_type, Path(str(output_ao_path) + "_ppb_1024_fast_export_furious_recompiles"), {**{"fast": None, "load-exported-model": str(export_model_path), "furious": None,  "allow-recompiles": None}, **ppb_kwarg})

        # TODO: Add a job that uses torchvision for I/O

        # Calculating mIoU w.r.t. baseline results
        # TODO: Call into compare_rle_lists.py

        # Run image annotations to visualize differences and the task at hand

    all_keys = set().union(*(d.keys() for d in results))
    normalized_data = [{key: d.get(key, None) for key in all_keys} for d in results]
    df = pd.DataFrame(normalized_data)
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)

if __name__ == "__main__":
    fire.Fire(main)
