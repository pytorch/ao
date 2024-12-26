import subprocess
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
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

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
output_base_path = "~/blogs/sam2_amg_example"


def run(task, output_path, kwargs):
    image_paths = f"{output_base_path}/sav_val_image_paths_shuf_1000"
    run_script_with_args(["generate_data.py", "~/checkpoints/sam2" "large", task, image_paths, output_path], kwargs, dry=True)


if __name__ == "__main__":
    # Generate baseline AMG data
    run("amg", "~/blogs/sam2_amg_example/output_data", {'baseline': None, 'points-per-batch': 64})

    # Postprocessing baseline AMG data for SPS and MPS tasks
    # Call into annotate_with_rle

    for task_type in ["amg", "sps", "mps"]:
        output_ao_path = f"{output_base_path}/output_data_{task_type}_ao"
        export_model_path = f"{output_base_path}/exported_models/{task_type}_ao"

        # AO version of baseline for sanity check
        ppb_kwarg = {"points-per-batch":   64} if task_type == "amg" else {}
        ppb_kwarg = {"points-per-batch":    1} if task_type == "sps" else ppb_kwarg
        # Generate data for various settings
        run(task_type, output_ao_path,                                   ppb_kwarg)

        ppb_kwarg = {"points-per-batch":   1024} if task_type == "amg" else {}
        ppb_kwarg = {"points-per-batch":      1} if task_type == "sps" else ppb_kwarg
        run(task_type, output_ao_path + "_ppb_1024",                     ppb_kwarg)
        # TODO: Add experiment to export model
        run(task_type, output_ao_path + "_ppb_1024_load_export",         {**{              "load_fast": export_model_path + "_fast"}, **ppb_kwarg})
        run(task_type, output_ao_path + "_ppb_1024_fast_export",         {**{"fast": None, "load_fast": export_model_path + "_fast"}, **ppb_kwarg})
        run(task_type, output_ao_path + "_ppb_1024_fast_export_furious", {**{"fast": None, "load_fast": export_model_path + "_fast_furious", "furious": None}, **ppb_kwarg})
        run(task_type, output_ao_path + "_ppb_1024_fast_export_furious", {**{"fast": None, "load_fast": export_model_path + "_fast_furious", "furious": None,  "allow-recompiles": None}, **ppb_kwarg})

        # Calculating mIoU w.r.t. baseline results
        # TODO: Call into compare_rle_lists.py
