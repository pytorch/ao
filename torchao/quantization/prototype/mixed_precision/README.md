# Bayesian Optimization for Mixed-Precision Quantization
We provide a Bayesian Optimization (BO) tool to decide the post-training mixed-precision weight-only quantization configuration of a given pre-trained transformer model. It assigns different bitwidth and groupsize for each layer to shrink the model size or speedup the inference while preserving model accuracy. It also provides a sensitivity analysis tool and opens an option to assign initial configurations based on the sensitivity analysis, to further improve BO.

## Usage

### Dependencies:
The tool relies on lm_eval to measure model accuracy and ax-platform to conduct BO search. To install:
```
pip install lm_eval
pip install ax-platform
```
### Optional Step: Usage of sensitivity tool

We provide a sensitivity tool to calculate the [average Hessian matrix trace](https://arxiv.org/pdf/1911.03852) and the [fisher information matrix trace (FIT)](https://arxiv.org/pdf/2210.08502). With the sensitivity scores, we are able to identify sensitivity-guided initial configurations to better initialize the BO search. This step is optinoal to use BO tool.

#### Average Hessian trace:
Hessian is the second order partial derivation of the loss function and a higher average Hessian trace indicates a higher sensitivity of a layer to perturbations. Now the tool supports calculating one layer at a time to avoid out of memory issue for large models, e.g., Llama3-8B. It leverages the fast vhp (vector-hessian product) function from torch to achieve higher efficiency. To calculate average Hessian matrix trace of a layer on a calibration dataset (wikitext-v2-document):
```
python scripts/hessian_vhp.py --layer_id=LAYER_ID --checkpoint=/tmp/Meta-Llama-3-8B --max_seqlen=256 --max_iter=100 --nsamples=512
```
where --layer_id specifies which layer to calculate the average Hessian trace, LAYER_ID is an integer number used to identify the layer in the module name. The tool will print out the average Hessian trace using the calibration dataset for the certain layer. An output example:
```
Iterations Done
Avg Hessian trace for layer 0 is: 20135.83
```
Calculating Hessian trace is both memory-intensive and computationally expensive, the current tool takes 4 days with 4 A100 GPUs with 80GB GPU memory on a calibration dataset of 512 samples for Llama3-8B.

#### FIT:
FIT quantifies the total amount of information in the data about the parameter. It has been theoretically and empirically proved to be very close to Hession but with higher efficiency ([FIT paper])(https://arxiv.org/pdf/2210.08502). The tool support calculate the FIT score for all the layers at once. To calculate the FIT of the whole model on a calibration dataset (wikitext):
```
python scripts/fit.py --num_layers=32 --checkpoint=/tmp/Meta-Llama-3-8B --max_seqlen=2048 --max_iter=100 --nsamples=128
```
The tool will print out the average FIT scores based on the calibration dataset for all the layers. An output example:
```
Iterations Done
FIT scores for 32 layers:
[237201.35, 547750.91,  87226.19,  50000.96,
  52017.47,  28319.72,  21997.11,  20681.59,
  21076.09,  21016.67,  18572.73,  19594.67,
  17585.58,  20135.83,  22986.77,  21849.15,
  21690.99,  21204.48,  19281.44,  17967.87,
  16843.32,  19385.39,  18394.11,  15991.45,
  15684.25,  15192.07,  15993.08,  16999.28,
  17418.69,  21241.36,  23579.92,  52762.86]
```
where the arguments checkpoint, max_seqlen, nsamples, max_iter are similar to the usage of running Hession. The only difference is that we replacing --layer_id with --num_layers to identify the total numbers of layers to calculate FIT scores for.

Calculating FIT takes 3.3h with 1 A100 GPU with 80GB GPU memory. on a calibration dataset of 512 samples for Llama3-8B.

### Usage of BO search

#### Step 1: Define parameter space

Given a model, to conduct a BO search, we first need to identify the parameter space for the model, ie., for each layer, set up the value or choices of bitwidth and groupsize. A simple example of parameter space configuration is shown below and an example for Llama3-8B is in Llama3-8B_parameters.json.

```
    {
      "name": "bitwidth",
      "name_format": "bitwidth.{i}.",
      "layers": [
        {"range": [0, 3], "type": "fixed", "value": 5}, # the first 3 layers are assigned to the fixed bitwidth = 5
        {"range": [3, 32], "type": "choice", "values": [2, 3, 4, 5, 6, 8]}, # the bitwidths of the rest 29 layers will choose from the list
      ]
    },
```
A parameter for a layer (specified in the range) can be either "fixed" or "choice" type for a fixed value or a list of possible choices. A default parameter space setting will be search from [2, 3, 4, 5, 6, 8] bit and [32, 64, 128, 512] groupsize for each layer.

#### Step 2: Define initial samples (optional)
Then an optional step is to obtain some better initial samples based on the sensitivity scores. A layer with a higher sensitivity score (Hessian or FIT) should be assigned with a higher bitwidth and a smaller groupsize, to preserve the model accuracy. E.g., the FIT scores for the first 3 layers are far higher then other layers, thus we can set <5-bit, groupsize=32> for them and <4-bit, groupsize=64> for all the other layers. A simple example of initial samples of BO search is shown below and an example for Llama3-8B is shown in Llama3-8B_initial_samples.json. A default initial samples will be randomly sampled from the valid parameter space. We recommend users to add at least 10 examples to better initialize the BO strategy.

```
{
  "initial_samples": [
    {
      "bitwidth.0.": 8,
      "groupsize.0.": 64,
      "bitwidth.1.": 4,
      "groupsize.1.": 32,
      "bitwidth.2.": 5,
      "groupsize.2.": 128,
    },
  ]
}

```

#### Step 3: Run BO experiment
To conduct BO search to optimize model accuracy under a certain model size constraint:

```
python --BO_acc_modelsize.py --checkpoint=/tmp/Meta-Llama-3-8B --num_trials=200 --model_size_constraint=6.0 --output_file=BO_acc_modelsize_output.csv --parameters_list=Llama3-8B_parameters.json --initial_samples=Llama3-8B_initial_samples.json --gpu_lists=0,1,2,3"
```

where
--num_trials identifies the number of search for BO
--model_size_constraint identifies the max model size for valid search results (unit: GB)
--parameters_list identifies the path to load parameter space.
--initial_samples identifies the path to get initial samples of BO search
--gpu_lists enbles evaluating BO different BO trials on different GPUs, otherwise will use only one GPU

For Llam3-8B, a search takes 1.5h on wikitext-document from lm_eval on 8 A100 GPUs with 80GB GPU memory.

The tool will print out the best configuration and results (accuracy ("cal_PPL"), model size ("model_size") or throughput ("cal_throughput")) among the search. Example output:

```
------Best config------
{'cal_PPL': 7.4736, 'model_size': 5.9766} {'bitwidth.0.': 5, 'groupsize.0.': 32, 'bitwidth.1.': 5, 'groupsize.1.': 32,...,'bitwidth.31.': 5, 'groupsize.31.': 32}
```

The tool will also write the BO search trial history to history_output csv file with three columns:

|   cal_PPL        |model size | quant_config|
| ---------------- | ------ | ------ |
| 7.5286  | 5.8418 | {'bitwidth.0.': 4, 'groupsize.0.': 64, 'bitwidth.1.': 6, 'groupsize.1.': 32,...,'bitwidth.31.': 5, 'groupsize.31.': 32} |
| 7.4736  | 5.9766 | {'bitwidth.0.': 5, 'groupsize.0.': 32, 'bitwidth.1.': 5, 'groupsize.1.': 32,...,'bitwidth.31.': 5, 'groupsize.31.': 32} |
...

#### Run BO to optimize inference speed
We also provide another version of BO search to optimize inference throughput (with torch.compile()) under a certain model accuracy constraint:
```
python --BO_acc_throughput.py --checkpoint=/tmp/Meta-Llama-3-8B --num_trials=200 --ppl_constraint=7.5 --output_file=BO_acc_modelsize_output.csv --parameters_list=Llama3-8B_parameters.json --initial_samples Llama3-8B_initial_samples.json
```
All the arguments are similar to the optmizing accuracy under model size constraint, except replacing --model_size_constraint with --ppl_constraint=7.5 to set up the perplexity limit of the valid search results.

Similarly, the tool will output the best configuration for both inference throughput and model accuracy.
```
------Best config------
{'cal_throughput': 147.72, 'cal_PPL': 7.3134} {'bitwidth.0.': 5, 'groupsize.0.': 32, 'bitwidth.1.': 5, 'groupsize.1.': 32,...,'bitwidth.31.': 5, 'groupsize.31.': 32}
```
and write out the BO search history file:
|   cal_throughput        | cal_PPL | quant_config|
| ---------------- | ------ | ------ |
| 135.64  | 7.5322 | {'bitwidth.0.': 6, 'groupsize.0.': 64, 'bitwidth.1.': 4, 'groupsize.1.': 128,...,'bitwidth.31.': 5, 'groupsize.31.': 64} |
| 147.72  | 7.3134 | {'bitwidth.0.': 5, 'groupsize.0.': 32, 'bitwidth.1.': 5, 'groupsize.1.': 32,...,'bitwidth.31.': 5, 'groupsize.31.': 32} |
...

#### Run BO for other models
We are supporting more models, such as more transformer models and ViT models. To run all the above experiments for a new model e.g., Mistrial-7B-v0.1, you will need to specified the correct path to load model with --checkpoint, the desired parameters space with --parameters_list and the optional your pre-defined initial samples with --initial_samples, with the following command, similarly for optimizing the inference speed:

```
python --BO_acc_modelsize.py --checkpoint=/tmp/Mistral-7B-v0.1/ --num_trials=200 --model_size_constraint=6.0 --output_file=BO_acc_modelsize_output.csv --parameters_list=Mistral-7B_parameters.json --initial_samples=Mistral-7B_initial_samples.json --gpu_lists=0,1,2,3"
```

Support for ViT models is coming soon.


## Results
We evaluated BO search for Llama3-8B and Mistral-7B-v0.1 under two settings: (1) optimizing model accuracy under model size constraint; (2) optimizing model inference throughput under model accuracy constraint, and compared the BO results with bfloat-16, [int8 weight only](https://github.com/pytorch/ao/blob/983f5653f5516e91c9fb9df73d6f407fbd4b381f/torchao/quantization/quant_api.py#L432) uniform quantization and [int4 weight only](https://github.com/pytorch/ao/blob/983f5653f5516e91c9fb9df73d6f407fbd4b381f/torchao/quantization/quant_api.py#L396) uniform quantization.

### Results of BO for optimizing model accuracy under model size constraint

For Llama3-8B, the BO search quantization saves 20.1% model size with 2.85% ppl degradation compared to int8wo uniform quantization baseline.
The manual baseline here means using <5-bit, groupsize=32> for the first-3 and last-2 layers which have higher sensitivity scores, and <4-bit, groupsize=64> for all the other layers.

|    Llama3-8B        |ppl | model size|
| ---------------- | ------ | ------ |
| bf16 baseline  | 7.260 | 15.01 |
| int8wo uniform quantization | 7.263 | 7.480 |
| int4wo uniform quantization  | 7.900 | 5.411 |
| manual baseline  | 7.679 | 5.545 |
| BO mixed-precision quantization  | 7.470 | 5.976 |


For Mistral-7B-v0.1, BO search quantization saves 30.6% model size with only 1.74% ppl degradation compared to int8wo uniform quantization baseline.
|    Mistral-7B-v0.1   |ppl | model size|
| ---------------- | ------ | ------ |
| bf16 baseline  | 8.021 | 13.49 |
| int8wo uniform quantization  | 8.028  | 7.90  |
| int4wo uniform quantization  | 8.387  | 4.65 |
| BO mixed-precision quantization  | 8.168 | 5.48 |


### Results of BO for optimizing model inference throughput under model accuracy constraint
For Llama3-8B, the BO search quantization improves 15.2% throughput with only 2.85% ppl degradation compared to int8wo uniform quantization baseline.

|    Llama3-8B     |ppl | throughput|
| ---------------- | ------ | ------ |
| bf16 baseline  | 7.260 | 94.97 |
| int8wo uniform quantization  | 7.263 | 139.76 |
| int4wo uniform quantization  | 7.900 | 179.44 |
| BO mixed-precision quantization  | 7.470 | 160.96 |
