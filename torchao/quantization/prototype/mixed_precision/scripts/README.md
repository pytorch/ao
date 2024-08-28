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
Hessian is the second order partial derivation of the loss function and a higher average Hessian trace demonstrates a higher sensitivity of a layer to perturbations. Now the tool supports calculating one layer at a time to avoid out of memory issue for large models, e.g., Llama3-8B. It leverages the fast vhp (vector-hessian product) function from torch to achieve more efficient To calculate average Hessian matrix trace of layer i on a calibration dataset (wikitext):
```
python scripts/hessian_vhp.py --layer_id=0 --checkpoint=/tmp/Meta-Llama-3-8B --max_seqlen=2048 --maxIter=100 --nsamples=128
```
where,
--layer_id identifies which layer to calculate the average Hessian trace
--checkpoint identifies the path to load the model
--max_seqlen identifies the max length of input samples of the calibration dataset
--nsamples identifies the number of samples of the calibration dataset
--maxIter identifies the max iterations to run to calulate the average Hessian trace

Calculating Hessian trace is both memory-intensive and computationally expensive, the current tool takes 4 days with 4 GPUs on a calibration dataset of 512 samples.

#### FIT:
FIT quantifies the total amount of information in the data about the parameter. It has been theoretically and empirically proved to be very close to Hession but with higher efficiency [FIT paper](https://arxiv.org/pdf/2210.08502). The tool support calculate the FIT score for all the layers at once. To calculate the FIT of the whole model on a calibration dataset (wikitext):
```
python scripts/fit.python --num_layers=32 --checkpoint=/tmp/Meta-Llama-3-8B --max_seqlen=2048 --maxIter=100 --nsamples=128
```
where the arguments checkpoint, max_seqlen, nsamples, maxIter are similar to the usage of running Hession. The only difference is that we replacing --layer_id with --num_layers to identify the total numbers of layers to calculate FIT scores for.

Calculating FIT takes 3.3h with 1 GPU on a calibration dataset of 512 samples.

### Step 1: Usage of BO search
A naive BO search tool will leverage random initialization. With the sensitivity scores, we can sample better initialize configurations for BO search.

An example of parameter space configuration is shown in Llama3-8B_parameters.json
An example of initial samples of BO search is shown in Llama3-8B_initial_samples.json

To conduct



## Results
We evaluate BO search for

### Results of BO for optimizing model accuracy under model size constraint

|    Llama3-8B        |ppl | model size|
| ---------------- | ------ | ------ |
| bf16 baseline  | 7.260 | 15.01 |
| int8wo uniform  | 7.263 | 7.480 |
| int4wo uniform quantization  | 7.900 | 5.411 |
| manual baseline  | 7.679 | 5.545 |
| BO mixed-precision quantization  | 7.470 | 5.976 |


|    Mistral-7B-v0.1   |ppl | model size|
| ---------------- | ------ | ------ |
| bf16 baseline  | 8.021 | 13.49 |
| int8wo uniform quantization  | 8.028  | 7.90  |
| int4wo uniform quantization  | 7.900 |  |
| manual baseline  | 8.387  | 4.65 |
| BO mixed-precision quantization  | 8.168 (+1.8%, 0.15) | 5.48 (-59.4%, 8.01 GB)  |



### Results of BO for optimizing model inference throughput under model accuracy constraint
|                  |ppl | throughput|
| ---------------- | ------ | ------ |
| bf16 baseline  | 7.260 | 94.97 |
| int8wo uniform quantization  | 7.263 | 139.76 |
| int4wo uniform quantization  | 7.900 | 179.44 |
| BO mixed-precision quantization  | 7.470 | 160.96 |
