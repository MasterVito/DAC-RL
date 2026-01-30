# DAC-RL Anonimous Repo

## ⚙️ Setup

We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) (0.10.1.1) to accelerate inference. Run the following commands to setup your environment:

```sh
conda create -n svs python=3.10.16
conda activate svs
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.6 for example
pip install -r requirements.txt
```

## 🪁 Evaluation

We provide evaluation scripts for both **CoT** and **DAC** inference. To use them, simply configure the `model_name_or_path` (default: `Qwen/Qwen3-4B-Instruct-2507`) and the `data_path` (by default, AIME 24, AIME 25, Beyond-AIME, and HMMT-25 are used for evaluation, as described in the paper) in [`scripts/eval_cot.sh`](scripts/eval_cot.sh) and [`scripts/eval_dac.sh`](scripts/eval_dac.sh), and then run the following command:


```sh
bash scripts/eval_cot.sh
bash scripts/eval_dac.sh
```

## ⚡️ Training
We also present our complete training scripts for the community. We provide the training data used in our paper in [data](data). 
For example, to train the <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507</a> model, run the following command:

```sh
bash scripts/run_dac_training.sh
```