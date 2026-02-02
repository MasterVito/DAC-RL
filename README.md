<h1 align="center">
Training LLMs for Divide-and-Conquer Reasoning
</h1>

<div align="center">

![](https://img.shields.io/badge/Task-LLM%20Reasoning-orange)
![](https://img.shields.io/badge/Paradigm-RLVR-blue)
![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<!-- <p align="center">
  <a href=""><b>[📜 Paper]</b></a> •
  <a href="https://github.com/MasterVito/SvS"><b>[🐱 GitHub]</b></a> •
  <a href="https://x.com/MasterVito0601/status/1959960582670766411"><b>[🐦 Twitter]</b></a> •
  <a href="https://www.xiaohongshu.com/discovery/item/68ac220f000000001d03b785?source=webshare&xhsshare=pc_web&xsec_token=ABgRNEq2SWHDpaqHQVIdEoTMd8NOnA-NAuilqp1VyEn0Y=&xsec_source=pc_share"><b>[📕 Rednote]</b></a>
</p>
 -->


<!-- <p align="center">
Repo for "<a href="https://arxiv.org/pdf/2508.14029v1" target="_blank">Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR</a>"
</p> -->


<p align="center">
    <img src="assets/DAC-RL-Main.png" width="1000">
        <br>
    <em>Figure 1: An overview of the DAC-style inference and reward assignments in training, illustrated with a case study.
    </em>
</p>


<!-- <br> -->

## 🔥 News

<!-- - [2023/10/13] 🔥🔥🔥 We release a demo for ToRA at [🐯 Gradio](https://9557c5365a6f44dc84.gradio.live), try it out!!! -->
<!-- - [2023/06/13] We release all prompts used in the SwS framework in <a href="https://github.com/MasterVito/SwS/tree/master/prompts"><b>prompts</b></a>.
- [2023/06/13] We update the demo set of synthetic problems from SwS in <a href="https://github.com/MasterVito/SwS/tree/master/datasets"><b>datasets</b></a>, including 500 samples for each model and category. You can also find them in <a href="https://huggingface.co/datasets/MasterVito/SwS-Demo-Dataset"><b>Demo Dataset</b></a>. -->
<!-- - [2025/12/13] 🔥🔥🔥 **We open-sourced three SvS model checkpoints at different scales, along with an additional 7B checkpoint for coding tasks, available at <a href="https://huggingface.co/RLVR-SvS"><b>[Models]</b></a>. Training parquet data are attached in the respective repos.**
- [2025/08/25] **We provide the full code for training and evaluation for SvS.**
- [2025/08/19] **Our full code and datasets are under review by Microsoft and will be released upon approval.** -->
- [2026/02/02] DAC-RL paper and repo are released.

<!-- <br> -->

<br>

## 💡 Introduction 
We propose an end-to-end reinforcement learning framework that trains LLMs to perform divide-and-conquer (DAC) reasoning, which substantially improves downstream performance and test-time scalability on challenging reasoning tasks.

<br>

## 📊 Experiments on <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507</a>

| Model | AIME 2024 |  | AIME 2025 |  | Beyond-AIME |  | HMMT 2025 |  | Average |  |
|-------|-----------|-----------|-----------|-----------|-------------|-----------|-----------|-----------|----------|----------|
|       | Pass@1 | Pass@32 | Pass@1 | Pass@32 | Pass@1 | Pass@32 | Pass@1 | Pass@32 | Pass@1 | Pass@32 |
| **Qwen3-4B-Instruct-2507** |  |  |  |  |  |  |  |  |  |  |
| Init-CoT | 62.6 | 90.0 | 45.7 | 76.7 | 32.1 | 65.0 | 30.3 | 56.7 | 42.7 | 72.1 |
| Init-DAC | 59.6 | 90.0 | 43.2 | 73.3 | 29.6 | 61.0 | 28.2 | 63.3 | 40.2 | 71.9 |
| RL-CoT | 45.9 | 85.8 | 52.1 | 77.4 | 30.4 | 58.1 | 21.8 | 54.4 | 37.5 | 69.0 |
| **RL-DAC** | **63.9** | **87.7** | **54.2** | **78.8** | **34.6** | **67.9** | **31.9** | **66.6** | **46.1** | **75.3** |
| **∆ (RL)** | **+18.0** | **+1.9** | **+2.1** | **+1.4** | **+4.2** | **+9.8** | **+10.1** | **+12.2** | **+8.6** | **+6.3** |

<br>

## 🚀 Quick Start

### ⚙️ Setup

We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) (0.10.1.1) to accelerate inference. Run the following commands to setup your environment:

```sh
conda create -n svs python=3.10.16
conda activate svs
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.6 for example
pip install -r requirements.txt
```

<br>

### 🪁 Evaluation

We provide evaluation scripts for both **CoT** and **DAC** inference. To use them, simply configure the `model_name_or_path` (default: `Qwen/Qwen3-4B-Instruct-2507`) and the `data_path` (by default, AIME 24, AIME 25, Beyond-AIME, and HMMT-25 are used for evaluation, as described in the paper) in [`scripts/eval_cot.sh`](scripts/eval_cot.sh) and [`scripts/eval_dac.sh`](scripts/eval_dac.sh), and then run the following command:


```sh
bash scripts/eval_cot.sh
bash scripts/eval_dac.sh
```

<br>

### ⚡️ Training
We also present our complete training scripts for the community. We provide the training data used in our paper in [data](data). 
For example, to train the <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507</a> model, run the following command:

```sh
bash scripts/run_dac_training.sh
```

<br>

## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
```
<br>


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mastervito/DAC-RL&type=Date)](https://star-history.com/#mastervito/DAC-RL&Date)
