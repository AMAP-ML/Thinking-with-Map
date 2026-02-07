<h1 align="center" style="margin-top: 10px;">Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization</h1>

<p align="center">
  <a href="https://yuxiang-ji.com/">Yuxiang Ji</a><sup>1,2</sup>&nbsp;
  Yong Wang<sup>2†</sup>&nbsp;
  Ziyu Ma<sup>2</sup>&nbsp;
  Yiming Hu<sup>2</sup>&nbsp;
  Hailang Huang<sup>2</sup>&nbsp;
  <br>
  Xuecai Hu<sup>2</sup>&nbsp;
  Guanhua Chen<sup>3</sup>&nbsp;
  Liaoni Wu<sup>1</sup>&nbsp;
  Xiangxiang Chu<sup>2</sup>&nbsp;
  <br>
  <sup>1</sup>Xiamen University &nbsp;&nbsp;
  <sup>2</sup>AMAP, Alibaba Group &nbsp;&nbsp;
  <sup>3</sup>Southern University of Science and Technology
  <br>
  <sup>†</sup>Project lead &nbsp;&nbsp;&nbsp;
</p>

<div align="center"> 

<p align="center">
<img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" height="14" style="display:inline;" /> <a href="https://amap-ml.github.io/Thinking-with-Map" target="_blank">Page</a> |
<img src="./resources/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2601.05432" target="_blank">Arxiv</a> |
🤗 <a href="https://huggingface.co/papers/2601.05432" target="_blank">Paper</a> |
🤗 <a href=https://huggingface.co/datasets/GD-ML/MAPBench-V2 target="_blank">Data</a> | 
🤗 Model
<p align="center">
<p align="center">
<img src="./resources/modelscope.svg" height="12px" /> <a href="https://modelscope.cn/datasets/yux1ang/MAPBench-V2" target="_blank">Data</a> | 
<img src="./resources/modelscope.svg" height="12px" /> Model |
<img src="./resources/x.png" width="14px" style="display:inline;"> <a href="https://x.com/_akhaliq/status/2010723473392546198" target="_blank">X@AK</a>

</div>

> [!NOTE]
> This project includes the codebase, datasets and chckpoints for **Thinking with Map**: a map-augmented agent for geolocalization. Given an image in-the-wild, the agent can conduct reasoning with map to inference the location.


## 🎬 Demo
<summary><h3>Demo of <em>Thinking with Map</em></h3></summary>
<p align="center">
  <img alt="demo" src="resources/demo.jpg" />
  <i>
  The illustration of a complete Thinking with Map process.
  </i>
</p>

<p align="center">
  <img alt="demo" src="resources/overall_comp.jpg" />
  <i>
  Comparison with open- and closed-source models.
  </i>
</p>

## News
- [Feb 3, 2026]: 🛠️ Our code and data are realeased now.
- [Jan 12, 2026]: 🔥 We are honored to be featured as HuggingFace [Daily Paper #1](https://huggingface.co/papers/date/2026-01-12).
- [Jan 12, 2026]: 📍 Our paper is released on [ArXiv](https://arxiv.org/abs/2601.05432) and [HuggingFace](https://huggingface.co/papers/2601.05432).

## Table of contents
- [Dataset Access](#dataset-access)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## <a id="dataset-access"></a> 💾 Dataset Access
We release two versions of the dataset. 
- V1 contains the training and test data used in the paper (lower resolution, will be deperecated).
- We also provide V2 with higher-resolution images (recommended): the released training set includes the subset of 6k samples, and the test set is the same size as V1 (~2.5k samples).

|                                      V1 (Low Resolution, Deperecated)                                      |                                     V2 (High Resolution, **Recommended**)                                      |
|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| [🤗HuggingFace](https://huggingface.co/datasets/GD-ML/MAPBench-V1)  | [🤗HuggingFace](https://huggingface.co/datasets/GD-ML/MAPBench-V2) |
| [<img src="./resources/modelscope.svg" height="11px" />ModelScope](https://modelscope.cn/datasets/yux1ang/MAPBench-V1) | [<img src="./resources/modelscope.svg" height="11px" />ModelScope](https://modelscope.cn/datasets/yux1ang/MAPBench-V2) |

## <a id="model-zoo"></a> 📦 Model Zoo
|  Qwen3-VL-30B-A3B RL-tuned on MAPBench-V2 37k  |
|:----------------------------------------------:|
|           🤗HuggingFace (Released soon)          |
|           <img src="./resources/modelscope.svg" height="11px" />ModelScope (Released soon)           |


## <a id="quick-start"></a> 🚀 Quick Start
After downloading the dataset, process it into the parquet format.
```bash
cd verl/examples/data_preprocess
bash preprocess_thinking_with_map.sh
```

### Installation
If you just want to try the demo without training/evaluating, you can skip the installation part and try [Inference part](#inference) directly.

Please refer to [VeRL Installation](https://verl.readthedocs.io/en/latest/start/install.html) for more details.
```bash
## VeRL Installation 
# use conda for environment management
conda create -n mapagent python==3.12
conda activate mapagent
# install torch and vllm
pip install torch==2.8.0
pip install vllm==0.11.0
# install sglang and other basic packages
cd Thinking-with-Map/verl/scripts
bash install_vllm_sglang_mcore.sh
# install verl
cd Thinking-with-Map/verl
pip install --no-deps -e .
```

The tool server can share the same environment with VeRL.
```bash
## Tool Sever Installation
pip install playwright
pip install "uvicorn[standard]"
pip install json5
pip install fastapi
```

### Inference
After downloading the model, employ it by vllm server.
```bash
# at least 2 GPUs with 80GB memory each for the trained Qwen3-VL-30B-A3B model
# try more GPUs when OOM
vllm serve /path/to/released/model \
    --tensor-parallel-size 2 \
    --port 8002
```
Then try the demo with `demo/cookbook_thinking_with_map.ipynb`.

### Training
Start the tool server with cache on each cluster node.
```bash
cd verl/tool_server
# need redis server running on port 6397 for cache
# change your own api key in the bash script
bash run_api_server.sh $RANK
```

Simply run distributed RL training on each cluster node.
```bash
cd verl/geoagent_scripts
# training on MAPBench
bash train_thinking_with_map.sh
```
The default training is conduct on 8 nodes with 8 GPUs each, while you can change it by modifying `N_NODES`, `train_batch_size` and `ppo_mini_batch_size` respectively to adapt to your own environment.

### Evaluation
The evaluation code is also conducted on VeRL.
```bash
cd verl/geoagent_scripts
bash test_thinking_with_map.sh
```

## <a id="acknowledgement"></a> 🙏 Acknowledgement
The repo is built upon [VeRL](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), and [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).
We appreciate these open-source communities for their great work.

## <a id="citation"></a> 📌 Citation
```bibtex
@article{ji2026thinking,
  title={Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization},
  author={Ji, Yuxiang and Wang, Yong and Ma, Ziyu and Hu, Yiming and Huang, Hailang and Hu, Xuecai and Chen, Guanhua and Wu, Liaoni and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2601.05432},
  year={2026}
}
```