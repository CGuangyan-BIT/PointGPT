# PointGPT

## PointGPT: Auto-regressively Generative Pre-training from Point Clouds [ArXiv](https://arxiv.org/abs/2305.11487)

In this work, we present PointGPT, a novel approach that extends the concept of GPT to point clouds, utilizing a point cloud auto-regressive generation task for pre-training transformer models. In object classification tasks, our PointGPT achieves 94.9% accuracy on the ModelNet40 dataset and 93.4% accuracy on the ScanObjectNN dataset, outperforming all other transformer models. In few-shot learning tasks, our method also attains new SOTA performance on all four benchmarks.

<div  align="center">    
 <img src="./figures/net.png" width = "666"  align=center />
</div>

## News

[2023.08.19] Code has been updated; PointGPT-B and PointGPT-L models have been released!

[2023.06.20] Code and the PointGPT-S models have been released!


## 1. Requirements

PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

Our training data for the PointGPT-S model encompasses ShapeNet, ScanObjectNN, ModelNet40, and ShapeNetPart datasets. For detailed information, please refer to [DATASET.md](./DATASET.md).

To pretrain the PointGPT-B and PointGPT-L models, we employ both Unlabeled hybrid datasets and labeled hybrid datasets, available for download at the following link: [here](https://drive.google.com/file/d/1TWgd3eJX1HDruFfU9JrGnBfcVhzJIXqT/view?usp=sharing).



## 3. PointGPT Models
### PointGPT-S Models
| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | ShapeNet       | [pretrain.yaml](./cfgs/PointGPT-S/pretrain.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1gTFI327kXVDFQ90JfYX0zIS4opM1EkqX/view?usp=drive_link) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/PointGPT-S/finetune_scan_hardest.yaml) | 86.9%      | [here](https://drive.google.com/file/d/12Tj2OFKsEPT5zd5nQQ2VNEZlCKHncdGh/view?usp=drive_link) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/PointGPT-S/finetune_scan_objbg.yaml)     | 91.6%      | [here](https://drive.google.com/file/d/1s4RrBkfwVr8r0H2FxwiHULcyMe_EAJ9D/view?usp=drive_link) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/PointGPT-S/finetune_scan_objonly.yaml) | 90.0%      | [here](https://drive.google.com/file/d/173yfDAlqqed-oRHaogX6DC4Uj1b8Rvxt/view?usp=drive_link) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointGPT-S/finetune_modelnet.yaml)         | 94.0%      | [here](https://drive.google.com/file/d/17uoJchAzwapTNHVxOWNH4HLNZz9kbGoo/view?usp=drive_link) |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](./cfgs/PointGPT-S/finetune_modelnet_8k.yaml)   | 94.2%      | [here](https://drive.google.com/file/d/1XocTFSsKZgKHx2cLqZJi2rcF74hQ-1nx/view?usp=drive_link) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 86.2% mIoU | [here](https://drive.google.com/file/d/1WVMTtIq4vPQOOnlDsymVA5541lNL-hm3/view?usp=drive_link) |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/fewshot.yaml) | 96.8 ± 2.0     | 98.6 ± 1.1     | 92.6 ± 4.6      | 95.2 ± 3.4      |

### PointGPT-B Models
| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | UnlabeledHybrid       | [pretrain.yaml](./cfgs/PointGPT-B/pretrain.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1Gyf9ZR8MCPg1XOCALjJR9VJepV7iAi5S/view?usp=sharing) |
| Post-pre-training | LabeledHybrid       | [post_pretrain.yaml](./cfgs/PointGPT-B/post_pretrain.yaml)                        | N.A.       | [here](https://drive.google.com/file/d/1Gc7thuU-D1Sq4NIMTV6-U1LhVN0E2z9l/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/PointGPT-B/finetune_scan_hardest.yaml) | 91.9%      | [here](https://drive.google.com/file/d/1tHi7W935DxVttXHG0Mgb0HSfYWUqXLwB/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/PointGPT-B/finetune_scan_objbg.yaml)     | 95.8%      | [here](https://drive.google.com/file/d/1te8DuC_-cOzt4JayyaNWvxHcRztjDlGF/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/PointGPT-B/finetune_scan_objonly.yaml) | 95.2%      | [here](https://drive.google.com/file/d/17c8KvDrAuY0GgcO7SGE-4zlMArjzkjLX/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointGPT-B/finetune_modelnet.yaml)         | 94.4%      | [here](https://drive.google.com/file/d/1l5zhy52erSp5gigbhYaT0nyMrV_lbh-C/view?usp=sharing) |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](./cfgs/PointGPT-B/finetune_modelnet_8k.yaml)   | 94.6%      | [here](https://drive.google.com/file/d/1FzM7ULPUAOk_J0BRHFvv0nS_Xd65oWbV/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 86.5% mIoU | [here](https://drive.google.com/file/d/1P6hELhX6Yr-rN04q6N71wZfvW2HnLhqD/view?usp=sharing) |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/PointGPT-B/fewshot.yaml) | 97.5 ± 2.0     | 98.8 ± 1.0     | 93.5 ± 4.0      | 95.8 ± 3.0      |

### PointGPT-L Models
| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | UnlabeledHybrid       | [pretrain.yaml](./cfgs/PointGPT-L/pretrain.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1nzCwriFbC2QoDbRpGhWvf_DbFIkFU6zV/view?usp=sharing) |
| Post-pre-training | LabeledHybrid       | [post_pretrain.yaml](./cfgs/PointGPT-L/post_pretrain.yaml)                        | N.A.       | [here](https://drive.google.com/file/d/1Kh6f6gFR12Y86FAeBtMU9NbNpB5vZnpu/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/PointGPT-L/finetune_scan_hardest.yaml) | 93.4%      | [here](https://drive.google.com/file/d/1e_qIfZCqQmq0eRpYhf9xrIxl6TkzsaZ9/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/PointGPT-L/finetune_scan_objbg.yaml)     | 97.2%      | [here](https://drive.google.com/file/d/1gd8gn0ffK0zfWv7AAUbygzIPSeeRU8fD/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/PointGPT-L/finetune_scan_objonly.yaml) | 96.6%      | [here](https://drive.google.com/file/d/1F2MnPmQGKnYUgmS5uz3PNInU23jWsNj1/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointGPT-L/finetune_modelnet.yaml)         | 94.7%      | [here](https://drive.google.com/file/d/1ntWwZCvD_Tqykq9F7QrDKXH7aL-dcCsQ/view?usp=sharing) |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](./cfgs/PointGPT-L/finetune_modelnet_8k.yaml)   | 94.9%      | [here](https://drive.google.com/file/d/1gKgdbtIuRinJY-NElSHwrKAL5OhBjrGD/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 86.6% mIoU | [here](https://drive.google.com/file/d/1d3fXLBkXvzl9YjX5DDMdm7rUtCvfwgUL/view?usp=sharing) |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/PointGPT-L/fewshot.yaml) | 98.0 ± 1.9     | 99.0 ± 1.0     | 94.1 ± 3.3      | 96.1 ± 2.8      |

## 4. PointGPT Pre-training

To pretrain PointGPT, run the following command. 

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/<MODEL_NAME>/pretrain.yaml --exp_name <output_file_name>
```

To post-pretrain PointGPT, run the following command. 

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/<MODEL_NAME>/post_pretrain.yaml --exp_name <output_file_name> --finetune_model
```

## 5. PointGPT Fine-tuning

Fine-tuning on ScanObjectNN, run the following command:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run the following command:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run the following command:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/<MODEL_NAME>/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run the following command:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run the following command:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300 --model_name <MODEL_NAME>
```

## 6. Visualization

Visulization of pre-trained model on validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/<MODEL_NAME>/pretrain.yaml --exp_name <name>
```

<div  align="center">    
 <img src="./figures/vis.png" width = "900"  align=center />
</div>

## 7. Ablation studies on post-pre-training stage 
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3"><u>ScanObjectNN</u></th>
      <th colspan="2"><u>ModelNet40</u></th>
      <th colspan="2">ShapeNetPart</th>
    </tr>
    <tr>
      <th>OBJ_BG</th>
      <th>OBJ_ONLY</th>
      <th>PB_T50_RS</th>
      <th>1k P</th>
      <th>8k P</th>
      <th>Cls.mIoU</th>
      <th>Inst.mIoU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="8">without post-pre-training</th>
    </tr>
    <tr>
      <td><i>PointGPT-B</i></td>
      <td>93.6</td>
      <td>92.5</td>
      <td>89.6</td>
      <td>94.2</td>
      <td>94.4</td>
      <td>84.5</td>
      <td>86.4</td>
    </tr>
    <tr>
      <td><i>PointGPT-L</i></td>
      <td>95.7</td>
      <td>94.1</td>
      <td>91.1</td>
      <td>94.5</td>
      <td>94.7</td>
      <td>84.7</td>
      <td>86.5</td>
    </tr>
    <tr>
      <th colspan="8">with post-pre-training</th>
    </tr>
    <tr>
      <td><i>PointGPT-B</i></td>
      <td>95.8 <span style="color:green">(+2.2)</span></td>
      <td>95.2 <span style="color:green">(+2.7)</span></td>
      <td>91.9 <span style="color:green">(+2.3)</span></td>
      <td>94.4 <span style="color:green">(+0.2)</span></td>
      <td>94.6 <span style="color:green">(+0.2)</span></td>
      <td>84.5 <span style="color:green">(+0.0)</span></td>
      <td>86.5 <span style="color:green">(+0.1)</span></td>
    </tr>
    <tr>
      <td><i>PointGPT-L</i></td>
      <td>97.2 <span style="color:green">(+1.5)</span></td>
      <td>96.6 <span style="color:green">(+2.5)</span></td>
      <td>93.4 <span style="color:green">(+2.3)</span></td>
      <td>94.7 <span style="color:green">(+0.2)</span></td>
      <td>94.9 <span style="color:green">(+0.2)</span></td>
      <td>84.8 <span style="color:green">(+0.1)</span></td>
      <td>86.6 <span style="color:green">(+0.1)</span></td>
    </tr>
  </tbody>
</table>


## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Reference

```
@article{chen2023pointgpt,
  title={PointGPT: Auto-regressively Generative Pre-training from Point Clouds},
  author={Chen, Guangyan and Wang, Meiling and Yang, Yi and Yu, Kai and Yuan, Li and Yue, Yufeng},
  journal={arXiv preprint arXiv:2305.11487},
  year={2023}
}
```
