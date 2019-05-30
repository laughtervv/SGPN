### SGPN:Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation [<a href="https://arxiv.org/abs/1711.08588">Arxiv</a>]

### Dependencies
- `tensorflow` (1.3.0)
- `h5py`

### Training & Testing 

We firstly split the training set into training part and validation part. SGPN is finetuned on a pre-trained semantic segmentation model with large batchsize. For training,
```bash
python train.py 
```
Use the following scripts to generate results. `valid.py` is used to compute the per-category theshold for group merging. We then use <a href="github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py">Scannet Evaluation</a> to evaluate test results.
```bash
python valid.py
python generate_results.py
```


### Data and Model 

Please refer to `data/` for example h5 file and input list file. A pre-trained model can be downloaded [<a href="https://drive.google.com/file/d/1-e7YCfrLB4zqbFyWfQGe8sm_QFNrr59K/view?usp=sharing">here</a>].

### Citation
If you find our work useful, please consider citing:

        @inproceedings{wang2018sgpn,
            title={SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation},
            author={Wang, Weiyue and Yu, Ronald and Huang, Qiangui and Neumann, Ulrich},
            booktitle={CVPR},
            year={2018}
        }

### Acknowledgemets

This project is built upon [<a href="https://github.com/charlesq34/pointnet">PointNet</a>] and [<a href="https://github.com/charlesq34/pointnet2">PointNet++</a>].
