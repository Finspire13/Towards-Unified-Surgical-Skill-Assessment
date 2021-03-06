# Towards-Unified-Surgical-Skill-Assessment

Codes for [Towards Unified Surgical Skill Assessment](http://www.vie.group/media/pdf/CVPR2021_Puz4Y7Z.pdf) (CVPR 2021).

![ ](https://github.com/Finspire13/Towards-Unified-Surgical-Skill-Assessment/blob/main/overview.png)

## Setup

* Recommended Environment: Python 3.7, Cuda 10.1, PyTorch 1.6.0
* Install dependencies: `pip3 install -r requirements.txt`.

## Data

 1. Complete [the access form of the JIGSAWS dataset](https://cs.jhu.edu/~los/jigsaws/info.php) and get the permission.
 2. Download our processed data for JIGSAWS from [Baidu Yun](https://pan.baidu.com/s/15l7VgVAdOfHSfGGp33cU8A) (PIN:sa67) or [Google Drive](https://drive.google.com/drive/folders/1fTDE764HVOAxUlaxWWc9fe66qSGoPxDi?usp=sharing).
 3. Unzip the files by `zip --fix data.zip --out data_full.zip && unzip data_full.zip`.
 4. Put the data into the parent directory of the codes.
 5. The data includes following sub-directories:

`video_encoded`  : Surgical videos after pre-processing.

`label`  : Ground truth scores of surgical skills.

`feature_resnet101`  : ImageNet-pretrained ResNet features with ten-crop augmentation (**Visual Path Input**).

`kinematics_GT_14_1`  : Kinematic data of the robotic surgical instruments (**Tool Path Input**).

`time_val_1`  : The sequences indicating task completion time (**Proxy Path Input**).

`gesture_prediction`  : Surgical event preditions from [MS-TCN](https://github.com/yabufarha/ms-tcn) models (**Event Path Input**).

As for the clinical dataset used in the paper, it might be released later if approved. 

## Run

Simply run `python3 main.py --config some_config_file.json` .

The config files for our full model under the JIGSAWS 4-fold cross-validation setting are provided in the `configs` folder.

Trained models and Tensorboard logs will be saved in the `result` folder.

Our trained models and logs are provided in the `pre_result` folder.

## Citation

@inproceedings{liu2021towards,
  title={Towards Unified Surgical Skill Assessment},
  author={Liu, Daochang and Li, Qiyue and Jiang, Tingting and Wang, Yizhou and Miao, Rulin and Shan, Fei and Li, Ziyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9522--9531},
  year={2021}
}

## License
MIT
