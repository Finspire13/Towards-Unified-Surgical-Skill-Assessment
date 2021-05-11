# Towards-Unified-Surgical-Skill-Assessment

Codes for ["Towards Unified Surgical Skill Assessment"](Paper to be uploaded) (CVPR 2021).

![ ](https://github.com/Finspire13/Towards-Unified-Surgical-Skill-Assessment/blob/main/overview.png)

## Setup

* Recommended Environment: Python 3.7, Cuda 10.1, PyTorch 1.6.0
* Install dependencies: `pip3 install -r requirements.txt`.

## Data

 1. Complete [the access form of the JIGSAWS dataset](https://cs.jhu.edu/~los/jigsaws/info.php) and get the permission.
 2. Download our processed data for JIGSAWS from [Baidu Yun]() (PIN:).
 3. Join the data chunks by `cat DATA* > data.zip`
 4. Unzip `data.zip` and put the data into the parent directory of the codes.
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

The [config files]() for our full model under the JIGSAWS 4-fold cross-validation setting are provided.









## Output

Results will be saved in a folder named with the `naming` in the config file. 

This output folder will include following sub-directories:

`logs` : A Tensorboard logging file and an numpy logging file.

`models`: Trained models.

`pos_prob`: Probability maps for instruments.

`pos_mask`: Segmentation masks for instruments.

`neg_prob`: Probability maps for non-instruments.

`neg_mask`: Segmentation masks for non-instruments.

## Trained Models

## Citation

TO DO

## License
MIT
