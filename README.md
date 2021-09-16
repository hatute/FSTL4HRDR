## Introduction

Source code for "Few-Shot Transfer Learning for Hereditary Retinal Diseases Recognition" (early accepted in MICCAI 2021).

### model
In our experiments, we used ResNet50 as the teacher model and ResNet18 as the student model by default. We also integrated many other commonly used models as additional choices in the `models` folder, please modify the individual training files if you select them.

### data
For confidentiality reasons, we do not provide the target dataset in the experiment, but the [BOE dataset](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) and [Cell dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) are publicly available for download. For training, the `dataset` directory needs to be modified in the corresponding dataloader files.

## Usage

### 0. Preprocess the OCT images (optional)

`python3 preprocess.py --path *path to dataset*`

### 1. train projector with SNNL by auxiliary dataset

`python3 train_projector.py -T 10 -a -50 --info "trial 1"`

### 2.transfer learning for teacher model by target dataset

`python3 transfer2teacher.py --methods (ALL/HL/FE) --path_t *path to the .pth file trained by projector*"`

### 3.Student-Teacher Learning for student model by target dataset

`python3 train_student.py --path_t *path to the .pth file trained by teacher*"`

## Contact
Still in the process of improvement, if you find any problems, or have any questions, please feel free to contact me (siweimai@buffalo.edu)
