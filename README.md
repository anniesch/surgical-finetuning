# Surgical Fine-Tuning Improves Adaptation to Distribution Shifts

This repo provides starter code for the following paper published at ICLR 2023: 
> [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://openreview.net/pdf?id=APuPRxjHvZ).

The purpose of this repo is to provide a sample implementation of surgical fine-tuning, which is simple to add to existing codebases: just optimize the parameters in the desired layers. Here we provide some sample code for running on CIFAR-C and ImageNet-C datasets.
The fine-tuning pipeline is all in `main.py` with argument configs for the datasets in `config/`.

## Environment

Create an environment with the following command:
```
conda env create -f conda_env.yml
```


## **Sample Commands for Surgical Fine-Tuning**

Before running, download the data ([CIFAR-10C](https://zenodo.org/record/2535967) or [ImageNet-C](https://zenodo.org/record/2235448)) and update the paths in the configs accordingly.

```
python main.py --config-name='cifar-10c' args.train_n=1000 args.seed=0 data.corruption_types=['defocus_blur'] wandb.use=True
python main.py --config-name='cifar-10c' args.train_n=1000 args.seed=0 data.corruption_types=[frost,gaussian_blur,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur] wandb.use=False args.auto_tune=none args.epochs=15 
python main.py --config-name='imagenet-c' args.train_n=5000 args.seed=0 data.corruption_types=[brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur] wandb.use=False args.auto_tune=none args.epochs=10
```

## Running Auto-RGN
```
python main.py --config-name='cifar-10c' args.train_n=1000 args.seed=0 data.corruption_types=[frost,gaussian_blur,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,saturate,shot_noise,snow,spatter,speckle_noise,zoom_blur]  wandb.use=True args.auto_tune=RGN args.epochs=15

python main.py --config-name='imagenet-c' args.train_n=5000 args.seed=2 data.corruption_types=[brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur] wandb.use=False args.auto_tune=RGN args.epochs=10

```

# Citing Surgical Finetuning
If surgical fine-tuning or this repository is useful in your own research, you can use the following BibTeX entry:

    @article{lee2022surgical,
          title={Surgical fine-tuning improves adaptation to distribution shifts},
          author={Lee, Yoonho and Chen, Annie S and Tajwar, Fahim and Kumar, Ananya and Yao, Huaxiu and Liang, Percy and Finn, Chelsea},
          journal={International Conference on Learning Representations},
          year={2023}
        }


