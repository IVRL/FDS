<div align="center">
<h1>FDS: Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing (CVPR'25) </h1>

<a href="" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-blue" alt="Paper PDF">
</a>
<a href=""><img src="https://img.shields.io/badge/arXiv-blue" alt="arXiv"></a>
<a href="/"><img src="https://img.shields.io/badge/project_page-green" alt="Project Page"></a>

**[School of Computer and Communication Sciences, EPFL](https://www.epfl.ch/labs/ivrl/)**<br>
**[Department of Electrical and Photonics Engineering, DTU](https://orbit.dtu.dk/en/organisations/department-of-electrical-and-photonics-engineering/)**<br>
**[Department of Electrical Engineering, Chalmers University of Technology](https://www.chalmers.se/en/departments/e2/)**

[Yufan Ren](https://yufan-ren.com/), [Zicong Jiang](https://zicongjiang.github.io/zicong-jiang/), [Tong Zhang](https://sites.google.com/view/tong-zhang/), [Søren Otto Forchhammer](https://orbit.dtu.dk/en/persons/s%C3%B8ren-otto-forchhammer/), [Sabine Süsstrunk](https://people.epfl.ch/sabine.susstrunk/)
</div>


```bibtex
@article{ren2025fds,
  title={FDS: Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing},
  author={Ren, Yufan and Jiang, Zicong and Zhang, Tong and Forchhammer, S{\o}ren and S{\"u}sstrunk, Sabine},
  journal={arXiv preprint arXiv:2503.19191},
  year={2025}
}
```
## Overview

Text-guided image editing using Text-to-Image (T2I) models often fails to yield satisfactory results, frequently introducing unintended modifications, such as the loss of local detail and color changes. In this paper, we analyze these failure cases and attribute them to the indiscriminate optimization across all frequency bands, even though only specific frequencies may require adjustment. To address this, we introduce a simple yet effective approach that enables the selective optimization of specific frequency bands within localized spatial regions for precise edits. Our method leverages wavelets to decompose images into different spatial resolutions across multiple frequency bands, enabling precise modifications at various levels of detail. To extend the applicability of our approach, we provide a comparative analysis of different frequency-domain techniques. Additionally, we extend our method to 3D texture editing by performing frequency decomposition on the triplane representation, enabling frequency-aware adjustments for 3D textures. Quantitative evaluations and user studies demonstrate the effectiveness of our method in producing high-quality and precise edits. Further details are available on our project website. 

![framework](./assets/framework.png)

Examples:

![teaser](./assets/teaser.png)

# Install dependencies

```
# step 1: create a new conda environment (tested on Linux)
conda create -n fds python=3.10 pip
# or `conda create --prefix /data/conda/fds python=3.10 pip` if you want to install it in a specific directory

# step 2: activate the environment
conda activate fds
# or `conda activate /data/conda/fds` if you installed it in a specific directory

# step 3: install the dependencies
pip install -r requirements.txt

# step 4: install additional dependencies
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd ..
```

# 2D Editing

##  Simple 2D editing example to start with

```python
python run2d.py # FDS

# which is equivalent to the default setting: python run2d.py --image_path "data/stones.png" --source_prompt "a stack of stone" --target_prompt "a Buddha statue" --dwt_dds --use_dds --J 2 --num_iters 600 --gs 7.5 --seed 24 --keep_low

python run2d.py --use_dds # DDS
```

More cases we prepared are available in `run2d-cases.sh`. 
