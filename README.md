# Simple PyTorch Implementation of Barlow Twins
An easy-to-use modular implementation of [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230).
The code is adapted from [the original github repo](https://github.com/facebookresearch/barlowtwins),
drawing inspiration from [lucidrains wonderful BYOL implementation](https://github.com/lucidrains/byol-pytorch)
(along with using their code for extracting latent representations, since it was far more elegant than my own).

## Usage
Using the package is relatively straightforward:

```python
from Twins.barlow import *
from Twins.transform_utils import *
import torch
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#This is just any generic model
model = torchvison.some_model

#Optional: define transformations for your specific dataset.
#Generally, it is best to use the original augmentations in the
#paper, replacing the Imagenet normalization with the normalization
#for your dataset.

transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size,
                                            interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

#For the transform argument for the dataset, pass in 
# Twins.transform_utils.Transform(transform_1, transform_2)
#If transforms are None, the Imagenet default is used.
dataset = dsets.some_dataset(**kwargs, 
                             transform=Transform(transform, transform))

loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
#Make the BT instance, passing the model, the latent rep layer id,
# hidden units for the projection MLP, the tradeoff factor,
# and the loss scale.
learner = BarlowTwins(model, 'avg_pool', [512,1024, 1024, 1024],
                      3.9e-3, 1)

optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)

#Single training epoch
for batch_idx, ((x1,x2), _) in enumerate(loader):
    loss = learner(x1, x2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
That is basically it. Hopefully this is helpful!

### References:
```bibtex
@article{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  journal={arXiv preprint arXiv:2103.03230},
  year={2021}
}
```
```bibtex
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{chen2020exploring,
    title={Exploring Simple Siamese Representation Learning}, 
    author={Xinlei Chen and Kaiming He},
    year={2020},
    eprint={2011.10566},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
