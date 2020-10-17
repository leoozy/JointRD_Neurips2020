from .distillerkd import Distiller
from .resnet_cifar import resnet50 as resnet50_cifar
from .resnet_cifar import resnet34 as resnet34_cifar
from .resnet_cifar import resnet18 as resnet18_cifar
from .resnet_cifar_dirac import resnet50 as resnet50_cifar_dirac
from .resnet_cifar_dirac import resnet34 as resnet34_cifar_dirac
from .resnet_cifar_dirac import resnet18 as resnet18_cifar_dirac
from .resnet_imagenet import resnet50 as resnet50_imagenet
from .resnet_imagenet import resnet34 as resnet34_imagenet
from .resnet_imagenet import resnet18 as resnet18_imagenet
from .distiller2   import Distiller as Distiller2
from .resnet_fus   import resnet50 as resnet50_fus
from .resnet_imagenet_diraconv import resnet50 as resnet50_imagenet_diraconv
from .resnet_imagenet_diraconv import resnet34 as resnet34_imagenet_diraconv
from .resnet_imagenet_diraconv import resnet18 as resnet18_imagenet_diraconv
from .diraconv import DiracConv2d


model_dict = {
    "resnet50_cifar": resnet50_cifar,
    "resnet34_cifar": resnet34_cifar,
    "resnet18_cifar": resnet18_cifar,
    "resnet50_cifar_dirac": resnet50_cifar_dirac,
    "resnet34_cifar_dirac": resnet34_cifar_dirac,
    "resnet18_cifar_dirac": resnet18_cifar_dirac,
    "Distiller"     : Distiller,
    "Distiller2"    : Distiller2,
    "resnet50_fus"  : resnet50_fus,
    "resnet50_imagenet": resnet50_imagenet,
    "resnet34_imagenet": resnet34_imagenet,
    "resnet18_imagenet": resnet18_imagenet,
    "resnet18_imagenet_diraconv": resnet18_imagenet_diraconv,
    "resnet34_imagenet_diraconv": resnet34_imagenet_diraconv,
    "resnet50_imagenet_diraconv": resnet50_imagenet_diraconv,
}