# -*- coding: utf-8 -*-
from __future__        import absolute_import
from __future__        import division
from __future__        import print_function
from model             import model_dict, Distiller, Distiller2, DiracConv2d
from tools.utils       import *
from tools import load_dataset, SGD, Adam
from process.multikd_train import * 
import torch.nn               as nn
import numpy                  as np
import torchvision.datasets   as dset
import tools.utils            as utils
import torchvision.transforms as transforms
import torchvision.datasets   as datasets
import torchvision.models     as models
import os, pickle, sys, time, torch, argparse, pdb, math


parser = argparse.ArgumentParser("IMAGENET")
parser.add_argument('--save_dir'         ,   type = str,      default = "./result")
parser.add_argument('--seed'             ,   type = int,      default = 1         )
parser.add_argument('--cutout_length'    ,   type = int,      default = 0         )
parser.add_argument('--num_workers'      ,   type = int,      default = 16        ) 
parser.add_argument('--learning_rate'    ,   type = float,    default = 0.1       )
parser.add_argument('--momentum'         ,   type = float,    default = 0.9       )
parser.add_argument('--dis_weight'       ,   type = float,    default = 1e-3      )
parser.add_argument('--weight_decay'     ,   type = float,    default = 0.0005    )
parser.add_argument('--alpha'            ,   type = float,    default = 0.9       )
parser.add_argument('--temperature'      ,   type = float,    default = 4.        )
parser.add_argument('--baseline_epochs'  ,   type = int,      default = 300       ) 
parser.add_argument('--batch_size'       ,   type = int,      default = 64        )
parser.add_argument('--data_dir'         ,   type = str,      default = ""        )
parser.add_argument('--stage'            ,   type = str,      default = ""        )
parser.add_argument('--aim'              ,   type = str,      default = ""        )
parser.add_argument('--model_dir'        ,   type = str,      default = ""        )
parser.add_argument('--tmodel_name'      ,   type = str,      default = ""        )
parser.add_argument('--smodel_name'      ,   type = str,      default = ""        )
parser.add_argument('--batch_pro'        ,   type = int,      default = 1         )
parser.add_argument('--windowsize'       ,   type = int,      default = 15        )
parser.add_argument('--start_epoch'      ,   type = int,      default = 0         )
parser.add_argument('--dataset'          ,   type = str,      default = ""        )
parser.add_argument('--kd_type'          ,   type = str,                          )
parser.add_argument('--load'             ,   type = str,      default = "",       ) 
parser.add_argument('--procedure'        ,   action='append', default = []        )
parser.add_argument('--lr_sch'           ,   type=str,                            )
parser.add_argument('--stone'            , type=int, nargs='+', default=[100, 150], 
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--config_s'          ,   type = str,                          )
parser.add_argument('--config_t'          ,   type = str,                          )
parser.add_argument('--dc'                ,   type = float,                          )

args   = parser.parse_args()
logger = prepare_logger(args)

bc_dict = {
        
       "resnet18_imagenet": {'teacher_bc' : [[1]*2, [1]*2, [1]*2, [1]*2],
                             'student_bc' : [[0]*2, [1,0], [1,0], [1,0]]},
       "resnet34_imagenet": {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet50_imagenet": {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet34_cifar":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet18_cifar":    {'teacher_bc' : [[1]*2, [1]*2, [1]*2, [1]*2],
                             'student_bc' : [[0]*2, [1,0], [1,0], [1,0]]},
       "resnet50_cifar":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
    
    
       "resnet34_cifar_dirac":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet50_cifar_dirac":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet18_cifar_dirac":    {'teacher_bc' : [[1]*2, [1]*2, [1]*2, [1]*2],
                             'student_bc' : [[0]*2, [1,0], [1,0], [1,0]]},
    
       "resnet18_imagenet_diraconv":    {'teacher_bc' : [[1]*2, [1]*2, [1]*2, [1]*2],
                             'student_bc' : [[0]*2, [1,0], [1,0], [1,0]]},
       "resnet34_imagenet_diraconv":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},
       "resnet50_imagenet_diraconv":    {'teacher_bc' : [[1]*3, [1]*4, [1]*6, [1]*3],
                             'student_bc' : [[0]*3, [1,0,0,0], [1,0,0,0,0,0], [1,0,0]]},

}


def trainer(train_loader, valid_loader, model, criterion,  optimizer_t, optimizer_s=None, lr_scheduler=None, stage=None):
    logger.log("start training..." + stage)
    best_top1  = 0.0
    epochs     = args.baseline_epochs
    start_time = time.time()
    epoch_time = utils.AverageMeter()
    
    for epoch in range(args.start_epoch, epochs):
        ##################################adjust learning rate##################################
        if args.lr_sch == "cosine":
            if optimizer_t is not None:
                adjust_learning_rateD(optimizer_t, epoch, epochs, 
                                  lr_max = args.learning_rate, lr_min = args.learning_rate * 1e-3)
            if optimizer_s is not None:
                adjust_learning_rateD(optimizer_s, epoch, epochs, 
                                      lr_max = args.learning_rate, lr_min = args.learning_rate * 1e-3)
        elif args.lr_sch == "imagenet":
            if optimizer_t is not None:
                adjust_learning_rateA(optimizer_t, epoch, args)
            if optimizer_s is not None:
                adjust_learning_rateA(optimizer_s, epoch, args)
        elif args.lr_sch == "step":
            if optimizer_t is not None:
                adjust_learning_rateS(optimizer_t, epoch, args)
            if optimizer_s is not None:
                adjust_learning_rateS(optimizer_s, epoch, args)
        else:
            raise NameError("lrsch name error")
        ########################################################################################
        
        lr = optimizer_t.param_groups[0]["lr"] if optimizer_t else  optimizer_s.param_groups[0]["lr"] 
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        logger.log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s} LR={:}'.format(
            args.smodel_name, epoch, epochs, time_string(), need_time, lr))
        train(train_loader, model, criterion, optimizer_t, optimizer_s, epoch, stage, logger, args)
        global_step = (epoch + 1) * len(train_loader) - 1
        valid_top1 = valid(valid_loader, model, criterion, epoch,
                           global_step, stage=stage, logger=logger, args=args)
        
        if epoch == 0 or best_top1 < valid_top1:
            best_top1 = valid_top1
            is_best = True
        else:
            is_best = False
            
        if epoch >= 89:
            utils.save_checkpoint(model, logger.path('info'), 
                                  is_best=is_best, pre = args.aim + "_" + "epoch_" + str(epoch) + "_" + stage)
  
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
           

    logger.log("Final best valid Prec@1: {:.4%}".format(best_top1))

    
def train_nmt(train_loader, valid_loader, model, criterion, stage):
 
    logger.log("*"*10 + "TRAIN NMT" + "*"*10)
    if "RES_NMT" in stage:
        
        optimizer = SGD(params=model.module.teacher.parameters(), lr=args.learning_rate, 
                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        trainer(train_loader, valid_loader, model, criterion, optimizer, 
                optimizer_s=None, lr_scheduler = None, stage=stage)
        
    elif "CNN_NMT" in stage:
        
        optimizer = SGD(params=model.module.student.parameters(), lr=args.learning_rate, 
                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        trainer(train_loader, valid_loader, model, criterion, optimizer_t = None, 
                optimizer_s=optimizer, lr_scheduler = None, stage=stage)
            

    
def Ta1(train_loader, valid_loader, model, criterion, stage):

        
    logger.log("*"*10 + "TRAIN TA1" + "*"*10)
    logger.log("not training param:")
    teacher_params = []
    student_params_wd = []
    student_params_others = []
    
            
    for k, v in model.named_parameters():
        if "teacher" in k:
            #print("teacer:{}".format(k))
            teacher_params.append(v)
        else:
            if v.dim() == 1 and ("alpha" in k or "beta" in k or "deta" in k):
                print(k)
                student_params_others.append(v)
            else:
                #print("student:{}".format(k))
                student_params_wd.append(v)

    groups = [{'params': student_params_wd, 'weight_decay': args.weight_decay},
              {'params': student_params_others}]
    optimizer_s = SGD(params=groups, lr=args.learning_rate, momentum=args.momentum, nesterov=True)
    optimizer_t = SGD(params=teacher_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    trainer(train_loader, valid_loader, model, criterion, optimizer_t, optimizer_s, lr_scheduler=None, stage=stage)

    
def main(**kwargs):   
    
    
    ##########################reproductive###################################
    model_dict = kwargs.get("model_dict")
    if not torch.cuda.is_available():
        logger.log("no gpu device available")
        sys.exit(1)
    torch.backends.cudnn.benchmark    = True
    #torch.backends.cudnn.deterministic= True
    prepare_seed(args.seed)
    logger.log("finish seed")
    #########################################################################
    
    ##########################dataset########################################
    logger.log("preparing data...")
    num_classes = 0
    
    if args.dataset == "cifar10" or args.dataset == "cifar100":

        input_size, channels_in, num_classes, train_data, valid_data = load_dataset(
            dataset = args.dataset, data_dir = args.data_dir, cutout_length = args.cutout_length,
            validation = True, auto_aug  = False)

        train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = args.batch_size,
            shuffle = True, num_workers= args.num_workers, pin_memory = True)

        valid_loader = torch.utils.data.DataLoader(dataset = valid_data, batch_size = args.batch_size,
            shuffle = False, num_workers= args.num_workers, pin_memory = True)
        
    elif args.dataset == "imagenet":
        num_classes = 1000
        traindir = os.path.join(args.data_dir, 'train')
        valdir   = os.path.join(args.data_dir, 'val'  )
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
    
        train_loader = torch.utils.data.DataLoader(
             train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None)
    
        valid_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ##################################################################################
    
    criterion  = nn.CrossEntropyLoss().cuda()

    model  = Distiller(  student=model_dict[args.smodel_name], 
                         teacher=model_dict[args.tmodel_name], 
                         tblock_choices=bc_dict[args.tmodel_name]["teacher_bc"], 
                         sblock_choices=bc_dict[args.smodel_name]["student_bc"],
                         kd_type=args.kd_type,
                         num_classes=num_classes,
                         logger=logger)
    '''
    model = Distiller2(  student=model_dict[args.smodel_name], 
                         teacher=model_dict[args.tmodel_name], 
                         config_t=args.config_t, 
                         config_s=args.config_s,
                         kd_type=args.kd_type,
                         num_classes=num_classes,
                         logger=logger)
    '''
    #prepare_seed(args.seed)
    
    if args.model_dir and "JOINT" not in args.procedure:
        pretrained_dict = torch.load(args.model_dir)["model_state_dict"]
        pretrained_dict = {k[7:].replace("stages", "layers"):v for k,v in pretrained_dict.items() if "margin" not in k}
        model_state_dict = model.state_dict()
        for i, (k, v) in enumerate(pretrained_dict.items()):
            ###old model modified
            if "teacher.bn" in k:
                k = "teacher.conv1.1." + k.split(".")[-1]
            if "teacher.conv1.weight" in k:
                k = "teacher.conv1.0.weight"
            #####################
            if "teacher" in k:
                model_state_dict[k] = v

        model.load_state_dict(model_state_dict, strict = True)
        model.reset_margin()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    model.module.reset_margin()
    #valid(valid_loader, model, criterion, 0, global_step=0, stage=args.stage, logger=logger, args=args)

    logger.log('model ====>>>>:\n{:}'.format(model))     
    logger.log('-'*50)
    logger.log('  Teacher model params: %.2fM' % (sum(p.numel() for p in model.module.teacher.parameters())/1000000.0))
    logger.log('  Student model params: %.2fM' % (sum(p.numel() for p in model.module.student.parameters())/1000000.0))
    logger.log('-'*50)
    logger.log('train_data : {:}'.format(train_loader.dataset))
    logger.log('valid_data : {:}'.format(valid_loader.dataset))

    if "RES_NMT" in args.procedure or "RES_KD" in args.procedure or "CNN_NMT" in args.procedure:
        train_nmt(train_loader, valid_loader, model, criterion, stage = args.procedure[0])
        
    if "TA" in args.procedure or "JOINT" in args.procedure or "KD" in args.procedure or "KL" in args.procedure:
        Ta1(train_loader, valid_loader, model, criterion, stage = args.procedure[0])
    
    
if __name__ == "__main__":
    main(model_dict = model_dict)
