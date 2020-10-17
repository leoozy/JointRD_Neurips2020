import math, torch, pdb
from tools import utils
from model import loss_KD_fn
import torch.nn.functional as F

def Cosine(min_v, max_v, epoch, epoch_max = 300):
    res = min_v + (max_v - min_v) * (1 + math.cos(math.pi * epoch / epoch_max)) / 2.
    return res

        
def train(data_loader, model, criterion, optimizer_t, optimizer_m, epoch, stage, logger, args, epoch_dict):
    loss_avg = utils.AverageMeter()
    mse_avg  = utils.AverageMeter()
    top1_cnn = utils.AverageMeter()
    top5_cnn = utils.AverageMeter()
    top1_res = utils.AverageMeter()
    top5_res = utils.AverageMeter()
    global_step = epoch * len(data_loader)
    model.train()
    if "TA" in stage:
        model.module.teacher.eval()
    else:
        model.module.teacher.train()
        
    logger.log("stage: {}".format(stage))
    m = Cosine(min_v = 0.5, max_v = 1., epoch = epoch)
    for step, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        num_samples = images.size(0)
        optimizer_t.zero_grad()
        optimizer_m.zero_grad()
        if "TA" in stage:
            logits, logits_teacher, loss_dis = model(
                x = images, 
                stage = stage, 
                epoch = epoch, 
                batch_pro = args.batch_pro, 
                windowsize = args.windowsize)
            if stage == "TA1":
                loss = loss_KD_fn(criterion, logits, logits_teacher, 
                                       targets = labels, alpha = args.alpha, temperature = args.temperature)
            elif stage == "TA2":
                loss = 0.
                for logit_student in logits[:-1]:

                    #loss += loss_KD_fn(criterion, logit_student, logits_teacher, 
                                      # targets = labels, alpha = args.alpha, temperature = args.temperature) * m * 0.25
                    loss += criterion(logit_student, labels) * m
                    
                #loss += loss_KD_fn(criterion, logits[-1], logits_teacher, 
                 #                      targets = labels, alpha = args.alpha, temperature = args.temperature) * (1.0 - 3*m*0.25)
                loss += criterion(logits[-1], labels) 
            loss_avg.update(loss.detach().item(), num_samples)
            if loss_dis is not None:
                for loss_d in loss_dis[:-1]:
                    loss += loss_d.mean() * m * 0.25 * args.dis_weight
                mse_avg.update(loss_dis[-1].detach().mean().item(), num_samples)
                loss += loss_dis[-1].mean() *  args.dis_weight
            
            #10^-3 for 32x32 image
            #10^-4 for 224x224 scale classification task
            #10^-5 for detection and segmentation task
            if isinstance(logits, list):
                prec1_cnn, prec5_cnn = utils.accuracy(logits[-1].detach(), labels, topk=(1, 5))
            else:
                prec1_cnn, prec5_cnn = utils.accuracy(logits.detach(), labels, topk=(1, 5))
            top1_cnn.update(prec1_cnn.item(), num_samples)
            top5_cnn.update(prec5_cnn.item(), num_samples)
        elif "RES_NMT" in stage:
            logits = model(images, stage = stage)
            loss = criterion(logits, labels)
            
            ## train mask
            mask = []
            mask_log = []
            for name, param in model.named_parameters():
                if 'mask' in name and "teacher" in name:
                    mask.append(param.view(-1))
                    mask_log.append(param.detach())

            mask = torch.cat(mask)
            error_sparse = args.sparse_lambda * torch.norm(mask, 1)
            error_sparse.backward()

            prec1_res, prec5_res = utils.accuracy(logits.detach(), labels, topk=(1, 5))
            top1_res.update(prec1_res.item(), num_samples)
            top5_res.update(prec5_res.item(), num_samples)
            loss_avg.update(loss.detach().data.item(), num_samples)
        elif "CNN_NMT" in stage:
            logits = model(images, stage = stage)
            loss = criterion(logits, labels)
            prec1_cnn, prec5_cnn = utils.accuracy(logits.detach(), labels, topk=(1, 5))
            top1_cnn.update(prec1_cnn.item(), num_samples)
            top5_cnn.update(prec5_cnn.item(), num_samples)
            loss_avg.update(loss.detach().data.item(), num_samples)
        elif "RES_KD" in stage:
            logit_student, logits_teacher = model(images, stage = stage)
            loss = loss_KD_fn(criterion, logit_student, logits_teacher, 
                                targets = labels, alpha = args.alpha, temperature = args.temperature)
            prec1_res, prec5_res = utils.accuracy(logit_student.detach(), labels, topk=(1, 5))
            top1_res.update(prec1_res.item(), num_samples)
            top5_res.update(prec5_res.item(), num_samples)
            loss_avg.update(loss.detach().data.item(), num_samples)
        else:
            raise NameError("invalide stage nanme") 
        loss.backward()
        optimizer_t.step()
        if epoch >= 1:
            optimizer_m.step()
        
        epochs = epoch_dict[stage]
        if step % 100 == 0 or step == len(data_loader) - 1:
            logger.log("Train, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Loss_dis: {:.4f}, Prec@(cnn1, res1, cnn5, res5): {:.4%},{:.4%}, {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg, mse_avg.avg, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg))
            
        global_step += 1
    logger.log("mask:")
    logger.log(mask_log)
    logger.log("Train, Epoch: [{:3d}/{}], Final Prec: cnn, res@1: {:.4%}, {:.4%},  Final Prec: cnn, res@5: {:.4%}, {:.4%} Loss: {:.4f}".format(
                epoch, epochs, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg, loss_avg.avg))



def valid(data_loader, model, criterion, epoch, global_step, stage, logger, args, epoch_dict):

    loss_avg   = utils.AverageMeter()
    top1_cnn   = utils.AverageMeter()
    top5_cnn   = utils.AverageMeter()
    top1_res   = utils.AverageMeter()
    top5_res   = utils.AverageMeter()
    global_step = epoch * len(data_loader)
    
    model.eval()
    logger.log("stage: {}".format(stage))
    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
    
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            num_samples = images.size(0)
            
            if "TA" in stage:
                logits, logits_teacher, _ = model(
                    images, 
                    stage = stage, 
                    epoch = epoch, 
                    batch_pro = args.batch_pro, 
                    windowsize = args.windowsize)
                #loss = loss_KD_fn(criterion, logits, logits_teacher, targets = labels, alpha = args.alpha, temperature = args.temperature)
                if isinstance(logits, list):
                    logits = logits[-1]
                loss = criterion(logits, labels)
                prec1_cnn, prec5_cnn = utils.accuracy(logits.detach(), labels, topk=(1, 5))
                loss_avg.update(loss.detach().item(), num_samples)
                top1_cnn.update(prec1_cnn.item(), num_samples)
                top5_cnn.update(prec5_cnn.item(), num_samples)
                
                
            elif "RES_NMT" in stage:
                logits = model(images, stage = stage)
                loss = criterion(logits, labels)
                prec1_res, prec5_res = utils.accuracy(logits, labels, topk=(1, 5))
                top1_res.update(prec1_res.item(), num_samples)
                top5_res.update(prec5_res.item(), num_samples)
                loss_avg.update(loss.data.item(), num_samples)
                
            elif "CNN_NMT" in stage:
                logits = model(images, stage = stage)
                loss = criterion(logits, labels)
                prec1_cnn, prec5_cnn = utils.accuracy(logits, labels, topk=(1, 5))
                top1_cnn.update(prec1_cnn.item(), num_samples)
                top5_cnn.update(prec5_cnn.item(), num_samples)
                loss_avg.update(loss.data.item(), num_samples)
            elif "RES_KD" in stage:
                logit_student, logits_teacher = model(images, stage = stage)
                loss = loss_KD_fn(criterion, logit_student, logits_teacher, 
                                targets = labels, alpha = args.alpha, temperature = args.temperature)
                prec1_res, prec5_res = utils.accuracy(logit_student.detach(), labels, topk=(1, 5))
                top1_res.update(prec1_res.item(), num_samples)
                top5_res.update(prec5_res.item(), num_samples)
                loss_avg.update(loss.detach().data.item(), num_samples)
            else:
                raise NameError("invalide stage nanme") 
    
            epochs = epoch_dict[stage]
            if step % 100 == 0 or step == len(data_loader) - 1:
                logger.log("Valid, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                            "Loss: {:.4f}, Prec@(cnn1, res1, cnn5, res5): {:.4%},{:.4%}, {:.4%}, {:.4%}".format(
                                epoch, epochs, step, len(data_loader), 
                                loss_avg.avg, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg))
    
            global_step += 1
    
        logger.log("Valid, Epoch: [{:3d}/{}], Final Prec: cnn, res@1: {:.4%}, {:.4%},  Final Prec: cnn, res@5: {:.4%}, {:.4%} Loss: {:.4f}".format(
                    epoch, epochs, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg, loss_avg.avg))
        
        if "RES" in stage:
            return top1_res.avg
        else:
            return top1_cnn.avg