import math, torch, time
from tools import utils

def Cosine(min_v, max_v, epoch, epoch_max=60):
    if epoch <= epoch_max:
        res = min_v + (max_v - min_v) * (1 + math.cos(math.pi * epoch / epoch_max)) / 2.
    else:
        res = min_v
    return res

def train(data_loader, model, criterion, optimizer_t, optimizer_s, optimizer_mask, epoch, stage, logger, args):
    
    [loss_avg, mse_avg, top1_cnn, top5_cnn, top1_res, top5_res] = [utils.AverageMeter() for _ in range(6)]
    global_step = epoch * len(data_loader)
    model.train()
    logger.log("stage: {}".format(stage))
    m = Cosine(min_v = 0.5, max_v = 1., epoch = epoch)
    #m = 1.0
    model.module.reset_margin()
    for step, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        num_samples = images.size(0)
        optimizer_t.zero_grad()
        optimizer_mask.zero_grad()
        if optimizer_s is not None:
            optimizer_s.zero_grad()
            
        if "TA" in stage:
            raise NameError("bad methods")
            
        elif "JOINT" in stage:
                ## teacher and student are jointly trained from scratch
                
            ###train teacher#####################
            
            model.module.teacher.train()
            logits_teacher, teacher_feas = model(images, stage='RES_TA', epoch=epoch) 
            #logits_teacher, _ = model(images, stage='RES_TA', epoch=epoch) 
            loss_teacher = criterion(logits_teacher, labels) 
            loss_teacher.backward()
            optimizer_t.step()
            optimizer_t.zero_grad()
            model.module.teacher.eval()
            #####################################
        
            logits_student, _, loss_dis = model(images, stage=stage, epoch=epoch, teacher_feas=teacher_feas[-1])
            loss = 0.
            for logit_student in logits_student[:-1]:
                KD_TRAIN=False
                if KD_TRAIN:
                    loss += loss_KD_fn(criterion, logit_student, logits_teacher, 
                                       targets=labels, alpha=args.alpha, temperature=args.temperature) * m * 0.25
                else:
                    loss += criterion(logit_student, labels) * m * 0.25
            loss_last = criterion(logits_student[-1], labels) * 0.25
            loss_avg.update(loss_last.detach().item(), num_samples)
            loss += loss_last
            
                
            if loss_dis is not None:
                for loss_d in loss_dis[:-1]:
                    loss += loss_d.mean() * m * 0.25 * args.dis_weight
                mse_avg.update(loss_dis[-1].detach().mean().item(), num_samples)
                loss += loss_dis[-1].mean() * args.dis_weight
            
            #10^-3 for 32x32 image
            #10^-4 for 224x224 scale classification task
            #10^-5 for detection and segmentation task
            if isinstance(logits_student, list):
                prec1_cnn, prec5_cnn = utils.accuracy(logits_student[-1].detach(), labels, topk=(1, 5))
            else:
                prec1_cnn, prec5_cnn = utils.accuracy(logits_student.detach(), labels, topk=(1, 5))
            
            prec1_cnn, prec5_cnn = utils.accuracy(logits_student[-1].detach(), labels, topk=(1, 5))
            prec1_res, prec5_res = utils.accuracy(logits_teacher.detach(), labels, topk=(1, 5))
            ### teacher is only updated by its own loss 

            loss.backward()
            optimizer_s.step()
            optimizer_mask.step()
            top1_cnn.update(prec1_cnn.item(), num_samples)
            top5_cnn.update(prec5_cnn.item(), num_samples)
            top1_res.update(prec1_res.item(), num_samples)
            top5_res.update(prec5_res.item(), num_samples)
            
        elif "RES_NMT" in stage:
            logits = model(images, stage = 'RES_NMT')
            loss = criterion(logits, labels)
            prec1_res, prec5_res = utils.accuracy(logits.detach(), labels, topk=(1, 5))
            top1_res.update(prec1_res.item(), num_samples)
            top5_res.update(prec5_res.item(), num_samples)
            loss_avg.update(loss.detach().data.item(), num_samples)
            loss.backward()
            optimizer_t.step()
            
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
            

        
        epochs = args.baseline_epochs
        if step % 100 == 0 or step == len(data_loader) - 1:
            logger.log("Train, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Loss_dis: {:.4f}, Prec@(cnn1, res1, cnn5, res5): {:.4%},{:.4%}, {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg, mse_avg.avg, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg))
            
        global_step += 1
    mask_param  = []
    for n, v in model.named_parameters():
        if "mask" in n:
            mask_param.append((n, v.detach().cpu().numpy()))
    logger.log(mask_param)
            
    logger.log("m is {}".format(m))
    logger.log("Train, Epoch: [{:3d}/{}], Final Prec: cnn, res@1: {:.4%}, {:.4%},  Final Prec: cnn, res@5: {:.4%}, {:.4%} Loss: {:.4f}".format(
                epoch, epochs, top1_cnn.avg, top1_res.avg, top5_cnn.avg, top5_res.avg, loss_avg.avg))



def valid(data_loader, model, criterion, epoch, global_step, stage, logger, args):

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
            
            
            if "TA" in stage or "JOINT" in stage:
                with torch.no_grad():
                    logits = model(images, stage = 'CNN_NMT')
                    logits_teacher = model(images, stage = 'RES_NMT')
                    prec1_cnn, prec5_cnn = utils.accuracy(logits.detach(), labels, topk=(1, 5))
                    prec1_res, prec5_res = utils.accuracy(logits_teacher.detach(), labels, topk=(1, 5))
                    loss = criterion(logits, labels) 
                loss_avg.update(loss.detach().item(), num_samples)
                top1_cnn.update(prec1_cnn.item(), num_samples)
                top5_cnn.update(prec5_cnn.item(), num_samples)
                top1_res.update(prec1_res.item(), num_samples)
                top5_res.update(prec5_res.item(), num_samples)
                
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
    
            epochs = args.baseline_epochs
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