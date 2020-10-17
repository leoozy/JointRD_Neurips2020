import math, torch
from tools import utils
from model import loss_KD_fn

        
def train(data_loader, model, criterion, optimizer, epoch, stage, logger, args):
    loss_avg = utils.AverageMeter()
    top1_res = utils.AverageMeter()
    top5_res = utils.AverageMeter()
    global_step = epoch * len(data_loader)
    model.train()
        
    logger.log("stage: {}".format(stage))

    for step, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        num_samples = images.size(0)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        prec1_res, prec5_res = utils.accuracy(logits.detach(), labels, topk=(1, 5))
        top1_res.update(prec1_res.item(), num_samples)
        top5_res.update(prec5_res.item(), num_samples)
        loss_avg.update(loss.detach().data.item(), num_samples)

        loss.backward()
        optimizer.step()
        
        epochs = args.baseline_epochs
        if step % 100 == 0 or step == len(data_loader) - 1:
            logger.log("Train, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Prec@(res1, res5):  {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg,   top1_res.avg,  top5_res.avg))
            
        global_step += 1
    logger.log("Train, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Prec@(res1, res5):  {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg, top1_res.avg,  top5_res.avg))



def valid(data_loader, model, criterion, epoch, global_step, stage, logger, args):
    
    loss_avg = utils.AverageMeter()
    top1_res = utils.AverageMeter()
    top5_res = utils.AverageMeter()
    global_step = epoch * len(data_loader)
    model.eval()
        
    logger.log("stage: {}".format(stage))

    for step, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        num_samples = images.size(0)

        logits = model(images)
        loss = criterion(logits, labels)
        prec1_res, prec5_res = utils.accuracy(logits.detach(), labels, topk=(1, 5))
        top1_res.update(prec1_res.item(), num_samples)
        top5_res.update(prec5_res.item(), num_samples)
        loss_avg.update(loss.detach().data.item(), num_samples)
        
        epochs = args.baseline_epochs
        if step % 100 == 0 or step == len(data_loader) - 1:
            logger.log("Valid, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f},  Prec@(res1, res5):  {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg,   top1_res.avg,  top5_res.avg))
            
        global_step += 1
    logger.log("Valid, Epoch: [{:3d}/{}], Step: [{:3d}/{}], " \
                        "Loss: {:.4f}, Prec@(res1, res5):  {:.4%}, {:.4%}".format(
                            epoch, epochs, step, len(data_loader), 
                            loss_avg.avg,  top1_res.avg,  top5_res.avg))
    
    return top1_res.avg