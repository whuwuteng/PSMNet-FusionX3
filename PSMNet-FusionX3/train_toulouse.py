"""
Training process.
$ python train.py
"""

import os, sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from misc import options
from misc import utils
from misc import metric

cudnn.benchmark = True
cudnn.deterministic = True

def main():
    # Setup workspace and backup files
    cfg = options.get_config()
    workspace = utils.setup_workspace(cfg.workspace)
    if cfg.pretrained is not None:
        logger = utils.Logger(os.path.join(workspace.log, 'train_log.txt'), mode='a')
    else:
        logger = utils.Logger(os.path.join(workspace.log, 'train_log.txt'))
    tf_logger = SummaryWriter(workspace.log)
    logger.write('Workspace: {}'.format(cfg.workspace), 'green')
    logger.write('CUDA: {}, Multi-GPU: {}'.format(cfg.cuda, cfg.multi_gpu), 'green')
    logger.write('To-disparity: {}'.format(cfg.to_disparity), 'green')

    # Define dataloader
    logger.write('Dataset: {}'.format(cfg.dataset_name), 'green')
    train_dataset, val_dataset = options.get_dataset(cfg.dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,#shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, sampler=None,
                              worker_init_fn=lambda work_id: np.random.seed(work_id))
                              # worker_init_fn ensures different sampling patterns for
                              # each data loading thread
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                            num_workers=cfg.workers)

    # Define model
    logger.write('Model: {}'.format(cfg.model_name), 'green')
    model = options.get_model(cfg.model_name)
    # check for multi-gpu
    if cfg.multi_gpu:
        model = nn.DataParallel(model)
    if cfg.cuda:
        model = model.cuda()

    # Define loss function
    criterion = options.get_criterion(cfg.criterion_name)
    if cfg.cuda:
        criterion = criterion.cuda()
    logger.write('Criterion: {}'.format(criterion), 'green')

    # Define optimizer and learning rate scheduler
    optim = options.get_optimizer(cfg.optimizer_name, model.parameters())
    lr_scheduler = options.get_lr_scheduler(cfg.lr_scheduler_name, optim)
    logger.write('Optimizer: {}'.format(optim), 'green')
    if lr_scheduler is not None:
        logger.write('Learning rate schedular: {}'.format(lr_scheduler), 'green')

    # [Optional] load pretrained model
    start_ep = 0
    global_step = 0
    local_start = 0
    if cfg.pretrained is not None:
        start_ep, global_step = utils.load_checkpoint(model, optim, lr_scheduler, cfg.pretrained, cfg.weight_only)
        logger.write('Load pretrained model from {}'.format(cfg.pretrained), 'green')
        #global_step = len(train_dataset) * start_ep # NOTE: global step start from the beginning of the epoch
        local_start = global_step % len(train_dataset)

    # Start training
    logger.write('Start training...', 'green')
    #for ep in range(start_ep, cfg.max_epoch):
    for ep in range(cfg.start_epoch, cfg.max_epoch):
        if lr_scheduler is not None:
            logger.write('Update learning rate: {} --> '.format(lr_scheduler.get_lr()[0]), 'magenta', end='')
            lr_scheduler.step()
            logger.write('{}'.format(lr_scheduler.get_lr()[0]), 'magenta')

        # Train an epoch
        model.train()
        meters = metric.Metrics(cfg.train_metric_field)
        avg_meters = metric.MovingAverageEstimator(cfg.train_metric_field)
        end = time.time()
        for it, data in enumerate(train_loader, local_start):
            # Pack data
            if cfg.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            inputs['left_sd'] = data['left_sd']
            inputs['right_sd'] = data['right_sd']
            target = data['left_d']
            data_time = time.time() - end

            # Inference, compute loss and update model
            end = time.time()
            optim.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, target)
            loss.backward()
            optim.step()
            update_time = time.time() - end
            end = time.time()

            # Measure performance
            pred_np = pred.data.cpu().numpy()
            target_np = target.data.cpu().numpy()
            results = meters.compute(pred_np, target_np)
            avg_meters.update(results)

            # Print results
            if (it % cfg.print_step) == 0:
                logger.write('[{:2d}/{:2d}][{:5d}/{:5d}] data time: {:4.3f}, update time: {:4.3f}, loss: {:.4f}'\
                             .format(ep, cfg.max_epoch, it, len(train_loader), data_time,
                                     update_time, loss.item()))
                avg_results = avg_meters.compute()
                logger.write('   [Average results] ', end='')
                for key, val in avg_results.items():
                    logger.write('{}: {:5.3f} '.format(key, val), end='')
                logger.write('')
               # avg_meters.reset()

            # Log to tensorboard
            if (it % cfg.tflog_step) == 0:
                tf_logger.add_scalar('A-Loss/loss', loss.data, global_step)
                for key, val in results.items():
                    tf_logger.add_scalar('B-Train-Dense-Metric/{}'.format(key), val, global_step)
                if cfg.lr_scheduler_name is not None:
                    tf_logger.add_scalar('C-Learning-Rate', lr_scheduler.get_lr()[0], global_step)
                # refer to https://github.com/zswang666/Stereo-LiDAR-CCVNorm/issues/3
                tf_logger.add_images('A-RGB/left', inputs['left_rgb'].data, global_step)
                tf_logger.add_images('A-RGB/right', inputs['right_rgb'].data, global_step)
                norm_factor = target.data.max(-1)[0].max(-1)[0].max(-1)[0][:, None, None, None]
                tf_logger.add_images('B-sD', inputs['left_sd'].data / norm_factor, global_step)
                tf_logger.add_images('C-Pred', pred.data / norm_factor, global_step)
                tf_logger.add_images('C-Ground-Truth', target.data / norm_factor, global_step)
                if cfg.dump_all_param: # NOTE: this will require a lot of HDD memory
                    for name, param in model.named_parameters():
                        tf_logger.add_histogram(name+'/vars', param.data.clone().cpu().numpy(), global_step)
                        if param.requires_grad:
                            tf_logger.add_histogram(name+'/grads', param.grad.clone().cpu().numpy(), global_step)

            # On-the-fly validation
            if (it % cfg.val_step) == 0:# and not (ep == 0 and it == 0):
                val_avg_results = validate(global_step, val_loader, model, logger, tf_logger, cfg)

            # Save model
            if (it % cfg.save_step) == 0:
                # train loss is not the real loss
                # train loss and test loss is both the 3-px error
                #print(avg_results)
                #print(val_avg_results)
                train_loss = avg_results['err_3px']
                test_loss = val_avg_results['err_3px']
                #print('training loss: ' + str(train_loss))
                #print('testing loss: ' + str(test_loss))
                ckpt_path = utils.save_checkpoint(workspace.ckpt, model, optim, lr_scheduler, ep, global_step, train_loss, test_loss)
                logger.write('Save checkpoint to {}'.format(ckpt_path), 'magenta')

                avg_meters.reset()

            # Update global step
            global_step += 1

            if it >= len(train_dataset):
                local_start = 0
                break

def validate(global_step, loader, model, logger, tf_logger, cfg):
    model.eval()

    pbar = tqdm(loader)
    pbar.set_description('Online validation')
    disp_meters = metric.Metrics(['err_3px'])
    disp_avg_meters = metric.MovingAverageEstimator(['err_3px'])
    with torch.no_grad():
        for it, data in enumerate(pbar):
            # Pack data
            if cfg.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            inputs['left_sd'] = data['left_sd']
            inputs['right_sd'] = data['right_sd']
            target_d = data['left_d']
            img_w = data['width'].item()

            # Inference
            pred = model(inputs)

            # calculate the error
            # depth
            pred_d_np = pred.data.cpu().numpy()
            target_d_np = target_d.data.cpu().numpy()
            disp_results = disp_meters.compute(pred_d_np, target_d_np)
            disp_avg_meters.update(disp_results)

    logger.write('\nValidation results: ', 'magenta')
    disp_avg_results = disp_avg_meters.compute()
    for key, val in disp_avg_results.items():
        logger.write('- [depth] {}: {}'.format(key, val), 'magenta')
        tf_logger.add_scalar('B-Val-Dense-Metric/depth-{}'.format(key), val, global_step)
    logger.write('\n')
    
    # NOTE: remember to set back to train mode after on-the-fly validation
    model.train()

    return disp_avg_results 

if __name__ == '__main__':
    main()
