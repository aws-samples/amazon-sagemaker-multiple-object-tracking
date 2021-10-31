from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    #torch.set_num_threads(4)
    
    resume_from_epoch = 0
    
    for try_epoch in range(opt.num_epochs, 0, -1):
        if os.path.exists(opt.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
    
    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    
    f = open(opt.data_cfg)
    data_config = json.load(f)
    
    trainset_paths = data_config['train']
    valset_paths = data_config['test']
    dataset_root = data_config['root']
    f.close()
    
    transforms = T.Compose([T.ToTensor()])
    train_dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    
    print('Loading data finished.')
    val_dataset = None
    if valset_paths and valset_paths != trainset_paths:
        val_dataset = Dataset(opt, dataset_root, valset_paths, (1088, 608), transforms=transforms)
    
    opt = opts().update_dataset_info_and_set_heads(opt, train_dataset, val_dataset)
    
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    pretrained = not opt.load_model.endswith('pth')
    model = create_model(opt.arch, opt.heads, opt.head_conv, pretrained=pretrained)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    if val_dataset:
        #val_batch_size = opt.batch_size if opt.num_workers < 1 else opt.num_workers*opt.batch_size
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.batch_size_val,
            num_workers=opt.num_workers_val
        )
    
    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    print(f'resume_from_epoch: {resume_from_epoch}')
    start_epoch = 0
    
    if resume_from_epoch > 0:
        print('resume_from_epoch greater than 1: ', resume_from_epoch)
        filepath = opt.checkpoint_format.format(epoch=resume_from_epoch)
        model, optimizer, start_epoch = load_model(
            model, filepath, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    elif opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
        resume_from_epoch = start_epoch + 1
    
    for epoch in range(resume_from_epoch, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        
        log_std = 'epoch: {} |'.format(epoch)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            
            log_std += 'train_{} {:8f} |'.format(k, v)
            logger.write('train_{} {:8f} | '.format(k, v))
        
        print(log_std)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            
            if val_dataset:
                print('Starting validation...')
                log_dict_val, _ = trainer.val(epoch, val_loader)
                log_std = 'epoch: {} |'.format(epoch)
                logger.write('epoch: {} |'.format(epoch))
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)

                    log_std += 'val_{} {:8f} |'.format(k, v)
                    logger.write('val_{} {:8f} | '.format(k, v))

                print(log_std)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                        epoch, model, optimizer)
        
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            
            print('Drop LR to', lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)