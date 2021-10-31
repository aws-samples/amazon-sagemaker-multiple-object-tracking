import json
import os
import shutil
import subprocess
import sys

dcn_cmd = "cd /DCNv2; sh make.sh"
subprocess.run(dcn_cmd, shell=True)

def copy_files(src, dest):
    src_files = os.listdir(src)
    for file in src_files:
        path = os.path.join(src, file)
        if os.path.isfile(path):
            shutil.copy(path, dest)
            
def train():
    import pprint
    
    pprint.pprint(dict(os.environ), width = 1) 
    
    model_dir = os.environ['SM_MODEL_DIR']
    log_dir = None
    
    copy_logs_to_model_dir = False
    
    try:
        log_dir = os.environ['SM_CHANNEL_LOG']
        copy_logs_to_model_dir = True
    except KeyError:
        log_dir = model_dir
        
    train_data_dir = os.environ['SM_CHANNEL_TRAIN']
    hyperparamters = json.loads(os.environ['SM_HPS'])
    
    try:
        data_name = hyperparamters['data_name']
    except KeyError:
        data_name = 'MOT20'
    
    try:
        load_model = hyperparamters['load_model']
    except KeyError:
        load_model = ''
    
    try:
        num_workers = hyperparamters['num_workers']
    except KeyError:
        num_workers = 0
        
    try:
        num_workers_val = hyperparamters['num_workers_val']
    except KeyError:
        num_workers_val = 2
    
    try:
        # model architecture.
        # Currently tested resdcn_34 | resdcn_50 | resfpndcn_34 dla_34 | hrnet_18
        arch = hyperparamters['arch']
    except KeyError:
        arch = 'dla_34'

    try:
        # input height. -1 for default from dataset.
        input_h = hyperparamters['input_h']
    except KeyError:
        input_h = 608

    try:
        # input width. -1 for default from dataset.
        input_w = hyperparamters['input_w']
    except KeyError:
        input_w = 1088
        
    try:
        # learning rate for batch size 12.
        lr = hyperparamters['lr']
    except KeyError:
        lr = 1e-4

    try:
        # drop learning rate by 10.
        lr_step = hyperparamters['lr_step']
    except KeyError:
        lr_step = 15

    try:
        # total training epochs.
        num_epochs = hyperparamters['num_epochs']
    except KeyError:
        num_epochs = 30

    try:
        # batch size, For MOT, less than 12 should be set on NVIDIA V100 GPU
        batch_size = hyperparamters['batch_size']
    except KeyError:
        batch_size = 12
    
    try:
        # batch size, For MOT, less than 12 should be set on NVIDIA V100 GPU
        batch_size_val = hyperparamters['batch_size_val']
    except KeyError:
        batch_size_val = 12
        
    try:
        max_label = hyperparamters['max_label']
    except KeyError:
        # should be greater than 1500 on MOT20
        max_label = 1500
        
    try:
        # default: #samples / batch_size.
        num_iters = hyperparamters['num_iters']
    except KeyError:
        num_iters = -1
    
    try:
        val_intervals = hyperparamters['val_intervals']
    except KeyError:
        val_intervals = 5
    
    try:
        # max number of output objects.
        paramK = hyperparamters['K']
    except KeyError:
        paramK = 500
    
    # #############################################
    # Loss
    # #############################################   
    try:
        # use mse loss or focal loss to train keypoint heatmaps.
        mse_loss = hyperparamters['mse_loss']
    except KeyError:
        mse_loss = True
    
    try:
        # regression loss: sl1 | l1 | l2
        reg_loss = hyperparamters['reg_loss']
    except KeyError:
        reg_loss = 'l1'
    
    try:
        # loss weight for keypoint heatmaps.
        hm_weight = hyperparamters['hm_weight']
    except KeyError:
        hm_weight = 1
    
    try:
        # loss weight for keypoint local offsets.
        off_weight = hyperparamters['off_weight']
    except KeyError:
        off_weight = 1
    
    try:
        # loss weight for bounding box size.
        wh_weight = hyperparamters['wh_weight']
    except KeyError:
        wh_weight = 0.1
    
    try:
        # reid loss: ce | triplet
        id_loss = hyperparamters['id_loss']
    except KeyError:
        id_loss = 'ce'
    
    try:
        # loss weight for id
        id_weight = hyperparamters['id_weight']
    except KeyError:
        id_weight = 1
    
    try:
        # feature dim for reid
        reid_dim = hyperparamters['reid_dim']
    except KeyError:
        reid_dim = 128
        
    try:
        # regress left, top, right, bottom of bbox
        ltrb = hyperparamters['ltrb']
    except KeyError:
        ltrb = True
        
    try:
        # L1(\hat(y) / y, 1) or L1(\hat(y), y)
        norm_wh = hyperparamters['norm_wh']
    except KeyError:
        norm_wh = True
    
    try:
        # apply weighted regression near center or just apply regression on center point.
        dense_wh = hyperparamters['dense_wh']
    except KeyError:
        dense_wh = True
    
    try:
        # category specific bounding box size.
        cat_spec_wh = hyperparamters['cat_spec_wh']
    except KeyError:
        cat_spec_wh = True
    
    try:
        # not regress local offset.
        not_reg_offset = hyperparamters['not_reg_offset']
    except KeyError:
        not_reg_offset = True
    
    gpus_per_host = int(os.environ['SM_NUM_GPUS'])
    gpus = ','.join([str(i) for i in range(gpus_per_host)])
    
    # optimize hyperparameter in multi-GPU model
    num_workers = gpus_per_host
    lr *= gpus_per_host
    batch_size *= gpus_per_host
    
    train_cmd = f"""
cd /fairmot/src && python train.py mot \
--batch_size {batch_size} \
--num_epochs {num_epochs} \
--lr_step '{lr_step}' \
--data_cfg {train_data_dir}/{data_name}/data.json \
--num_workers {num_workers} \
--reg_loss {reg_loss} \
--hm_weight {hm_weight} \
--off_weight {off_weight} \
--wh_weight {wh_weight} \
--id_loss {id_loss} \
--id_weight {id_weight} \
--reid_dim {reid_dim} \
--arch {arch} \
--input_h {input_h} \
--input_w {input_w} \
--lr {lr} \
--val_intervals {val_intervals} \
--gpus {gpus} \
--save_dir {model_dir} \
--batch_size_val {batch_size_val} \
--num_workers_val {num_workers_val} \
"""
    
    if len(load_model) > 0:
        train_cmd += f" --load_model {train_data_dir}/pretrained-models/{load_model}"
        
    print("--------Begin Model Training Command----------")
    print(train_cmd)
    print("--------End Model Training Comamnd------------")
    exitcode = 0
    try:
        process = subprocess.Popen(
            train_cmd,
            encoding='utf-8', 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)

        while True:
            if process.poll() != None:
                break

            output = process.stdout.readline()
            if output:
                print(output.strip())
        
        exitcode = process.poll() 
        print(f"exit code:{exitcode}")
        exitcode = 0 
    except Exception as e:
        print("train exception occured", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)
    finally:
        if copy_logs_to_model_dir:
            copy_files(log_dir, model_dir)
    
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)

if __name__ == "__main__":
    train()