import os
import glob
import argparse
import json

def gen_conf(data_dir, data_name, half_val):
    if half_val == 1:
        val_path = f"/opt/ml/input/data/train/{data_name}/data.val"
    else:
        val_path = f"/opt/ml/input/data/train/{data_name}/data.train"
    
    conf_dict = {"root":"/opt/ml/input/data/train",
                    "train":
                    {
                        data_name.lower(): f"/opt/ml/input/data/train/{data_name}/data.train"
                    },
                    "test_emb":
                    {
                        data_name.lower(): f"/opt/ml/input/data/train/{data_name}/data.train"
                    },
                    "test":
                    {
                        data_name.lower(): val_path
                    }
                }
    
    conf_path = os.path.join(data_dir, data_name, 'data.json')
    with open(conf_path, 'w') as fp:
        json.dump(conf_dict, fp)

def main(data_dir, data_name, save_dir='/opt/ml/input/data/train', half_val=1):
    real_path = os.path.join(data_dir, data_name, 'images/train')
    seq_names = [s for s in sorted(os.listdir(real_path))]
    
    def write_data_file(phase='train', seq_names=[], half_val=half_val):
        with open(os.path.join(data_dir, '{}/data.{}'.format(data_name, phase)), 'w') as f:
            for seq_name in seq_names:
                print(seq_name)
                seq_path = os.path.join(real_path, seq_name)
                seq_path = os.path.join(seq_path, 'img1')
                images = sorted(glob.glob(seq_path + '/*.jpg'))
                len_all = len(images)
                if half_val == 1:
                    len_half = int(len_all / 2)
                    if phase == 'train':
                        s = 0
                        e = len_half+1
                    else:
                        s = len_half
                        e = len_all
                else:
                    s = 0
                    e = len_all
                for i in range(s, e):
                    image = images[i].replace(data_dir, save_dir)
                    print(image, file=f)
    
    write_data_file('train', seq_names=seq_names, half_val=half_val)
    if half_val == 1:
        write_data_file('val', seq_names=seq_names)
    gen_conf(data_dir, data_name, half_val=half_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating data path')
    parser.add_argument('--dataset-dir', required=True, type=str)
    parser.add_argument('--dataset-name', required=True, type=str)
    parser.add_argument('--half-val', required=True, type=int)
    args = parser.parse_args()
    
    main(data_dir=args.dataset_dir, data_name=args.dataset_name, half_val=args.half_val)
