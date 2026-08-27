import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dataset import CounterfactualDataset_5cf_NAA_dataset
from methods import *


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack', type=str, default='COPIA_Inception_v3_five_cf', help='the name of specific attack method')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for reference (default: 16)')
    parser.add_argument('--source_model', type=str, default='inception_v3', help='')
    parser.add_argument('--lam', type=float, default=0.01, help='adaptive_factor')
    args = parser.parse_args()
    args.opt_path = os.path.join('./adv_images_NAA_dataset/',
                                 'model_{}-method_{}'.format(args.source_model, args.attack))

    if not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataset = CounterfactualDataset_5cf_NAA_dataset(args.source_model, os.path.join('', 'NAA_images'),csv_file='NAA_dataset_sub_1.csv')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(args.attack, args.source_model)

    # Attack
    method = globals()[args.attack]
    if args.attack in ['LowRankAttack']:
        otas_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        calibration_data, _, _, _ = next(iter(otas_loader))
        attack_method = getattr(method, args.attack)(args.source_model, args, calibration_data=calibration_data)
    else:
        attack_method = getattr(method, args.attack)(args.source_model, args)
    # Main
    all_loss_info = {}
    for batch_idx, batch_data in tqdm(enumerate(data_loader)):
        batch_x = batch_data[0]
        batch_y = batch_data[1]
        batch_name = batch_data[3]
        batch_cf = batch_data[4]
        adv_inps, loss_info = attack_method(batch_x, batch_y, batch_cf)
        attack_method._save_images(adv_inps, batch_name, args.opt_path)

