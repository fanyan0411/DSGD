import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    device = args.device
    memory_size_supervise = args.memory_size_supervise
    memory_size_unsupervise = args.memory_size_unsupervise
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    prefix = args["prefix"]
    args.update(param)  # Add parameters from json
    args["prefix"] = prefix
    args["device"] = device
    args["memory_size_supervise"] = memory_size_supervise
    args["memory_size_unsupervise"] = memory_size_unsupervise
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/icarl_10.json',
                        help='Json file of settings.')
    parser.add_argument('--prefix', type=str, default='test', help='prefix of log file') #uloss0.04_sbshalf_2_200epoch_nounlabelkd uloss0.5_reweight_sbshalf_2_200epoch_nounlabelkd
    parser.add_argument('--label_num', type=str, default='300',
                        help='number of labeled data, need to be equal to prefix and bigger than memory_size_supervise') 
    parser.add_argument('--full_supervise', action='store_true')

    ## Knowledge distillation
    parser.add_argument('--kd_weight', type=float, default=1, help='the weight of kd distill loss')
    parser.add_argument('--kd_onlylabel', action='store_true') #, default=True
    ## Memory setting
    parser.add_argument('--memory_size_supervise', type=int, default=500, help='the number of supervised samples in memory buffer, 500|1000|1500')
    parser.add_argument('--memory_size_unsupervise', type=int, default=1500, help='the number of unsupervised samples in memory buffer, 1500|1000|500')
    ## DSGD
    parser.add_argument('--usp_weight',type=float, default=1, help='weight of semi-supervised loss')
    parser.add_argument('--insert_pse_progressive', action='store_true', help='adapt pseudo label of last model in semi-supervised loss', default=True)
    parser.add_argument('--insert_pse', type=str, help='threshold or logitstic', default='logitstic')
    parser.add_argument('--pse_weight', type=float, default=1.0, help='the weight of pseudo label in uloss')
    parser.add_argument('--oldpse_thre', type=float, default=0.8, help='the threshold of old pseudo label')
    parser.add_argument('--rw_alpha', type=float, default=0, help='the weight of graph embedding') #TODO
    parser.add_argument('--match_weight', type=float, default=0.2, help='the weight of sub-graph distill, 0.2 for cifar10, 1 for cifar100, 0.04 for imagenet')
    parser.add_argument('--rw_T', type=int, default=2, help='the order of structure embedding') #TODO
    parser.add_argument('--gamma_ml', type=float, default=1)
    ## Commom setting
    parser.add_argument('--resume', default=True, action='store_true') #, action='store_true' , type=bool, default=True
    parser.add_argument('--save_all_resume', action='store_true', default=False)
    parser.add_argument('--device', type=str, default=['0','1'])

    return parser


if __name__ == '__main__':
    main()
