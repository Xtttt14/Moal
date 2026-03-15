import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    device_override = args.device_override
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    # Command-line --device overrides the config file value
    if device_override is not None:
        args["device"] = device_override
    args.pop("device_override", None)

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--device', dest='device_override', type=int, nargs='+', default=None,
                        help='GPU device id(s) to use. E.g. --device 0 1 2 for multi-GPU. '
                             'Use --device -1 for CPU. Overrides the "device" field in the config file.')
    return parser

if __name__ == '__main__':
    main()
