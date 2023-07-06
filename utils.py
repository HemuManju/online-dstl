import sys
import os
from contextlib import contextmanager

import random
import numpy as np

import torch

from tensorboard import program


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format('Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block', '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def get_gpu_list():
    if torch.cuda.device_count() == 0:
        return None
    else:
        return list(range(torch.cuda.device_count()))


def launch_tensorboard(cfg):
    tb = program.TensorBoard()
    tb.configure(
        argv=[
            None,
            '--logdir',
            cfg.log_dir,
            '--reload_multifile',
            'true',
            '--reload_interval',
            '15',
        ]
    )
    tb.launch()
    return None


def seed_everything(seed=None):
    if seed is None:
        seed = 1337

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_weights(trained_date=None, model=None, name='imitation-v', read_path=None):
    if read_path is None:
        read_path = 'trained_models/' + trained_date + '/' + name + str(model) + '.ckpt'
    checkpoint = torch.load(read_path, map_location=torch.device('cpu'))
    try:
        weights = {
            k.replace('net.', '', 1): checkpoint['state_dict'][k]
            for k in checkpoint['state_dict'].keys()
            if k.startswith('net.')
        }
    except KeyError:
        return checkpoint
    return weights


def load_learnt_model(trained_date=None, model=None, read_path=None):
    if read_path is None:
        read_path = (
            'trained_models/'
            + trained_date
            + '/'
            + 'autoencoder-v'
            + str(model)
            + '.pth'
        )

    model = torch.load(read_path, map_location=torch.device('cpu'))
    return model
