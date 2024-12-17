import argparse
import os
import yaml

global_print_hparams = True
hparams = {
#Favored configuration that I have selected upon
    'hidden_size': 128,
    'unet_dim_mults': '1|2|4|8',
    'rrdb_num_feat': 64,
    'rrdb_num_block': 8,
    'fix_rrdb': True,  # Set to true if you have a pretrained block you don't want to change
    'sr_scale': 5,  # Scale factor for super-resolution
    'use_wn': True,  # Use weight normalization
    'use_attn': True,  # Use attention in UNet
    'weight_init': True,  # Use weight initialization
    'res': True,  # Use residual connections
    'up_input': True,  # Use upsampled input in the model
    'use_rrdb': True,
    'rrdb_ckpt': 'model_rrdb_spatial_checkpoint.pth',#Blank because we have no pre-trained checkpoint

    # Training parameters
    'batch_size': 4,  # Batch size for training
    'eval_batch_size': 4,  # Batch size for validation/testing
    'lr': 5e-3,  # Learning rate
    'decay_steps': 10000,  # Steps for learning rate decay
    'decay_gamma': 0.5,  # Decay factor for learning rate
    'max_updates': 300000,  # Maximum training updates
    'val_check_interval': 1000,  # Validation interval
    'num_workers': 16,  # Number of data loader workers

    # Diffusion configuration
    'timesteps': 1000,
    'beta_schedule': 'cosine',  #new noise scheduler added - we can have it quadratic too
    'beta_start': 0.0001,  # Start value for beta
    'beta_end': 0.02,  # End value for beta
    'beta_s': 0.008,  # Beta start value for cosine schedule

    # Loss and evaluation
    'loss_type': 'mse',  #custom loss type being used - don't need this
    'metrics': ['psnr', 'ssim'], #lpips is an option but it is removed #remove metrics to save time
    'lpips_net': [],  #don't need this right now alex and vgg are the options

    # Checkpoints and directories
    'work_dir': '/content/SRDiff',  # Working directory
    'num_ckpt_keep': 5,  # Number of checkpoints to retain

    # Other configuration
    'groups': 1,  #using a group value of one allows us to make sure that all of the gene channels are observed together - more computationally demanding
    'num_sanity_val_steps': 2,
    #TRAIN DATA - CHANGE IF USING DIFFERENT DATA - truncated data atm 28-1 (ignore last incomplete)
    'num_train_files': 27,
    'bayes_space': True
                        }


class Args:
    """
    A utility class for managing argument values as object attributes.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    """
    Recursively override old configuration values with new ones.
    """
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', hparams_str='', print_hparams=True, global_hparams=True):
    """
    Main function for setting hyperparameters (hparams).
    Supports:
    - Loading YAML-based configurations.
    - Inheriting from base configurations.
    - Overriding with command-line arguments.
    """
    if config == '' and exp_name == '':
        # Parse command-line arguments if no config or exp_name is directly provided
        parser = argparse.ArgumentParser(description='Set hyperparameters for training.')
        parser.add_argument('--config', type=str, default='', help='Path to the main config file.')
        parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment.')
        parser.add_argument('-hp', '--hparams', type=str, default='', help='Overrides for hyperparameters.')
        parser.add_argument('--infer', action='store_true', help='Run inference.')
        parser.add_argument('--validate', action='store_true', help='Run validation only.')
        parser.add_argument('--reset', action='store_true', help='Reset hyperparameters to the config defaults.')
        parser.add_argument('--debug', action='store_true', help='Enable debugging mode.')
        args, unknown = parser.parse_known_args()
        print("| Unknown hparams: ", unknown)
    else:
        # Handle directly provided configurations
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, validate=False, reset=False, debug=False)

    global hparams
    assert args.config != '' or args.exp_name != '', "Either config or exp_name must be provided."
    if args.config != '':
        assert os.path.exists(args.config), f"Config file {args.config} does not exist."

    config_chains = []
    loaded_config = set()

    def load_config(config_fn):
        """
        Recursively load configuration files and handle inheritance via 'base_config'.
        """
        if not os.path.exists(config_fn):
            return {}
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        loaded_config.add(config_fn)
        if 'base_config' in hparams_:
            ret_hparams = {}
            if not isinstance(hparams_['base_config'], list):
                hparams_['base_config'] = [hparams_['base_config']]
            for c in hparams_['base_config']:
                # Handle relative paths for base configs
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                if c not in loaded_config:
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams

    saved_hparams = {}
    args_work_dir = ''
    if args.exp_name != '':
        # Load saved hyperparameters from a previous checkpoint
        args_work_dir = f'checkpoints/{args.exp_name}'
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                saved_hparams_ = yaml.safe_load(f)
                if saved_hparams_ is not None:
                    saved_hparams.update(saved_hparams_)
    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(args.config))
    if not args.reset:
        hparams_.update(saved_hparams)  # Incorporate saved hparams if reset is not specified
    hparams_['work_dir'] = args_work_dir

    # Override hparams from command-line arguments
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            for k_ in k.split(".")[:-1]:
                config_node = config_node[k_]
            k = k.split(".")[-1]
            if k not in config_node:
                config_node[k] = v
            elif v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                config_node[k] = eval(v)  # Safely evaluate lists and boolean values
            else:
                config_node[k] = type(config_node[k])(v)  # Cast to original type
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)

    # Add global flags and attributes
    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name

    # Print hyperparameters if required
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_


