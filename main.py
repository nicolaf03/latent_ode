import argparse
from random import SystemRandom
import os
import time
import sys

import torch
# import torch.nn as nn
# from torch.nn.functional import relu
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import pandas as pd
from pathlib import Path
import torchcde
import wandb

import lib.utils as utils
from lib.plotting import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.pytorchtools import EarlyStopping


# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
# parser.add_argument('-n', type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('--t-size', type=int, default=7)
parser.add_argument('--batch-size', type=int, default=16)

parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

# parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

# parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
# 	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

# parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

# parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
# parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
# parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

# parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

# parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=10, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

# parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
# parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
# parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

parser.add_argument('--wandb', type=str, default='offline')

args = parser.parse_args()

os.environ['WANDB_MODE'] = args.wandb
wandb.init(project='wind_latentODE')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

curr_dir = Path(__file__).parent


if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")
    
    start = time.time()
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    
    utils.makedirs("results/")
    
    #############################
    # import data
    def _load_data(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True, drop=False)
        return df

    def _load_historic_data(folder, zone):
        filename = f'res_{zone.upper()}.csv'
        data_file_path = folder / filename
        df = _load_data(data_file_path)
        print(f'Historical data are from: {df["date"].min()} - to: {df["date"].max()}')
        return df
    
    def _create_dataloader(data, data_type):
        t_size = args.t_size
        batch_size = args.batch_size

        ts = torch.linspace(0, t_size - 1, t_size)
        
        value_array = np.array(data.iloc[:,1], dtype='float32')
        values = []
        for i in range(len(data)-t_size):
            sub_array = value_array[i:i+t_size]
            x = torch.from_numpy(np.expand_dims(sub_array,0))   # [1,7]
            #x = torch.from_numpy(sub_array)                     # [7]
            values.append(x)
        ys = torch.stack(values).transpose(1,2)                 # [2185, 7, 1]
        
        dataset_size = ys.size()[0]
        
        ###################
        # Normalise data
        # y0_flat = ys[0].view(-1)
        # y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat)) #? unnecessary
        # ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()
        
        # todo: do it without loop
        # make all paths start from 0
        for i in range(dataset_size):
            ys[i] = ys[i] - ys[i][0]
            
        ###################
        # Time must be included as a channel for the discriminator.
        # ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1), ys], dim=2)
        
        ###################
        # Package up
        # data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
        ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
        #dataset = torch.utils.data.TensorDataset(ys_coeffs)
        dataloader = torch.utils.data.DataLoader(
            ys_coeffs, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=(lambda batch: basic_collate_fn(batch=batch, time_steps=ts, data_type=data_type))
        )
        
        return ts, dataloader
    
    def basic_collate_fn(batch, time_steps, args=args, device=device, data_type='train'):
        batch = torch.stack(batch)
        data_dict = {
            'data': batch,
            'time_steps': time_steps
        }
        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type)
        return data_dict
    
    
    train_window = ('2015-01','2020-12')
    valid_window = ('2021-01','2021-09')
    test_window = ('2021-10','2021-12')
    
    start_train, end_train = train_window
    start_valid, end_valid = valid_window
    start_test, end_test = test_window
    
    historic_folder = curr_dir / 'data'
    data = _load_historic_data(historic_folder, 'SUD')
    
    df_train = data.loc[start_train:end_train]
    df_valid = data.loc[start_valid:end_valid]
    df_test = data.loc[start_test:end_test]

    # remove valid_set from train_set
    index_diff = df_train.index.difference(df_valid.index)
    df_train = df_train.loc[index_diff]
    
    ts, train_dataloader = _create_dataloader(df_train, 'train')
    _, valid_dataloader = _create_dataloader(df_valid, 'valid')
    _, test_dataloader = _create_dataloader(df_test, 'test')
    
    input_dim = 1
    
    data_obj = {
		"train_dataloader": utils.inf_generator(train_dataloader), 
		"valid_dataloader": utils.inf_generator(valid_dataloader),
		"input_dim": input_dim,
		"n_train_batches": len(train_dataloader),
		"n_valid_batches": len(valid_dataloader)
	}
    
    classif_per_tp = False
    n_labels = 1
    
    
    # TODO: move to device
    
    ##################################################################
	# Create the model
    obsrv_std = 0.01
    obsrv_std = torch.tensor([obsrv_std]).to(device)
    
    z0_prior = Normal(
        torch.Tensor([0.0]).to(device),
        torch.Tensor([1.0]).to(device)
    )
    
    model = create_LatentODE_model(
        args, 
        input_dim, 
        z0_prior, 
        obsrv_std, 
        device, 
		classif_per_tp = classif_per_tp,
		n_labels = n_labels
    )
    
    
    ##################################################################
    # visualization
    if args.viz:
        viz = Visualizations(device)
    
    
    ##################################################################
    # load checkpoints and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
    
    
    ##################################################################
    # training
    log_path = 'logs/' + file_name + '_' + str(experimentID) + '.log'
    if not os.path.exists('logs/'):
        utils.makedirs('logs/')
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    num_batches = data_obj['n_train_batches']
    
    
    # initialize the early_stopping object
    path = ckpt_path #f'./train/checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=path,
        verbose=True
    )
    
    for itr in range(1, num_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr/10)
        
        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1 - 0.99**(itr // num_batches - wait_until_kl_inc))
        
        batch_dict = utils.get_next_batch(data_obj['train_dataloader'])
        train_res = model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
        
        train_res['loss'].backward()
        optimizer.step()
        
        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * num_batches) == 0:
            with torch.no_grad():
                valid_res = compute_loss_all_batches(
                    model,
                    data_obj['valid_dataloader'],
                    args,
                    n_batches=data_obj['n_valid_batches'],
                    experimentID=experimentID,
                    device=device,
                    n_traj_samples=3,
                    kl_coef=kl_coef
                )
                
                message = 'Epoch {:04d} [Valid seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr // num_batches,
                    valid_res['loss'].detach(),
                    valid_res['likelihood'].detach(),
                    valid_res['kl_first_p'],
                    valid_res['std_first_p']
                )
                
                logger.info('Experiment ' + str(experimentID))
                logger.info(message)
                logger.info(f'KL coef: {kl_coef}')
                logger.info(f'Train loss (one batch): {train_res["loss"].detach()}')
                logger.info(f'Train CE loss (one batch): {train_res["ce_loss"].detach()}')
                
                if 'auc' in valid_res:
                    logger.info(f'AUC (TEST): {valid_res["auc"]}')
                if 'mse' in valid_res:
                    logger.info(f'MSE (TEST): {valid_res["mse"]}')
                if 'accuracy' in train_res:
                    logger.info(f'Accuracy (TRAIN): {train_res["accuracy"]}')
                if 'accuracy' in valid_res:
                    logger.info(f'Accuracy (TEST): {valid_res["accuracy"]}')
                if 'pois_likelihood' in valid_res:
                    logger.info(f'Poisson likelihood (TEST): {valid_res["pois_likelihood"]}')
                if 'ce_loss' in valid_res:
                    logger.info(f'CE loss (TEST): {valid_res["ce_loss"]}\n')
            
            # save checkpoint
            torch.save(
                {
                    'args': args,
                    'state_dict': model.state_dict()
                },
                ckpt_path
            )
            
            wandb.log(
                {
                    'train loss': train_res["loss"].detach(),
                    'valid MSE': valid_res["mse"],
                    'kl_coef': kl_coef
                }
            )
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_res["mse"], model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # plot
            # if args.viz:
            #     with torch.no_grad():
            #         test_dict = utils.get_next_batch(data_obj['test_dataloader'])
                    
            #         print('plotting...')
                    
            #         # todo...
    
    # save model
    torch.save(
        {
            'args': args,
            'state_dict': model.state_dict()
        },
        ckpt_path
    )
        
    
    print('Done')