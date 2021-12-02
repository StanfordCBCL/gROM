import sigopt
import sys

sys.path.append("../network")

import training as tr
import numpy as np
import time

def log_checkpoint(loss):
    sigopt.log_checkpoint({'loss': loss})
    sigopt.log_metric(name="loss", value=loss)

if __name__ == "__main__":
    sigopt.params.setdefaults(
        latent_size_gnn=32,
        latent_size_mlp=64,
        learning_rate=0.001,
        weight_decay=0.999,
        momentum=0.0,
        process_iterations=1,
        hl_mlp=2,
        normalize=True,
        nepochs=30,
        batch_size=100,
        rate_noise=1e-4,
        random_walks=0,
        resample_freq_timesteps=1,
        normalization='standard'
    )
    network_params = {'infeat_nodes': 6,
                    'infeat_edges': 5,
                     'latent_size_gnn': sigopt.params.latent_size_gnn,
                    'latent_size_mlp': sigopt.params.latent_size_mlp,
                    'out_size': 2,
                    'process_iterations': sigopt.params.process_iterations,
                    'hl_mlp': sigopt.params.hl_mlp,
                    'normalize': sigopt.params.normalize}
    train_params = {'learning_rate': sigopt.params.learning_rate,
                    'weight_decay': sigopt.params.weight_decay,
                    'momentum': sigopt.params.momentum,
                    'resample_freq_timesteps': sigopt.params.resample_freq_timesteps,
                    'nepochs': sigopt.params.nepochs,
                    'batch_size': sigopt.params.batch_size}
    dataset_params = {'rate_noise': sigopt.params.rate_noise,
                      'random_walks': sigopt.params.random_walks,
                      'normalization': sigopt.params.normalization}

    dir(sigopt)

    start = time.time()
    gnn_model, loss, train_dataloader, coefs_dict, out_fdr = tr.launch_training(sys.argv[1], 'adam',
                                                             network_params, train_params, True,
                                                             log_checkpoint,
                                                             dataset_params)
    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))

    err_p, err_q, global_err = tr.evaluate_error(gnn_model, sys.argv[1],
                                                 train_dataloader,
                                                 coefs_dict,
                                                 do_plot = True,
                                                 out_folder = out_fdr)

    if err_p != err_p or err_p > 1e10:
        err_p = 1e10

    if err_q != err_q or err_q > 1e10:
        err_q = 1e10

    if global_err != global_err or global_err > 1e10:
        global_err = 1e10

    print('Error pressure ' + str(err_p))
    print('Error flowrate ' + str(err_q))
    print('Global error ' + str(global_err))
    sigopt.log_metric(name="loss", value=loss)
    sigopt.log_metric(name="error_pressure", value=err_p)
    sigopt.log_metric(name="error_flowrate", value=err_q)
    sigopt.log_metric(name="global_error", value=global_err)
    sigopt.log_metric(name="training_time", value=elapsed_time)
