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
        batch_size=100
    )
    params_dict = {'infeat_nodes': 6,
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
                    'resample_freq_timesteps': -1,
                    'nepochs': sigopt.params.nepochs,
                    'batch_size': sigopt.params.batch_size}

    start = time.time()
    gnn_model, loss, train_dataloader, coefs_dict, out_fdr = tr.launch_training(sys.argv[1], 'adam',
                                                             params_dict, train_params, True,
                                                             log_checkpoint)
    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))

    err_p, err_q, global_err = tr.evaluate_error(gnn_model, sys.argv[1],
                                                 train_dataloader,
                                                 coefs_dict,
                                                 do_plot = True,
                                                 out_folder = out_fdr)

    print('Error pressure ' + str(err_p))
    print('Error flowrate ' + str(err_q))
    print('Global error ' + str(global_err))
    sigopt.log_metric(name="loss", value=loss)
    sigopt.log_metric(name="error pressure", value=err_p)
    sigopt.log_metric(name="error flowrate", value=err_q)
    sigopt.log_metric(name="global error", value=global_err)
    sigopt.log_metric(name="training time", value=elapsed_time)
