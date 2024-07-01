from model.EquiVSet import EquiVSet
from utils.config import MOONS_CONFIG, GAUSSIAN_CONFIG, BINDINGDB_CONFIG, AMAZON_CONFIG, CELEBA_CONFIG, PDBBIND_CONFIG
import psutil
import pandas as pd
import time
import os


if __name__ == "__main__":
    argparser = EquiVSet.get_model_specific_argparser()
    hparams = argparser.parse_args()
    # if hparams.mode == 'diffMF':
    #     hparams.RNN_steps = 5
    data_name = hparams.data_name
    # update the arguments dictionary custom to chosen input dataset
    if data_name == 'moons':
        hparams.__dict__.update(MOONS_CONFIG)
    elif data_name == 'gaussian':
        hparams.__dict__.update(GAUSSIAN_CONFIG)
    elif data_name == 'amazon':
        hparams.__dict__.update(AMAZON_CONFIG)
    elif data_name == 'celeba':
        hparams.__dict__.update(CELEBA_CONFIG)
    elif data_name == 'pdbbind':
        hparams.__dict__.update(PDBBIND_CONFIG)
    elif data_name == 'bindingdb':
        hparams.__dict__.update(BINDINGDB_CONFIG)
    else:
        raise ValueError('invalid dataset...')

    start_time = time.time()
    # Track memory usage before training
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    # initialize the model with the updated arguments
    model = EquiVSet(hparams)

    if hparams.train:
        # training mode
        metrics_dict = model.run_training_sessions()
        # Track memory usage after training
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before

        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert dictionary to a list of tuples and insert
        items = list(metrics_dict.items())
        items.insert(3, ("lr", hparams.lr))
        metrics_dict = dict(items)

        metrics_dict["batch_size"] = hparams.batch_size
        metrics_dict["time"] = elapsed_time
        metrics_dict["memory_used_MB"] = memory_used / (1024 ** 2)
        print(metrics_dict)

        # Convert dictionary to DataFrame
        df_metrics = pd.DataFrame([metrics_dict])

        # Filepath for the output CSV
        output_filepath = f"../history/baseline_metrics_{hparams.data_name}.csv"

        # Append metrics to CSV
        if os.path.exists(output_filepath):
            # If the file exists, load the existing DataFrame and concatenate
            output = pd.read_csv(output_filepath)
            output = pd.concat([output, df_metrics], ignore_index=True)
        else:
            # If the file does not exist, create a new DataFrame
            output = df_metrics

        # Save the updated DataFrame to CSV
        output.to_csv(output_filepath, index=False)

        # Print the head of the DataFrame
        print(output.head())

    else:
        # testing mode
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))