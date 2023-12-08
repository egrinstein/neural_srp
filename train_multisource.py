#
# A wrapper script that trains the DOAnet. The training stops when the early stopping metric - SELD error stops improving.
#

import matplotlib.pyplot as plot
import numpy as np
import os
import sys
import time
import torch

from datasets.tau_nigens_dataset import TauNigensDataLoader
from loss import MultiSourceLoss
from metrics import MultiSourceMetrics
from trainers.tau_nigens import TauNigensTrainer
from utils import get_device, get_params

plot.switch_backend("agg")


def create_datasets(params):
    test_splits = [1]
    val_splits = [2]
    train_splits = [[3, 4, 5, 6]]

    for split_cnt, split in enumerate(test_splits):
        print("Split: {}".format(split))

        # Load train and validation data
        print("Loading training dataset:")

        data_gen_train = TauNigensDataLoader(
            params=params, split=train_splits[split_cnt]
        )

        print("Loading validation dataset:")
        data_gen_val = TauNigensDataLoader(
            params=params, split=val_splits[split_cnt], shuffle=False
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()

        print("Loading unseen test dataset:")
        data_gen_test = TauNigensDataLoader(
            params=params, split=test_splits[split_cnt], shuffle=False
        )

        print("FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n".format(data_in, data_out))

        return data_gen_train, data_gen_val, data_gen_test, data_in, data_out


def plot_functions(fig_name, _tr_loss, _val_loss, _tr_hung_loss, _val_hung_loss):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(211)
    plot.plot(range(nb_epoch), _tr_loss, label="train loss")
    plot.plot(range(nb_epoch), _val_loss, label="test loss")
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(nb_epoch), _tr_hung_loss, label="train hung loss")
    plot.plot(range(nb_epoch), _val_hung_loss, label="test hung loss")
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def evaluate_on_test_set(checkpoint_path, params, data_gen_test,
                         trainer: TauNigensTrainer):
    # ---------------------------------------------------------------------
    # Evaluate on unseen test data
    # ---------------------------------------------------------------------
    print("Loading best model weights")
    trainer.load_checkpoint(
        checkpoint_path
    )

    test_metric = MultiSourceMetrics(params)
    (
        test_loss,
        test_dMOTP_loss,
        test_dMOTA_loss,
        test_act_loss,
    ) = trainer.test_epoch(data_gen_test, test_metric)

    (
        test_loc_error,
        test_mota,
        test_ids,
        test_recall_doa,
        test_precision_doa,
        test_fscore_doa,
    ) = test_metric.get_results()

    print(
        "test_loss: {:0.2f} {}, LE/MOTA/IDS/LR/LP/LF: {:0.3f}/{}".format(
            test_loss,
            "({:0.2f},{:0.2f},{:0.2f})".format(
                test_dMOTP_loss, test_dMOTA_loss, test_act_loss
            ),
            test_loc_error,
            "{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}".format(
                test_mota,
                test_ids,
                test_recall_doa,
                test_precision_doa,
                test_fscore_doa,
            ),
        )
    )


def main():
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """

    params = get_params()

    # Unique name for the run
    os.makedirs("checkpoints/", exist_ok=True)
    unique_name = "tau_" + params["model"]
    unique_name = os.path.join("checkpoints/", unique_name)
    model_name = "{}_model.h5".format(unique_name)

    print("unique_name: {}\n".format(unique_name))


    device = get_device()
    # torch.autograd.set_detect_anomaly(True)

    loss = MultiSourceLoss(params, device)

    # create data generators for train, val and test sets
    data_gen_train, data_gen_val, data_gen_test, data_in, data_out = create_datasets(
        params
    )

    # create trainer
    trainer = TauNigensTrainer(params, loss, data_in, data_out)

    # start training
    best_val_epoch = -1
    best_doa = 180
    best_mota = 0
    best_ids = 1000
    best_recall = 0
    best_precision = 0
    best_fscore = 0

    nb_epoch = 2 if params["quick_test"] else params["training"]["nb_epochs"]
    tr_loss_list = np.zeros(nb_epoch)
    val_loss_list = np.zeros(nb_epoch)
    hung_tr_loss_list = np.zeros(nb_epoch)
    hung_val_loss_list = np.zeros(nb_epoch)

    for epoch_cnt in range(nb_epoch):
        # ---------------------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------------------
        start_time = time.time()
        (
            train_loss,
            train_dMOTP_loss,
            train_dMOTA_loss,
            train_act_loss,
        ) = trainer.train_epoch(data_gen_train)
        train_time = time.time() - start_time
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        start_time = time.time()
        val_metric = MultiSourceMetrics(params)
        (
            val_loss,
            val_dMOTP_loss,
            val_dMOTA_loss,
            val_act_loss,
        ) = trainer.test_epoch(data_gen_val, val_metric)

        (
            loc_error,
            val_mota,
            val_ids,
            val_recall_doa,
            val_precision_doa,
            val_fscore_doa,
        ) = val_metric.get_results()
        val_time = time.time() - start_time

        # Save model if loss is good
        if loc_error > 0 and loc_error <= best_doa:
            (
                best_val_epoch,
                best_doa,
                best_mota,
                best_ids,
                best_recall,
                best_precision,
                best_fscore,
            ) = (
                epoch_cnt,
                loc_error,
                val_mota,
                val_ids,
                val_recall_doa,
                val_precision_doa,
                val_fscore_doa,
            )
            torch.save(
                trainer.model.state_dict(),
                model_name.replace(".h5", f"_epoch{best_val_epoch}.h5"),
            )

        torch.save(trainer.model.state_dict(), model_name.replace(".h5", "_last.h5"))

        # Print stats and plot scores
        print(
            "epoch: {}, time: {:0.2f}/{:0.2f}, "
            "train_loss: {:0.2f} {}, val_loss: {:0.2f} {}, "
            "LE/MOTA/IDS/LR/LP/LF: {:0.3f}/{}, "
            "best_val_epoch: {} {}".format(
                epoch_cnt,
                train_time,
                val_time,
                train_loss,
                "({:0.2f},{:0.2f},{:0.2f})".format(
                    train_dMOTP_loss, train_dMOTA_loss, train_act_loss
                ),
                val_loss,
                "({:0.2f},{:0.2f},{:0.2f})".format(
                    val_dMOTP_loss, val_dMOTA_loss, val_act_loss
                ),
                loc_error,
                "{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}".format(
                    val_mota,
                    val_ids,
                    val_recall_doa,
                    val_precision_doa,
                    val_fscore_doa,
                ),
                best_val_epoch,
                "({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})".format(
                    best_doa,
                    best_mota,
                    best_ids,
                    best_recall,
                    best_precision,
                    best_fscore,
                ),
            )
        )

        (
            tr_loss_list[epoch_cnt],
            val_loss_list[epoch_cnt],
            hung_val_loss_list[epoch_cnt],
        ) = (train_loss, val_loss, loc_error)
        plot_functions(
            unique_name,
            tr_loss_list,
            val_loss_list,
            hung_tr_loss_list,
            hung_val_loss_list,
        )

    checkpoint_path = model_name.replace(".h5", f"_epoch{best_val_epoch}.h5")
    evaluate_on_test_set(checkpoint_path, params, data_gen_test, trainer)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
