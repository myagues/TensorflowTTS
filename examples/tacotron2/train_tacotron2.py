# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train Tacotron 2."""

import argparse
import logging
import os

import tensorflow_tts

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.optimizers import AdamWeightDecay
from tensorflow_tts.utils import TFGriffinLim
from tqdm import tqdm

from tacotron_dataset import CharMelDataset

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron 2 Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(
        self, config, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from YAML format configuration file.
            is_mixed_precision (bool): Whether or not to use mixed precision.
        """
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary to log them
        list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss",
        ]
        self.init_train_eval_metrics(list_metrics_name)
        self.reset_metric_states(self.train_metrics)
        self.reset_metric_states(self.eval_metrics)

        self.config = config
        self.griffin_lim_tf = None
        if "dataset_config_path" in config:
            config = yaml.load(open(config["dataset_config_path"]), Loader=yaml.Loader)
            self.griffin_lim_tf = TFGriffinLim(config, config["stats_path"])

    def init_train_eval_metrics(self, list_metrics_name):
        """Init train and eval metrics to save it to TensorBoard."""
        self.train_metrics = {}
        self.eval_metrics = {}
        for name in list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name="train_" + name, dtype=tf.float32)}
            )
            self.eval_metrics.update(
                {name: tf.keras.metrics.Mean(name="eval_" + name, dtype=tf.float32)}
            )

    def reset_metric_states(self, metric_dict):
        """Reset metrics after writing to TensorBoard."""
        for metric in metric_dict.values():
            metric.reset_states()

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # create scheduler for teacher forcing.
        self.teacher_forcing_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.config["start_ratio_value"],
            decay_steps=self.config["schedule_decay_steps"],
            end_learning_rate=self.config["end_ratio_value"],
            cycle=True,
            name="teacher_forcing_scheduler",
        )

    def _train_step(self, batch):
        """Train model one step."""
        batch_data = map(batch.get, ["char", "char_len", "mel", "mel_len", "ga"])
        self._one_step_tacotron2(*batch_data)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()
        self._apply_schedule_teacher_forcing()

    def _apply_schedule_teacher_forcing(self):
        if self.steps >= self.config["start_schedule_teacher_forcing"]:
            # change _ratio on sampler.
            self.model.decoder.sampler._ratio = self.teacher_forcing_scheduler(
                self.steps - self.config["start_schedule_teacher_forcing"]
            )
            if self.steps == self.config["start_schedule_teacher_forcing"]:
                logging.info("(Steps: %d) Start schedule teacher forcing.", self.steps)

    @tf.function(experimental_relax_shapes=True)
    def _one_step_tacotron2(self, char, char_length, mel, mel_length, guided_att):
        bs, seq_len, _ = tf.unstack(tf.shape(mel))
        with tf.GradientTape() as tape:
            (mel_outputs, post_mel_outputs, stop_outputs, alignments) = self.model(
                char,
                char_length,
                speaker_ids=tf.zeros([bs]),
                mel_outputs=mel,
                mel_lengths=mel_length,
                training=True,
            )
            # calculate mel loss
            mel_loss_before = self.mae(mel, mel_outputs)
            mel_loss_after = self.mae(mel, post_mel_outputs)

            # calculate stop ground truth based on "mel_len"
            stop_gts = ~tf.sequence_mask(mel_length, seq_len)
            stop_token_loss = self.binary_crossentropy(stop_gts, stop_outputs)

            # calculate guided attention loss
            att_mask = tf.cast(tf.math.not_equal(guided_att, -1.0), tf.float32)
            loss_att = tf.reduce_sum(tf.abs(alignments * guided_att) * att_mask)
            loss_att /= tf.reduce_sum(att_mask)

            # sum all loss
            loss = stop_token_loss + mel_loss_before + mel_loss_after + loss_att

            if self.is_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(
                scaled_loss, self.model.trainable_variables
            )
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables), 5.0
        )
        # accumulate loss into metrics
        self.train_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.train_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.train_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.train_metrics["guided_attention_loss"].update_state(loss_att)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info("(Steps: %d) Start evaluation.", self.steps)

        # set traing = False on decoder_cell
        self.model.decoder.cell.training = False

        # calculate loss for each batch
        for idx, batch in enumerate(tqdm(self.eval_data_loader, desc="[eval]"), 1):
            # eval one step
            batch_items = map(batch.get, ["char", "char_len", "mel", "mel_len", "ga"])
            mel_outputs, post_mel_outputs, _, alignments = self._eval_step(*batch_items)

            if idx <= self.config["num_save_intermediate_results"]:
                self.generate_and_save_intermediate_result(
                    batch, mel_outputs, post_mel_outputs, alignments
                )

        logging.info(
            "(Steps: %d) Finished evaluation (%d steps per epoch).", self.steps, idx
        )
        # average loss
        for key, val in self.eval_metrics.items():
            logging.info("(Steps: %d) eval_%s = %.4f.", self.steps, key, val.result())

        # record
        self._write_to_tensorboard(self.eval_metrics, stage="eval")

        # reset
        self.reset_metric_states(self.eval_metrics)

        # enable training = True on decoder_cell
        self.model.decoder.cell.training = True

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, char, char_length, mel, mel_length, guided_att):
        """Evaluate model one step."""
        bs, seq_len, _ = tf.unstack(tf.shape(mel))
        mel_outputs, post_mel_outputs, stop_outputs, alignments = self.model(
            char,
            char_length,
            speaker_ids=tf.zeros([bs]),
            mel_outputs=mel,
            mel_lengths=mel_length,
            training=False,
        )
        # calculate mel loss
        mel_loss_before = self.mae(mel, mel_outputs)
        mel_loss_after = self.mae(mel, post_mel_outputs)

        # calculate stop ground truth based on "mel_length"
        stop_gts = ~tf.sequence_mask(mel_length, seq_len)
        stop_token_loss = self.binary_crossentropy(stop_gts, stop_outputs)

        # calculate guided attention loss
        att_mask = tf.cast(tf.math.not_equal(guided_att, -1.0), tf.float32)
        loss_att = tf.reduce_sum(tf.abs(alignments * guided_att) * att_mask)
        loss_att /= tf.reduce_sum(att_mask)

        # accumulate loss into metrics
        self.eval_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.eval_metrics["guided_attention_loss"].update_state(loss_att)
        return mel_outputs, post_mel_outputs, stop_outputs, alignments

    def _check_log_interval(self):
        """Log to TensorBoard."""
        if self.steps % self.config["log_interval_steps"] == 0:
            for key, val in self.train_metrics.items():
                logging.info(
                    "(Step: %d) train_%s = %.4f.", self.steps, key, val.result()
                )
            self._write_to_tensorboard(self.train_metrics, stage="train")
            # reset metric states
            self.reset_metric_states(self.train_metrics)

    @tf.function(experimental_relax_shapes=True)
    def predict(self, char, char_length, mel, mel_length):
        """Predict."""
        bs, _, _ = tf.unstack(tf.shape(mel))
        mel_outputs, post_mel_outputs, _, alignments = self.model(
            char,
            char_length,
            speaker_ids=tf.zeros([bs]),
            mel_outputs=mel,
            mel_lengths=mel_length,
            training=False,
        )
        return mel_outputs, post_mel_outputs, alignments

    def generate_and_save_intermediate_result(
        self, batch, mel_before, mel_after, alignments
    ):
        """Generate and save intermediate result."""
        # check directory
        pred_output_dir = os.path.join(
            self.config["outdir"], "predictions", f"{self.steps}_steps"
        )
        os.makedirs(pred_output_dir, exist_ok=True)

        # get audio samples output with Griffin-Lim algorithm
        if self.griffin_lim_tf:
            tf_wav = self.griffin_lim_tf(mel_after)

        num_mels = self.config["tacotron2_params"]["n_mels"]
        items = zip(
            batch["utt_id"], batch["mel"], mel_before, mel_after, alignments, tf_wav
        )
        for (utt_id, mel_gt, mel_pred_before, mel_pred_after, alignment, wav) in items:
            # reshape outputs and convert to values
            mel_gt = tf.reshape(mel_gt, (-1, num_mels)).numpy()
            mel_pred_before = tf.reshape(mel_pred_before, (-1, num_mels)).numpy()
            mel_pred_after = tf.reshape(mel_pred_after, (-1, num_mels)).numpy()
            utt_id_str = utt_id.numpy().decode("utf8")

            # save audio file
            if self.griffin_lim_tf:
                self.griffin_lim_tf.save_wav(wav, pred_output_dir, utt_id_str)

            # plot mel
            figname = os.path.join(pred_output_dir, f"{utt_id_str}.png")
            fig, axes = plt.subplots(figsize=(10, 8), nrows=3)
            for ax, data in zip(axes, [mel_gt, mel_pred_before, mel_pred_after]):
                im = ax.imshow(np.rot90(data), aspect="auto", interpolation="none")
                fig.colorbar(im, pad=0.02, aspect=15, orientation="vertical", ax=ax)
            axes[0].set_title(f"Target mel spectrogram ({utt_id_str})")
            axes[1].set_title(
                f"Predicted mel spectrogram before post-net @ {self.steps} steps"
            )
            axes[2].set_title(
                f"Predicted mel spectrogram after post-net @ {self.steps} steps"
            )
            plt.tight_layout()
            plt.savefig(figname, bbox_inches="tight")
            plt.close()

            # plot alignment
            figname = os.path.join(pred_output_dir, f"{utt_id_str}_alignment.png")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, aspect=15, ax=ax)
            ax.set_xlabel("Decoder timestep")
            ax.set_ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname, bbox_inches="tight")
            plt.close()

    def _check_train_finish(self):
        """Check whether or not training has finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def fit(self, train_dataset, valid_dataset, saved_path, resume=None):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(valid_dataset)
        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        if len(resume) > 2:
            self.load_checkpoint(resume)
            logging.info("Successfully resumed from %s.", resume)
        self.run()


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Train Tacotron 2.")
    parser.add_argument(
        "--train_dir",
        default=argparse.SUPPRESS,
        type=str,
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--valid_dir",
        default=argparse.SUPPRESS,
        type=str,
        help="Directory containing validation data.",
    )
    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="Whether or not to use normalized features.",
    )
    parser.add_argument(
        "--stats_path",
        default=argparse.SUPPRESS,
        type=str,
        help="Path to statistics file with mean and std values for standardization.",
    )
    parser.add_argument(
        "--outdir",
        default=argparse.SUPPRESS,
        type=str,
        help="Output directory where checkpoints and results will be saved.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help="Checkpoint file path to resume training. (default='')",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO, 2: WARN.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Whether or not to use mixed precision training.",
    )
    args = parser.parse_args()

    # load and save config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__

    # set logger and print parameters
    fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN}
    logging.basicConfig(level=log_level[config["verbose"]], format=fmt)
    _ = [logging.info("%s = %s", key, value) for key, value in config.items()]

    # check required arguments
    missing_dirs = list(
        filter(lambda x: x not in config, ["train_dir", "valid_dir", "outdir"])
    )
    if missing_dirs:
        raise ValueError(f"{missing_dirs}.")
    if config["use_norm"] and "stats_path" not in config:
        raise ValueError("'--stats_path' should be provided when using '--use_norm'.")
    if not config["use_norm"]:
        config["stats_path"] = None

    # check output directory
    os.makedirs(config["outdir"], exist_ok=True)

    with open(os.path.join(config["outdir"], "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)

    # set mixed precision config
    if config["mixed_precision"]:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    # get dataset
    train_dataset = CharMelDataset(
        dataset_dir=config["train_dir"],
        use_norm=config["use_norm"],
        stats_path=config["stats_path"],
        return_guided_attention=True,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        n_mels=config["tacotron2_params"]["n_mels"],
        use_fixed_shapes=config["use_fixed_shapes"],
        mel_len_threshold=config["mel_length_threshold"],
    ).create(
        batch_size=config["batch_size"],
        shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        training=True,
    )

    valid_dataset = CharMelDataset(
        dataset_dir=config["valid_dir"],
        use_norm=config["use_norm"],
        stats_path=config["stats_path"],
        return_guided_attention=True,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        n_mels=config["tacotron2_params"]["n_mels"],
        use_fixed_shapes=False,
        mel_len_threshold=config["mel_length_threshold"],
    ).create(batch_size=config["batch_size"], allow_cache=config["allow_cache"])

    tacotron_config = Tacotron2Config(**config["tacotron2_params"])
    tacotron2 = TFTacotron2(config=tacotron_config, training=True, name="tacotron2")
    tacotron2._build()
    tacotron2.summary()

    # define trainer
    trainer = Tacotron2Trainer(
        config=config, steps=0, epochs=0, is_mixed_precision=config["mixed_precision"]
    )

    # AdamW for Tacotron 2
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_steps=config["optimizer_params"]["decay_steps"],
        end_learning_rate=config["optimizer_params"]["end_learning_rate"],
    )

    learning_rate_fn = WarmUp(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=int(
            config["train_max_steps"] * config["optimizer_params"]["warmup_proportion"]
        ),
    )
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config["optimizer_params"]["weight_decay"],
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )

    # compile trainer
    trainer.compile(model=tacotron2, optimizer=optimizer)

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints"),
            resume=config["resume"],
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info("Successfully saved checkpoint @ %d steps.", trainer.steps)


if __name__ == "__main__":
    main()
