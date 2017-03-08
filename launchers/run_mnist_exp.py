from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import sys
import os
from infogan.misc.datasets import MnistDataset
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

import gflags

if __name__ == "__main__":

    FLAGS = gflags.FLAGS
    gflags.DEFINE_string("restore_point", None, "(optional) Path to restore model from. ")
    gflags.DEFINE_string("experiment_name", None, "Experiment name.")
    gflags.DEFINE_integer("max_epoch", 1, "Number of training epochs.")

    FLAGS(sys.argv)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"
    batch_size = 128
    updates_per_epoch = 100

    exp_name = "mnist_%s" % timestamp

    if FLAGS.experiment_name is None:
        FLAGS.experiment_name = exp_name

    log_dir = os.path.join(root_log_dir, FLAGS.experiment_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, FLAGS.experiment_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = MnistDataset()

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=FLAGS.experiment_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=FLAGS.max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train(
        restore_point=FLAGS.restore_point,
    )
