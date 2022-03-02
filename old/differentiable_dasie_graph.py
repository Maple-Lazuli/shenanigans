"""
A library for the computation of DASIE MTF ensembles.
Author: Justin Fletcher
"""

import os
import math
import time
import copy
import json
import math
import glob
import hcipy
import codecs
import joblib
import datetime
import argparse
import itertools

import pandas as pd
import numpy as np

from decimal import Decimal

# TODO: Refactor this import.
from dataset_generator import DatasetGenerator

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Tentative.
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import tensorflow as tf


# First, prevent TensorFlow from foisting filthy eager execution upon us.
tf.compat.v1.disable_eager_execution()

def cosine_similarity(u, v):
    """
    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching u in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """

    u = tf.reshape(u, [-1])
    v = tf.reshape(v, [-1])
    u = tf.cast(u, tf.float64)
    v = tf.cast(v, tf.float64)

    projection = tf.tensordot(u, v, axes=1)

    norm_product = tf.math.multiply(tf.norm(u), tf.norm(v))

    cos_sim = tf.math.divide(projection, norm_product)

    return cos_sim


def generalized_gaussian(X, mu, alpha, beta):
    """
    :param X: A 1d array onto which the gaussian wil be projected
    :param mu: The mean of the gaussian.
    :param alpha:
    :param beta:
    :return: A vector with shape matching X.
    """

    return (beta / (2 * alpha * math.gamma(1 / beta))) * np.exp(
        -(np.abs(X - mu) / alpha) ** beta)


def plane_2d(x, y, x_0, y_0, slope_x, slope_y, height):
    return ((x - x_0) * slope_x) + ((y - y_0) * slope_y) + height


@np.vectorize
def generalized_gaussian_2d(u, v, mu_u, mu_v, alpha, beta):
    scale_constant = (beta / (2 * alpha * tf.exp(tf.math.lgamma((1 / beta)))))

    exponent = -(((u - mu_u) ** 2 + (v - mu_v) ** 2) / alpha) ** beta

    value = tf.exp(exponent)

    z = scale_constant * value

    return z

def tensor_generalized_gaussian_2d(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, alpha, beta = T

    # TODO: This is horrible, but works around tf.math.lgamma not supporting real valued complex datatypes.
    u = tf.cast(u, dtype=tf.float64)
    v = tf.cast(v, dtype=tf.float64)
    mu_u = tf.cast(mu_u, dtype=tf.float64)
    mu_v = tf.cast(mu_v, dtype=tf.float64)
    alpha = tf.cast(alpha, dtype=tf.float64)
    beta = tf.cast(beta, dtype=tf.float64)

    scale_constant = beta / (2 * alpha * tf.exp(tf.math.lgamma((1 / beta))))
    exponent = -(((u - mu_u) ** 2 + (v - mu_v) ** 2) / alpha) ** beta
    value = tf.exp(exponent)
    z = scale_constant * value

    # TODO: This is horrible, but works around tf.math.lgamma not supporting real valued complex datatypes.
    z = tf.cast(z, dtype=tf.complex128)

    return z

@np.vectorize
def circle_mask(X, Y, x_center, y_center, radius):
    r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
    return r < radius

def aperture_function_2d(X, Y, mu_u, mu_v, alpha, beta, tip, tilt, piston):

    print("Starting aperture function.")
    # generalized_gaussian_2d_sample = tf.vectorized_map(generalized_gaussian_2d, X, Y, mu_u, mu_v, alpha, beta)
    T_mu_u = tf.ones_like(X) * mu_u
    T_mu_v = tf.ones_like(X) * mu_v
    T_alpha = tf.ones_like(X) * alpha
    T_beta = tf.ones_like(X) * beta
    T = (X, Y, T_mu_u, T_mu_v, T_alpha, T_beta)
    generalized_gaussian_2d_sample = tf.vectorized_map(tensor_generalized_gaussian_2d, T)

    # generalized_gaussian_2d_sample = generalized_gaussian_2d(X, Y, mu_u, mu_v,
    #                                                          alpha, beta)

    # Subsistitute simple circular mask function
    # Warning!  Ugly alpha fudge factor!
    # subaperture_radius = alpha*4
    # generalized_gaussian_2d_sample = circle_mask(X, Y, mu_u, mu_v, subaperture_radius)

    print("getting plane.")
    plane_2d_sample = plane_2d(X, Y, mu_u, mu_v, tip, tilt, piston)

    # print(g.gradient(plane_2d_sample, ttp_variables))
    # die

    # The piston tip and tilt are encoded as the phase-angle of pupil plane
    print("generating phase angle field.")
    plane_2d_field = tf.exp(plane_2d_sample)
    # plane_2d_field = tf.exp(1.j * plane_2d_sample)

    print("multiplying.")
    generalized_gaussian_2d_sample = generalized_gaussian_2d_sample
    aperture_sample = plane_2d_field * generalized_gaussian_2d_sample

    print(aperture_sample)

    print("Ending aperture function.")
    return aperture_sample

class DASIEModel(object):

    def __init__(self,
                 sess,
                 train_dataset,
                 valid_dataset,
                 inputs=None,
                 learning_rate=1.0,
                 num_apertures=15,
                 spatial_quantization=256,
                 num_exposures=1,
                 alpha=0.001,
                 writer=None):

        self.learning_rate = learning_rate
        self.num_apertures = num_apertures
        self.sess = sess
        self.writer = writer


        train_iterator = train_dataset.get_iterator()
        self.train_iterator_handle = sess.run(train_iterator.string_handle())

        valid_iterator = valid_dataset.get_iterator()
        self.valid_iterator_handle = sess.run(valid_iterator.string_handle())

        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        # Abstract specific iterators as type.
        iterator_output_types = train_iterator.output_types
        iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle, iterator_output_types)
        dataset_batch = iterator.get_next()

        with tf.name_scope("dasie_model"):


            self.inputs, self.output_images = self._build_dasie_model(
                inputs=dataset_batch,
                spatial_quantization=spatial_quantization,
                num_apertures=self.num_apertures,
                alpha=alpha,
                num_exposures=num_exposures,
                )

            # self.trainable_variables = tf.compat.v1.trainable_variables()

    def _build_dasie_model(self,
                           inputs=None,
                           num_apertures=15,
                           num_exposures=1,
                           radius_meters=1.25,
                           field_of_view_arcsec=4.0,
                           spatial_quantization=256,
                           alpha=0.01,
                           beta=10.0,
                           filter_wavelength_micron=1.0):

        # Construct placeholders for inputs.
        # TODO: expand to batches as shown:
        # batch_shape = (None,) + (spatial_quantization, spatial_quantization)
        batch_shape = (spatial_quantization, spatial_quantization)
        if inputs is not None:

            # TODO: Add shape test here.
            perfect_image_placeholder = inputs

        else:
            perfect_image_placeholder = tf.compat.v1.placeholder(tf.float64,
                                                                 shape=batch_shape,
                                                                 name="perfect_image_batch")



        # Compute the pupil extent: 4.848 microradians / arcsec
        # pupil_extend = [m] * [count] / ([microradians / arcsec] * [arcsec])
        # pupil_extent = [count] [m] / [microradian]
        # pixel_extent = [m] / [microradian]
        pupil_extent = filter_wavelength_micron * spatial_quantization / (4.848 * field_of_view_arcsec)

        # Build the simulation mesh grid.
        x = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        y = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        X, Y = np.meshgrid(x, y)
        X = tf.complex(tf.constant(X), tf.constant(0.0, dtype=tf.float64))
        Y = tf.complex(tf.constant(Y), tf.constant(0.0, dtype=tf.float64))


        randomize_initial_states = True
        tip_std = 1.0
        tilt_std = 1.0
        piston_std = 1.0

        self.num_exposures = num_exposures

        # TODO: Migrate this out to a separate function.
        output_ttp_from_model = True
        target_output_shape = (num_exposures, num_apertures, 3)
        target_output_size = np.prod(target_output_shape)
        print(target_output_shape)
        print(target_output_size)

        with tf.name_scope("hidden_layer_1"):
            random_vector_size = target_output_size * 6
            n_hidden_1 = target_output_size * 4

            glorot_relu_init_std = np.sqrt(2 / (random_vector_size + n_hidden_1))
            W = tf.Variable(tf.random.normal((random_vector_size, n_hidden_1), stddev=glorot_relu_init_std))
            b = tf.Variable(tf.random.normal((n_hidden_1,), stddev=glorot_relu_init_std))
            x = tf.matmul(tf.constant(np.random.normal(size=(1, random_vector_size)), dtype=tf.float32), W) + b
            # net = tflearn.layers.normalization.batch_normalization(net)
            x = tf.nn.relu(x)
        with tf.name_scope("hidden_layer_2"):
            n_hidden_2 = target_output_size * 2
            glorot_relu_init_std = np.sqrt(2 / (n_hidden_1 + n_hidden_2))
            W = tf.Variable(tf.random.normal((n_hidden_1, n_hidden_2), stddev=glorot_relu_init_std))
            b = tf.Variable(tf.random.normal((n_hidden_2,), stddev=glorot_relu_init_std))
            x = tf.matmul(x, W) + b
            # net = tflearn.layers.normalization.batch_normalization(net)
            x = tf.nn.relu(x)
        with tf.name_scope("hidden_layer_3"):
            n_hidden_3 = target_output_size
            glorot_tanh_init_std = np.sqrt(2) * np.sqrt(2 / (n_hidden_2 + n_hidden_3))
            W = tf.Variable(tf.random.normal((n_hidden_2, n_hidden_3), stddev=glorot_tanh_init_std))
            b = tf.Variable(tf.random.normal((n_hidden_3,), stddev=glorot_tanh_init_std))
            x = tf.matmul(x, W) + b
            # net = tflearn.layers.normalization.batch_normalization(net)
            # x = tf.nn.relu(x)
            x = tf.math.tanh(x)

        scale_factor = 0.1
        x = scale_factor * x
        x = tf.reshape(x, [num_exposures, num_apertures, 3])
        x = tf.cast(x, dtype=tf.float64)


        self.pupil_planes = list()
        for exposure_num in range(num_exposures):
            with tf.name_scope("exposure_" + str(exposure_num)):

                # Construct subaperture TF Variables.
                ttp_variables = list()

                # Iterate over each aperture and build the TF variables needed.
                for aperture_num in range(num_apertures):
                    with tf.name_scope("subaperture_variables_" + str(aperture_num)):

                        # If we have a model, use it's outputs as the ttp.
                        if output_ttp_from_model:

                            # TODO: Rename model output space or depricate.
                            # Construct the variables wrt which we differentiate.
                            # TODO: Add names to slices, somehow.
                            tip_variable_name = str(aperture_num) + "_tip"

                            tip_parameter = x[exposure_num, aperture_num, 0]
                            # tip_parameter = tf.Variable(tip_distribution,
                            #                             dtype=tf.float64,
                            #                             name=tip_variable_name)
                            tip = tf.complex(tip_parameter,
                                             tf.constant(0.0, dtype=tf.float64))

                            # microns / meter (not far off from microradian tilt)
                            # TODO: Add names to slices, somehow.
                            tilt_variable_name = str(aperture_num) + "_tilt"
                            # tilt_parameter = tf.Variable(tilt_distribution,
                            #                              dtype=tf.float64,
                            #                              name=tilt_variable_name)

                            tilt_parameter = x[exposure_num, aperture_num, 1]
                            tilt = tf.complex(tilt_parameter,
                                              tf.constant(0.0, dtype=tf.float64))
                            # microns
                            # TODO: Add names to slices, somehow.
                            piston_variable_name = str(aperture_num) + "_piston"

                            # piston_parameter = tf.Variable(piston_distribution,
                            #                                dtype=tf.float64,
                            #                                name=piston_variable_name)

                            piston_parameter = x[exposure_num, aperture_num, 2]
                            piston = tf.complex(piston_parameter,
                                                tf.constant(0.0, dtype=tf.float64))
                            ttp_variables.append([tip, tilt, piston])

                        # If we don't have a model, then ttp are the variables.
                        else:

                            # We can either init each variable randomly...
                            if randomize_initial_states:

                                tip_distribution = np.random.normal(0.0, tip_std)
                                tilt_distribution = np.random.normal(0.0, tilt_std)
                                piston_distribution = 0.001 + np.abs(np.random.normal(0.0, piston_std))

                            # ...or init them to the same small values.
                            else:

                                tip_distribution = 0.0
                                tilt_distribution = 0.0
                                piston_distribution = 0.001

                            # Use a constant real-valued complex number as a gradient stop to the
                            # imaginary part of the t/t/p variables.

                            # Construct the variables wrt which we differentiate.
                            tip_variable_name = str(aperture_num) + "_tip"
                            tip_parameter = tf.Variable(tip_distribution,
                                                        dtype=tf.float64,
                                                        name=tip_variable_name)
                            tip = tf.complex(tip_parameter,
                                             tf.constant(0.0, dtype=tf.float64))

                            # microns / meter (not far off from microradian tilt)
                            tilt_variable_name = str(aperture_num) + "_tilt"
                            tilt_parameter = tf.Variable(tilt_distribution,
                                                         dtype=tf.float64,
                                                         name=tilt_variable_name)
                            tilt = tf.complex(tilt_parameter,
                                              tf.constant(0.0, dtype=tf.float64))
                            # microns
                            piston_variable_name = str(aperture_num) + "_piston"

                            piston_parameter = tf.Variable(piston_distribution,
                                                           dtype=tf.float64,
                                                           name=piston_variable_name)
                            piston = tf.complex(piston_parameter,
                                                tf.constant(0.0, dtype=tf.float64))

                            ttp_variables.append([tip, tilt, piston])

                # Construct the model of the pupil plane, conditioned on the Variables.
                with tf.name_scope("pupil_plane_model"):

                    # Initialize the pupil plan grid.
                    pupil_plane = tf.zeros((spatial_quantization, spatial_quantization),  dtype=tf.complex128)

                    for aperture_num in range(num_apertures):

                        with tf.name_scope("subaperture_model_" + str(aperture_num)):

                            print("Building aperture number %d." % aperture_num)

                            # Parse the tip, tilt, & piston variables for the aperture.
                            tip, tilt, piston = ttp_variables[aperture_num]

                            # Set piston, in microns, encoded in phase angle.
                            pison_phase = 2 * np.pi * piston / filter_wavelength_micron

                            # Set tip & tilt, in microns/meter, encoded in phase angle.
                            # (micron/meter) ~= (microradians of tilt)
                            tip_phase = 2 * np.pi * tip / filter_wavelength_micron
                            tilt_phase = 2 * np.pi * tilt / filter_wavelength_micron

                            rotation = (aperture_num + 1) / self.num_apertures

                            # TODO: correct radius to place the edge, rather than the center, at radius

                            mu_u = radius_meters * tf.cos((2 * np.pi) * rotation)
                            mu_v = radius_meters * tf.sin((2 * np.pi) * rotation)
                            mu_u = tf.cast(mu_u, dtype=tf.complex128)
                            mu_v = tf.cast(mu_v, dtype=tf.complex128)

                            pupil_plane += aperture_function_2d(X,
                                                                Y,
                                                                mu_u,
                                                                mu_v,
                                                                alpha,
                                                                beta,
                                                                tip_phase,
                                                                tilt_phase,
                                                                pison_phase)

                self.pupil_planes.append(pupil_plane)

        self.psfs = list()
        self.otfs = list()
        self.mtfs = list()
        self.distributed_aperture_images = list()

        # Iterate over each pupil plane, one per exposure.
        for pupil_plane in self.pupil_planes:


            # Basically the following is the loss function of the pupil plane.
            # The pupil plan here can be thought of an estimator, parameterized by
            # t/t/p values, of the true image.

            # Compute the PSF from the pupil plane.
            # TODO: ask for advice, should I NOT be taking the ABS here?
            with tf.name_scope("psf_model"):

                psf = tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(pupil_plane)))) ** 2
                self.psfs.append(psf)

            # Compute the OTF, which is the Fourier transform of the PSF.
            with tf.name_scope("otf_model"):

                otf = tf.signal.fft2d(tf.cast(psf, tf.complex128))
                self.otfs.append(otf)

            # Compute the mtf, which is the real component of the OTF.
            with tf.name_scope("mtf_model"):

                mtf = tf.math.abs(otf)
                self.mtfs.append(mtf)

            with tf.name_scope("image_spectrum_model"):
                self.perfect_image_spectrum = tf.signal.fft2d(tf.cast(tf.squeeze(perfect_image_placeholder, axis=-1), dtype=tf.complex128))


            with tf.name_scope("image_plane_model"):
                distributed_aperture_image_spectrum = self.perfect_image_spectrum * tf.cast(mtf, dtype=tf.complex128)
                distributed_aperture_image_plane = tf.abs(tf.signal.fft2d(distributed_aperture_image_spectrum))
                distributed_aperture_image_plane = distributed_aperture_image_plane / tf.reduce_max(distributed_aperture_image_plane)


            with tf.name_scope("sensor_model"):

                # TODO: Implement Gaussian and Poisson process noise.
                distributed_aperture_image = distributed_aperture_image_plane


                self.distributed_aperture_images.append(distributed_aperture_image)

        # Now, construct a monolithic aperture of the same radius.
        with tf.name_scope("monolithic_aperture"):

            with tf.name_scope("monolithic_aperture_pupil_plane"):
                # TODO: Adjust this to be physically correct.
                monolithic_alpha = np.pi * radius_meters / 1 / 4
                self.monolithic_pupil_plane = aperture_function_2d(X, Y, 0.0, 0.0, monolithic_alpha, beta, tip=0.0, tilt=0.0, piston=0.001)

                # Compute the PSF from the pupil plane.
                # TODO: ask for advice, should I NOT be taking the ABS here?
                with tf.name_scope("monolithic_psf_model"):
                    self.monolithic_psf = tf.math.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(self.monolithic_pupil_plane)))) ** 2

                # Compute the OTF, which is the Fourier transform of the PSF.
                with tf.name_scope("monolithic_otf_model"):
                    self.monolithic_otf = tf.signal.fft2d(tf.cast(self.monolithic_psf, tf.complex128))

                # Compute the mtf, which is the real component of the OTF.
                with tf.name_scope("monolithic_mtf_model"):
                    self.monolithic_mtf = tf.math.abs(self.monolithic_otf)

                with tf.name_scope("image_spectrum_model"):
                    self.perfect_image_spectrum = tf.signal.fft2d(tf.cast(tf.squeeze(perfect_image_placeholder, axis=-1), dtype=tf.complex128))

                with tf.name_scope("monolithic_image_plane_model"):
                    monolithic_aperture_image_spectrum = self.perfect_image_spectrum * tf.cast(self.monolithic_mtf, dtype=tf.complex128)
                    monolithic_aperture_image = tf.abs(tf.signal.fft2d(monolithic_aperture_image_spectrum))
                    self.monolithic_aperture_image = monolithic_aperture_image / tf.reduce_max(monolithic_aperture_image)

        # TODO: refactor reconstruction out of this model
        with tf.name_scope("distributed_aperture_image_recovery"):

            # Combine the ensemble of images with the restoration function.
            self.recovered_image = self._build_recovery_model(self.distributed_aperture_images)

            # self.recovered_image = tf.math.reduce_mean(self.distributed_aperture_images, 0)

        with tf.name_scope("dasie_loss"):
            perfect_image_flipped = tf.squeeze(perfect_image_placeholder, axis=-1)
            perfect_image_flipped = tf.reverse(perfect_image_flipped, [0])
            perfect_image_flipped = tf.reverse(perfect_image_flipped, [-1])
            perfect_image_flipped = tf.reverse(perfect_image_flipped, [1])
            # perfect_image_flipped = tf.reverse(perfect_image_placeholder, [1])
            # perfect_image_flipped = tf.reverse(perfect_image_placeholder, [0])
            # perfect_image_flipped = perfect_image_placeholder
            # perfect_image_flipped = tf.squeeze(perfect_image_flipped, axis=-1)
            # self.distributed_aperture_image_cosine_similarity = cosine_similarity(self.recovered_image, perfect_image_flipped)
            # self.loss = -tf.math.log(self.distributed_aperture_image_cosine_similarity)

            # self.distributed_aperture_mtf_cosine_similarity = cosine_similarity(self.mtfs[0], self.perfect_image_spectrum)
            # self.loss = -tf.math.log(self.distributed_aperture_mtf_cosine_similarity)

            # TODO: Explore other losses.
            self.image_mse = tf.reduce_mean((self.recovered_image - perfect_image_flipped)**2)
            self.loss = tf.math.log(self.image_mse)
            # self.loss = self.image_difference

            # self.distributed_aperture_mtf_product = tf.reduce_sum(tf.abs(tf.cast(self.mtfs[0], dtype=tf.complex128) * self.perfect_image_spectrum))
            # self.loss = -tf.math.log(self.distributed_aperture_mtf_product)
            # self.loss = -self.distributed_aperture_mtf_product

        # I wonder if this works...
        with self.writer.as_default():

            # TODO: replace all these endpoints with _batch...
            tf.summary.scalar("in_graph_loss", self.loss)
            # tf.summary.scalar("monolithic_aperture_image_cosine_similarity", self.monolithic_aperture_image_cosine_similarity)
            tf.summary.scalar("monolithic_aperture_image_mse", tf.reduce_mean((self.monolithic_aperture_image - perfect_image_flipped) ** 2))
            # tf.summary.scalar("distributed_aperture_image_cosine_similarity", self.distributed_aperture_image_cosine_similarity)
            # tf.summary.scalar("distributed_aperture_mtf_cosine_similarity", self.distributed_aperture_mtf_cosine_similarity)
            tf.summary.scalar("distributed_aperture_image_mse", tf.reduce_mean((self.recovered_image - perfect_image_flipped)**2))
            tf.summary.scalar("da_mse_mono_mse_ratio", tf.reduce_mean((self.recovered_image - perfect_image_flipped)**2) / tf.reduce_mean((self.monolithic_aperture_image - perfect_image_flipped) ** 2))

        with tf.name_scope("dasie_optimizer"):
            # Build an op that applies the policy gradients to the model.
            # self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))
            self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimize = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.compat.v1.Graph().as_default():
            tf.summary.scalar("debug_metric", 0.5)

        self.summaries = tf.compat.v1.summary.all_v2_summary_ops()

        # Maybe remove or move these.
        self.perfect_image_flipped = perfect_image_flipped
        self.perfect_image = perfect_image_placeholder
        self.ttp_variables = ttp_variables

        output_batch = self.recovered_image
        return perfect_image_placeholder, output_batch


    def _build_recovery_model(self, distributed_aperture_images_batch):
        """

        :param distributed_aperture_images_batch: a batch of num_exposure
        images.
        :return:
        """

        with tf.name_scope("recovery_model"):

            # # TODO: Validate the shape of this stack.
            # # distributed_aperture_images_batch = tf.expand_dims(distributed_aperture_images, axis=-1)
            # Stack the the images in the ensemble to form a batch of inputs.
            distributed_aperture_images_batch = tf.stack(distributed_aperture_images_batch, axis=-1)
            with tf.name_scope("recovery_feature_extractor"):
                x = distributed_aperture_images_batch
                x = self.conv_block(x,
                                    input_channels=self.num_exposures,
                                    output_channels=15,
                                    kernel_size=7,
                                    stride=1,
                                    activation="LRelu")

            recovered_image_batch = x

            # TODO: Correct this to output the batch shape - input less the ensemble dimension.
            recovered_image = tf.math.reduce_mean(recovered_image_batch[0], -1)

        return recovered_image

    def conv_block(self,
                   input_feature_map,
                   input_channels,
                   output_channels=1,
                   kernel_size=2,
                   stride=1,
                   activation="LRelu",
                   name=None):

        if not name:

            name = "conv-c" + str(output_channels) + "-k" + str(kernel_size) + "-s" + str(stride) + "-" + activation

        with tf.name_scope(name):

            # Initialize the filter variables as he2015delving.
            he_relu_init_std = np.sqrt(2 / (input_channels * (kernel_size**2)))
            filters = tf.Variable(tf.random.normal((kernel_size,
                                                    kernel_size,
                                                    input_channels,
                                                    output_channels),
                                                   stddev=he_relu_init_std,
                                                   dtype=tf.float64))

            # Encode the strides for TensorFlow, and build the conv graph.

            # print("make this shape match (8, 256, 256, 2)")
            # print(input_feature_map.shape)
            # nowdie
            strides = [1, stride, stride, 1]
            conv_output = tf.nn.conv2d(input_feature_map,
                                       filters,
                                       strides,
                                       padding="SAME",
                                       data_format='NHWC',
                                       dilations=None,
                                       name=None)

            # Apply an activation function.
            output_feature_map = tf.nn.leaky_relu(conv_output, alpha=0.02)

        return output_feature_map


    def plot(self, logdir=None, step=None):



        num_da_samples = len(self.pupil_planes)
        num_rows = num_da_samples + 2
        num_cols = 6

        scale = 6
        plt.figure(figsize=[scale * 4, scale])

        # Replace this with a random crop from the validation set.
        # perfect_image_flipped = self.sess.run(self.perfect_image_flipped, feed_dict={self.inputs: inputs})
        perfect_image_flipped = self.sess.run(self.perfect_image_flipped, feed_dict={self.handle: self.valid_iterator_handle})
        perfect_image_flipped = perfect_image_flipped[0]
        perfect_image_spectrum = self.sess.run(self.perfect_image_spectrum, feed_dict={self.handle: self.valid_iterator_handle})
        perfect_image_spectrum = perfect_image_spectrum[0]

        for i, (pupil_plane,
                psf,
                mtf,
                distributed_aperture_image) in enumerate(zip(self.pupil_planes,
                                                             self.psfs,
                                                             self.mtfs,
                                                             self.distributed_aperture_images)):
            # (pupil_plane,
            #  psf,
            #  distributed_aperture_image) = self.sess.run([pupil_plane,
            #                                               psf,
            #                                               distributed_aperture_image],
            #                                              feed_dict={self.inputs: inputs})
            (pupil_plane,
             psf,
             mtf,
             distributed_aperture_image) = self.sess.run([pupil_plane,
                                                          psf,
                                                          mtf,
                                                          distributed_aperture_image], feed_dict={self.handle: self.valid_iterator_handle})

            # These are actually batches, so just take the first one.
            distributed_aperture_image = distributed_aperture_image[0]

            plot_index = (num_cols * i)
            # Ian's alternative plots

            plt.subplot(num_rows, num_cols, plot_index + 1)
            # Plot phase angle
            plt.imshow(np.angle(pupil_plane), cmap='twilight_shifted')
            plt.colorbar()
            # Overlay aperture mask
            plt.imshow(np.abs(pupil_plane), cmap='Greys', alpha=.2)

            plt.subplot(num_rows, num_cols, plot_index + 2)
            plt.imshow(np.log10(psf), cmap='inferno')
            plt.colorbar()

            plt.subplot(num_rows, num_cols, plot_index + 3)
            plt.imshow(np.log10(mtf), cmap='inferno')
            plt.colorbar()

            plt.subplot(num_rows, num_cols, plot_index + 4)
            plt.imshow(distributed_aperture_image, cmap='inferno')
            plt.colorbar()

            plt.subplot(num_rows, num_cols, plot_index + 5)
            plt.imshow(perfect_image_flipped, cmap='inferno')
            plt.colorbar()

            plt.subplot(num_rows, num_cols, plot_index + 6)
            plt.imshow(np.log10(np.abs(perfect_image_spectrum)), cmap='inferno')
            plt.colorbar()

        # recovered_image = self.sess.run([self.recovered_image],
        #                                 feed_dict={self.inputs: inputs})
        recovered_image = self.sess.run([self.recovered_image], feed_dict={self.handle: self.valid_iterator_handle})

        # This is a batch, so just take the first one.
        recovered_image = recovered_image[0]

        recovered_image = np.squeeze(recovered_image)

        plot_index = num_cols * (num_rows - 2)
        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 1))
        # Plot phase angle
        plt.colorbar()
        # Overlay aperture mask
        plt.imshow(recovered_image, cmap='inferno')

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 2))
        plt.imshow(np.log(np.abs(np.fft.fft2(recovered_image))), cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 3))
        plt.imshow(np.log(np.fft.fftshift(np.abs(np.fft.fft2(recovered_image)))), cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 4))
        plt.imshow(np.log(np.fft.fftshift(np.abs(np.fft.fft2(recovered_image)))), cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 5))
        plt.imshow(np.log(np.abs(np.fft.fft2(np.fft.fftshift(np.abs(np.fft.fft2(recovered_image)))))), cmap='inferno')
        plt.colorbar()

        plt.subplot(num_rows, num_cols, plot_index + 6)
        plt.imshow(np.log10(np.abs(perfect_image_spectrum)), cmap='inferno')
        plt.colorbar()
        # (monolithic_pupil_plane,
        #  monolithic_psf,
        #  monolithic_aperture_image,
        #  perfect_image_flipped) = self.sess.run([self.monolithic_pupil_plane,
        #                                          self.monolithic_psf,
        #                                          self.monolithic_aperture_image,
        #                                          self.perfect_image_flipped],
        #                                          feed_dict={self.inputs: inputs})
        (monolithic_pupil_plane,
         monolithic_psf,
         monolithic_mtf,
         monolithic_aperture_image) = self.sess.run([self.monolithic_pupil_plane,
                                                     self.monolithic_psf,
                                                     self.monolithic_mtf,
                                                     self.monolithic_aperture_image], feed_dict={self.handle: self.valid_iterator_handle})


        # These are actually batches, so just take the first element.
        monolithic_pupil_plane = monolithic_pupil_plane
        monolithic_psf = monolithic_psf
        monolithic_aperture_image = monolithic_aperture_image[0]
        perfect_image_flipped = np.abs(perfect_image_flipped)

        plot_index = num_cols * (num_rows - 1)
        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 1))
        # Plot phase angle
        plt.imshow(np.angle(monolithic_pupil_plane), cmap='twilight_shifted')
        plt.colorbar()
        # Overlay aperture mask
        plt.imshow(np.abs(monolithic_pupil_plane), cmap='Greys', alpha=.2)

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 2))
        plt.imshow(np.log10(monolithic_psf), cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 3))
        plt.imshow(np.log10(monolithic_mtf), cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 4))
        plt.imshow(monolithic_aperture_image, cmap='inferno')
        plt.colorbar()

        plt.subplot(plt.subplot(num_rows, num_cols, plot_index + 5))
        plt.imshow(perfect_image_flipped, cmap='inferno')
        plt.colorbar()

        plt.subplot(num_rows, num_cols, plot_index + 6)
        plt.imshow(np.log10(np.abs(perfect_image_spectrum)), cmap='inferno')
        plt.colorbar()

        # fig = Figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.gca()
        #
        # ax.text(0.0, 0.0, "Test", fontsize=45)
        # ax.axis('off')
        #
        # canvas.draw()  # draw the canvas, cache the renderer
        #
        # # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        #
        # width, height = fig.get_size_inches() * fig.get_dpi()
        #
        # # image = tf.constant(np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(np.int(height), np.int(width), 3)).transpose((2, 0, 1))
        #
        # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((np.int(height), np.int(width), 3))
        # image = np.expand_dims(image, axis=0)
        #
        # with self.writer.as_default():
        #     #tf.compat.v1.summary.
        #     tf.summary.image("overview_image",
        #                      image,
        #                      max_outputs=1,
        #                      description=None
        #     )

        if logdir:
            run_id = step
            fig_path = os.path.join(logdir, str(run_id) + '.png')
            plt.gcf().set_dpi(1200)
            plt.savefig(fig_path)
            plt.close()

        else:

            fig_path = os.path.join('./', 'tmp.png')
            plt.savefig(fig_path)
            plt.close()

    # def train(self, inputs):
    #     return self.sess.run([self.output_images, self.optimize], feed_dict={
    #             self.inputs: inputs
    #     })

    def train(self):
        return self.sess.run([self.loss, self.image_mse, self.optimize], feed_dict={self.handle: self.train_iterator_handle})

    def validate(self):
        return self.sess.run([self.loss, self.image_mse], feed_dict={self.handle: self.valid_iterator_handle})

    # def get_loss(self, inputs):
    #     return self.sess.run([self.loss], feed_dict={
    #             self.inputs: inputs
    #     })
    #
    # def get_metric(self, inputs):
    #     return self.sess.run([self.distributed_aperture_image_cosine_similarity], feed_dict={
    #             self.inputs: inputs
    #     })


def train(sess,
          dasie_model,
          train_dataset,
          valid_dataset,
          spatial_quantization=256,
          image_path='sample_image.png',
          num_steps=1,
          plot_periodicity=1,
          writer=None,
          step_update=None,
          all_summary_ops=None,
          writer_flush=None,
          logdir=None,
          save_plot=False):
    """
    This function realizes a DASIE model optimization loop.

    :param sess: a tf.Session, in which the main graph is built.
    :param env: a gym.Environment to train the models against.
    :param flags: a namespace containing user-specified flags.
    :param actor: an ActorModel built in sess against env.
    :param critic: a CriticModel built in sess against env.
    :param actor_noise: a callable which returns action-sized arrays.
    :return: None
    """

    # Begin training by initializing the graph.
    sess.run(tf.compat.v1.global_variables_initializer())

    # TODO: replace this with a dataset interface.
    # Read the target image only once.
    # perfect_image = plt.imread(image_path)

    # Normalize the image, just in case. You never know.
    # perfect_image = perfect_image / np.max(perfect_image)

    # Initialize with required Datasets

    # Enter the main training loop.
    for i in range(num_steps):

        print("Beginning Epoch %d" % i)
        tf.summary.experimental.set_step(i)

        # If requested, plot the model status.
        if save_plot:
            if (i % plot_periodicity) == 0:

                print("Plotting...")

                dasie_model.plot(logdir=logdir,
                                 step=i)

                print("Plotting completed.")


        print("Training...")
        try:


            # TODO: Add while true.
            # Execute one gradient update step.
            train_loss, train_image_mse, _ = dasie_model.train()

            print("Train Loss: %d" % train_loss)
            print("Train MSE: %d" % train_image_mse)

        except tf.errors.OutOfRangeError:

            print("Epoch %d Training Complete." % i)
            pass

        print("Validating...")
        try:

            # TODO: Add while true.
            # Execute one gradient update step.
            valid_loss, valid_image_mse = dasie_model.validate()

            print("Validation Loss: %d" % valid_loss)
            print("Validation MSE: %d" % valid_image_mse)

        except tf.errors.OutOfRangeError:

            print("Epoch %d Validation Complete." % i)
            pass

        # run_loss = dasie_model.get_loss(perfect_image)[0]
        # with writer.as_default():
        #     tf.summary.scalar("run_loss", run_loss, step=i)
        # print(run_loss)
        #
        # run_metric = dasie_model.get_metric(perfect_image)[0]
        # with writer.as_default():
        #     tf.summary.scalar("run_metric", run_metric, step=i)
        # print(run_metric)

        # Execute the summary writer ops to write their values.
        sess.run(all_summary_ops, feed_dict={dasie_model.handle: dasie_model.valid_iterator_handle})
        sess.run(step_update, feed_dict={dasie_model.handle: dasie_model.valid_iterator_handle})
        sess.run(writer_flush, feed_dict={dasie_model.handle: dasie_model.valid_iterator_handle})

def speedplus_parse_function(example_proto):
    """
    This is the first step of the generator/augmentation chain. Reading the
    raw file out of the TFRecord is fairly straight-forward, though does
    require some simple fixes. For instance, the number of bounding boxes
    needs to be padded to some upper bound so that the tensors are all of
    the same shape and can thus be batched.

    :param example_proto: Example from a TFRecord file
    :return: The raw image and padded bounding boxes corresponding to
    this TFRecord example.
    """
    # Define how to parse the example
    features = {
        "image_raw": tf.io.VarLenFeature(dtype=tf.string),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    # Parse the example
    features_parsed = tf.io.parse_single_example(
        serialized=example_proto, features=features
    )
    width = tf.cast(features_parsed["width"], tf.int32)
    height = tf.cast(features_parsed["height"], tf.int32)

    # filename = tf.cast(
    #     tf.sparse.to_dense(features_parsed["filename"], default_value=""),
    #     tf.string,
    # )

    image = tf.sparse.to_dense(
        features_parsed["image_raw"], default_value=""
    )
    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [width, height, 1])
    image = tf.cast(image, tf.float64)

    return image

def main(flags):

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(".", "logs", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Radius of the telescope in meters.
    # TODO: identify exact coordinates of measure.
    radius = 1.25

    # TODO: Document how distance to the target is quantified implicitly.
    # TODO: Document how extend of the target is quantified implicitly.

    # Completely BS alpha scaling factor again...
    # TODO: Set alpha based on new subaperture radius flag.
    # TODO: Add self-consistency check ensuring non-overlapping subapertures.
    alpha = np.pi * radius / flags.num_subapertures / 4

    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        print("\n\n\n\n\n\n\n\n\n Session Created \n\n\n\n\n\n\n")

        # Set all our seeds.
        np.random.seed(flags.random_seed)
        tf.compat.v1.set_random_seed(flags.random_seed)

        # Make summary management variables.
        step = tf.Variable(0, dtype=tf.int64)
        step_update = step.assign_add(1)
        tf.summary.experimental.set_step(step)
        writer = tf.summary.create_file_writer(save_dir)


        # TODO: Externalize this flag.
        dataset = "speedplus"
        # Different datasets may require different parsers, so we choose one.
        if dataset == "speedplus":
            parse_function = speedplus_parse_function
        else:
            # TODO: Throw exception when not supplied.
            replacemewithanexception

        # TODO: Force DatasetGenerators to fix the size of the batches.
        # We create a tf.data.Dataset object wrapping the train dataset here.

        # Check the dims of the input and manually crop.

        crop_size = None
        if flags.crop:
            crop_size = flags.spatial_quantization

        train_dataset = DatasetGenerator(flags.train_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=False,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=2,
                                         buffer=32,
                                         encoding_function=None,
                                         cache_dataset_memory=False,
                                         cache_dataset_file=False,
                                         cache_path="")

        # TODO: Depricate once validation is working.
        # train_dataset_iter = train_dataset.get_iterator()
        # train_dataset_batch = train_dataset_iter.get_next()

        # Manual debug here, to diagnose data problems.
        # plot_data = False
        # if plot_data:
        #     train_dataset_batch = sess.run(train_dataset_batch)
        #     plt.subplot(141)
        #     plt.imshow(np.abs(np.fft.fft2(np.abs(train_dataset_batch[0]))))
        #     # plt.subplot(142)
        #     # plt.imshow(train_dataset_batch[1])
        #     # plt.subplot(143)
        #     # plt.imshow(train_dataset_batch[2])
        #     # plt.subplot(144)
        #     # plt.imshow(train_dataset_batch[3])
        #     print(np.min(train_dataset_batch[0]))
        #     print(np.max(train_dataset_batch[0]))
        #     plt.show()

        # We create a tf.data.Dataset object wrapping the valid dataset here.
        valid_dataset = DatasetGenerator(flags.valid_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=False,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=2,
                                         buffer=32,
                                         encoding_function=None,
                                         cache_dataset_memory=False,
                                         cache_dataset_file=False,
                                         cache_path="")
        # valid_dataset = None


        # Make a template dataset to be initialized with train or val data.
        # dataset_placeholder = tf.placeholder(tf.float32, [None, flags.spatial_quantization, flags.spatial_quantization, 1])
        # dataset = tf.data.Dataset.from_tensor_slices((dataset_placeholder))
        # dataset = dataset.batch(flags.batch_size)
        # iterator = dataset.make_initializable_iterator()
        # dataset_batch = iterator.get_next()

        # Build a DA model. Inputs: n p/t/t tensors. Output: n image tensors.
        # dasie_model = DASIEModel(sess,
        #                          inputs=train_dataset_batch,
        #                          num_exposures=flags.num_exposures,
        #                          spatial_quantization=flags.spatial_quantization,
        #                          learning_rate=flags.learning_rate,
        #                          writer=writer,
        #                          alpha=alpha)
        dasie_model = DASIEModel(sess,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset,
                                 num_exposures=flags.num_exposures,
                                 spatial_quantization=flags.spatial_quantization,
                                 learning_rate=flags.learning_rate,
                                 writer=writer,
                                 alpha=alpha)

        # Merge all the summaries from the graphs, flush and init the nodes.
        all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        writer_flush = writer.flush()
        sess.run([writer.init(), step.initializer])

        # Optimize the DASIE model parameters.
        train(sess,
              dasie_model,
              train_dataset,
              valid_dataset,
              spatial_quantization=flags.spatial_quantization,
              image_path=flags.image_path,
              num_steps=flags.num_steps,
              plot_periodicity=flags.plot_periodicity,
              writer=writer,
              step_update=step_update,
              all_summary_ops=all_summary_ops,
              writer_flush=writer_flush,
              logdir=save_dir,
              save_plot=flags.save_plot)



if __name__ == '__main__':

    # TODO: I need to enable a test of negligable, random, and learned articulations to measure validation set reconstructions.

    parser = argparse.ArgumentParser(
        description='provide arguments for training.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')


    parser.add_argument('--logdir',
                        type=str,
                        default=".\\logs\\",
                        help='The directory to which summaries are written.')

    parser.add_argument('--run_name',
                        type=str,
                        default=datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                        help='The name of this run')

    parser.add_argument('--num_steps',
                        type=int,
                        default=4096,
                        help='The number of optimization steps to perform..')

    parser.add_argument('--image_path',
                        type=str,
                        default="sample_image.png",
                        help='The path to the location of the image.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=np.random.randint(0, 2048),
                        help='A random seed for repeatability.')

    parser.add_argument('--plot_periodicity',
                        type=int,
                        default=64,
                        help='Number of epochs to wait before plotting.')

    parser.add_argument('--num_subapertures',
                        type=int,
                        default=15,
                        help='Number of DASIE subapertures.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='The size of the optimizer step.')

    parser.add_argument('--spatial_quantization',
                        type=int,
                        default=256,
                        help='Quantization of all images.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Number of perfect images per batch.')

    parser.add_argument('--num_exposures',
                        type=int,
                        default=1,
                        help='The number of sequential frames to model.')

    parser.add_argument("--show_plot", action='store_true',
                        default=False,
                        help="Show the plot?")

    parser.add_argument("--save_plot", action='store_true',
                        default=False,
                        help='Save the plot?')

    parser.add_argument("--crop", action='store_true',
                        default=False,
                        help='If true, crop images to spatial_quantization.')

    # parser.add_argument('--train_data_dir', type=str,
    #                     default="C:\\Users\\justin.fletcher\\research\\speedplus_tfrecords\\train",
    #                     help='Path to the train data TFRecords directory.')
    #
    # parser.add_argument('--valid_data_dir', type=str,
    #                     default="C:\\Users\\justin.fletcher\\research\\speedplus_tfrecords\\valid",
    #                     help='Path to the val data TFRecords directory.')

    parser.add_argument('--train_data_dir', type=str,
                        default="C:\\Users\\justin.fletcher\\research\\onesat_example_tfrecords\\train",
                        help='Path to the train data TFRecords directory.')

    parser.add_argument('--valid_data_dir', type=str,
                        default="C:\\Users\\justin.fletcher\\research\\onesat_example_tfrecords\\valid",
                        help='Path to the val data TFRecords directory.')




    parser.add_argument('--distributed_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--filter_psf_extent',
                        type=float,
                        default=2.0,
                        help='Angular extent of simulated PSF (arcsec)')

    parser.add_argument('--monolithic_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--distributed_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--monolithic_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--aperture_diameter_num',
                        type=int,
                        default=64,
                        help='Number of linspaced aperture values to simulate')


    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)