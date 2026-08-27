"""Implementation of attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # This error is caused by the version (issue) and already exists in NAA; it does not affect the operation.
import time
import utils_IG2_xr_5
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
from tqdm import tqdm
import random

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'resnet_v2_50', 'The Model used to generate adv.')

tf.flags.DEFINE_string('attack_method', 'Ours', 'The name of attack method.')  # if PIDI  --attack_method  OursPIDI

tf.flags.DEFINE_string('layer_name','resnet_v2_50/block1/unit_1/bottleneck_v2/add','The layer to be attacked.')

tf.flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './COPIA_adv/resnet_v2_50_xr_5_lr_0.025_scale_0.2_grad_hybrid_COPIA/', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float('lr', 0.025, 'learning rate for IG2.')


tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float('alpha', 1.6, 'Step size.')

tf.flags.DEFINE_integer('batch_size', 5, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

tf.flags.DEFINE_float('scale', 0.2, 'noise deviation')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 299, 'size of each input images.')

tf.flags.DEFINE_integer('image_resize', 331, 'size of each diverse images.')

tf.flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')

"""parameter for PIM"""
tf.flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')

tf.flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')

tf.flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

tf.flags.DEFINE_float('ens', 30.0, 'Number of aggregated n.')


FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []

    operations = tf.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape

    return opt_operations,shape

def get_opt_layers_2(layer_name):
    opt_operations = []
    opt_operations1 = []
    opt_operations2 = []
    opt_operations3 = []
    opt_operations4 = []
    opt_operations5 = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.get_default_graph().get_operations()  #  get the layer
    for op in operations:
        if 'resnet_v2_50/block4/unit_3/bottleneck_v2/add' == op.name:
            print(op.name, op.outputs)
            opt_operations.append(op.outputs[0])
            shape = op.outputs[0][:FLAGS.batch_size].shape
        elif 'resnet_v2_50/block4/unit_1/bottleneck_v2/add' == op.name:
            print(op.name, op.outputs)
            opt_operations1.append(op.outputs[0])
            shape1 = op.outputs[0][:FLAGS.batch_size].shape
        elif 'resnet_v2_50/block3/unit_6/bottleneck_v2/add' == op.name:
            print(op.name, op.outputs)
            opt_operations2.append(op.outputs[0])
            shape2 = op.outputs[0][:FLAGS.batch_size].shape
        elif 'resnet_v2_50/block2/unit_3/bottleneck_v2/add' == op.name:
            print(op.name, op.outputs)
            opt_operations3.append(op.outputs[0])
            shape3 = op.outputs[0][:FLAGS.batch_size].shape
        elif 'resnet_v2_50/block1/unit_1/bottleneck_v2/add' == op.name:
            print(op.name, op.outputs)
            opt_operations4.append(op.outputs[0])
            shape4 = op.outputs[0][:FLAGS.batch_size].shape
        elif 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv2/Relu' == op.name:
            print(op.name, op.outputs)
            opt_operations5.append(op.outputs[0])
            shape5 = op.outputs[0][:FLAGS.batch_size].shape
    return opt_operations, opt_operations1, opt_operations2, opt_operations3, opt_operations4, opt_operations5, shape, shape1, shape2, shape3, shape4, shape5


#  Obtain the MSE loss between the counterfactual samples and the feature layer of the input x.
def get_MSE_loss(opt_operations_xr, xr_feature):  #
    for layer in opt_operations_xr:  #
        adv_expression_x = layer[FLAGS.batch_size:]  #
        squared_difference = tf.square(adv_expression_x - xr_feature)
        MSE_loss = tf.reduce_mean(squared_difference)
        return MSE_loss

def get_NAA_loss(opt_operations,weights,base_feature):
    loss = 0
    gamma = 1.0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        attribution = (adv_tensor-base_feature)*weights
        #attribution = adv_tensor*weights
        blank = tf.zeros_like(attribution)
        positive = tf.where(attribution >= 0, attribution, blank)
        negative = tf.where(attribution < 0, attribution, blank)
        ## Transformation: Linear transformation performs the best
        positive = positive
        negative = negative
        ##
        balance_attribution = positive + gamma*negative  #
        loss += tf.reduce_sum(balance_attribution) / tf.cast(tf.size(layer), tf.float32)
        
    loss = loss / len(opt_operations)
    return loss

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2

def tfnormalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=tf.reduce_sum(tf.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = tf.reduce_sum(tf.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/tf.sqrt(square)
    return nor_grad


def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    return x

def input_diversity(input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret=tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_size, FLAGS.image_size],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

P_kern, kern_size = project_kern(FLAGS.Pkern_size)


def main(_):
    
    random.seed(2013)

    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    image_preprocessing_fn = utils_IG2_xr_5.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils_IG2_xr_5.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils_IG2_xr_5.checkpoint_paths[FLAGS.model_name]
    layer_name = FLAGS.layer_name

    with tf.Graph().as_default():
        # Prepare graph
        ori_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)

        ori_input_xr_0 = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input_xr_0 = tf.placeholder(tf.float32, shape=batch_shape)

        ori_input_xr_1 = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input_xr_1 = tf.placeholder(tf.float32, shape=batch_shape)

        ori_input_xr_2 = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input_xr_2 = tf.placeholder(tf.float32, shape=batch_shape)

        ori_input_xr_3 = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input_xr_3 = tf.placeholder(tf.float32, shape=batch_shape)

        ori_input_xr_4 = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input_xr_4 = tf.placeholder(tf.float32, shape=batch_shape)

        num_classes = 1000 + utils_IG2_xr_5.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*2,num_classes])
        accumulated_grad_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)
        amplification_ph = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        network_fn = utils_IG2_xr_5.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input],axis=0)
        x_xr_0 = tf.concat([ori_input_xr_0, adv_input_xr_0], axis=0)
        x_xr_1 = tf.concat([ori_input_xr_1, adv_input_xr_1], axis=0)
        x_xr_2 = tf.concat([ori_input_xr_2, adv_input_xr_2], axis=0)
        x_xr_3 = tf.concat([ori_input_xr_3, adv_input_xr_3], axis=0)
        x_xr_4 = tf.concat([ori_input_xr_4, adv_input_xr_4], axis=0)

        # whether using DIM or not
        if 'DI' in FLAGS.attack_method:
            logits, end_points = network_fn(input_diversity(x))
            # 获取四张参考图片的输出
            logits_xr_0, end_points_xr_0 = network_fn(input_diversity(x_xr_0))
            logits_xr_1, end_points_xr_1 = network_fn(input_diversity(x_xr_1))
            logits_xr_2, end_points_xr_2 = network_fn(input_diversity(x_xr_2))
            logits_xr_3, end_points_xr_3 = network_fn(input_diversity(x_xr_3))
            logits_xr_4, end_points_xr_4 = network_fn(input_diversity(x_xr_4))



        else:
            logits, end_points = network_fn(x)
            logits_xr_0, end_points_xr_0 = network_fn(x_xr_0)
            logits_xr_1, end_points_xr_1 = network_fn(x_xr_1)
            logits_xr_2, end_points_xr_2 = network_fn(x_xr_2)
            logits_xr_3, end_points_xr_3 = network_fn(x_xr_3)
            logits_xr_4, end_points_xr_4 = network_fn(x_xr_4)
        problity=tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)

        entropy_loss = tf.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])
        opt_operations, opt_operations1, opt_operations2, opt_operations3, opt_operations4, opt_operations5, shape, shape1, shape2, shape3, shape4, shape5 = get_opt_layers_2(
            layer_name)
        # opt_operations, shape = get_opt_layers(layer_name)
        # xr_feature = tf.placeholder(dtype=tf.float32, shape=shape1)  #

        xr_logits = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])   #  xr 0
        xr_logits_1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])  # xr 1
        xr_logits_2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])  # xr 2
        xr_logits_3 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])  # xr 3
        xr_logits_4 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * 2, num_classes])  # xr 4

        # xr 0
        logits_MSE_loss = tf.losses.mean_squared_error(xr_logits[FLAGS.batch_size:], logits_xr_0[FLAGS.batch_size:])
        xr_path_gradient = tf.gradients(logits_MSE_loss, adv_input_xr_0)[0]
        # xr 1
        logits_MSE_loss_xr_1 = tf.losses.mean_squared_error(xr_logits_1[FLAGS.batch_size:], logits_xr_1[FLAGS.batch_size:])
        xr_path_gradient_xr_1 = tf.gradients(logits_MSE_loss_xr_1, adv_input_xr_1)[0]
        # xr 2
        logits_MSE_loss_xr_2 = tf.losses.mean_squared_error(xr_logits_2[FLAGS.batch_size:], logits_xr_2[FLAGS.batch_size:])
        xr_path_gradient_xr_2 = tf.gradients(logits_MSE_loss_xr_2, adv_input_xr_2)[0]
        # xr 3
        logits_MSE_loss_xr_3 = tf.losses.mean_squared_error(xr_logits_3[FLAGS.batch_size:], logits_xr_3[FLAGS.batch_size:])
        xr_path_gradient_xr_3 = tf.gradients(logits_MSE_loss_xr_3, adv_input_xr_3)[0]
        # xr 4
        logits_MSE_loss_xr_4 = tf.losses.mean_squared_error(xr_logits_4[FLAGS.batch_size:],
                                                            logits_xr_4[FLAGS.batch_size:])
        xr_path_gradient_xr_4 = tf.gradients(logits_MSE_loss_xr_4, adv_input_xr_4)[0]



        # weights_ph = tf.placeholder(dtype=tf.float32, shape=shape)
        # base_feature = tf.placeholder(dtype=tf.float32, shape=shape)

        weights_ph = tf.placeholder(dtype=tf.float32, shape=shape)
        base_feature = tf.placeholder(dtype=tf.float32, shape=shape)
        weights_ph1 = tf.placeholder(dtype=tf.float32, shape=shape1)
        base_feature1 = tf.placeholder(dtype=tf.float32, shape=shape1)
        weights_ph2 = tf.placeholder(dtype=tf.float32, shape=shape2)
        base_feature2 = tf.placeholder(dtype=tf.float32, shape=shape2)
        weights_ph3 = tf.placeholder(dtype=tf.float32, shape=shape3)
        base_feature3 = tf.placeholder(dtype=tf.float32, shape=shape3)
        weights_ph4 = tf.placeholder(dtype=tf.float32, shape=shape4)
        base_feature4 = tf.placeholder(dtype=tf.float32, shape=shape4)


        weights_tensor = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations[0])[0]  #
        weights_tensor1 = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations1[0])[0]
        weights_tensor2 = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations2[0])[0]
        weights_tensor3 = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations3[0])[0]
        weights_tensor4 = tf.gradients(tf.nn.softmax(logits) * label_ph, opt_operations4[0])[0]

        # loss, neuronal_importance = get_NAA_loss(opt_operations,weights_ph,base_feature)
        quanzhi = 1

        loss_0 = get_NAA_loss(opt_operations, weights_ph, base_feature)
        weights_tensor1_fromnext = tf.gradients(loss_0, opt_operations1[0])[0]
        loss_1 = get_NAA_loss(opt_operations1, quanzhi * tfnormalize(
            weights_tensor1_fromnext[FLAGS.batch_size:]) + weights_ph1, base_feature1)
        weights_tensor2_fromnext = tf.gradients(loss_1, opt_operations2[0])[0]
        loss_2 = get_NAA_loss(opt_operations2,
                              quanzhi * tfnormalize(weights_tensor2_fromnext[FLAGS.batch_size:]) + weights_ph2,
                              base_feature2)
        weights_tensorx1_fromnext = tf.gradients(loss_2, opt_operations3[0])[0]
        loss_3 = get_NAA_loss(opt_operations3,
                              quanzhi * tfnormalize(weights_tensorx1_fromnext[FLAGS.batch_size:]) + weights_ph3,
                              base_feature3)
        weights_tensorx2_fromnext = tf.gradients(loss_3, opt_operations4[0])[0]
        loss_4 = get_NAA_loss(opt_operations4,
                              quanzhi * tfnormalize(weights_tensorx2_fromnext[FLAGS.batch_size:]) + weights_ph4,
                              base_feature4)
        # weights_tensorx3_fromnext = tf.gradients(lossx2, opt_operations5[0])[0]
        # lossx3 = get_NAA_loss(opt_operations5,
        #                       quanzhi * tfnormalize(weights_tensorx3_fromnext[FLAGS.batch_size:]) + weights_ph5, base_feature5)
        loss = loss_4

        gradient=tf.gradients(loss,adv_input)[0]

        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph

        # the default optimization process with momentum
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * accumulated_grad_ph + noise

        # whether using PIM or not
        if 'PI' in FLAGS.attack_method:
            # amplification factor
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta

            # Project cut noise
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))

            amplification_update += projection

            adv_input_update = adv_input_update + alpha_beta * tf.sign(noise) + projection
        else:
            adv_input_update = adv_input_update + alpha * tf.sign(noise)


        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            count=0
            k = 1
            for images, names, labels, images_xr, images_xr_1, images_xr_2, images_xr_3, images_xr_4 in utils_IG2_xr_5.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size, load_num=1000):

                start_time = time.time()
                count += FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)
                images_tmp = image_preprocessing_fn(np.copy(images))
                images_xr_tmp = image_preprocessing_fn(np.copy(images_xr))  # xr 0
                images_xr_tmp_1 = image_preprocessing_fn(np.copy(images_xr_1))  # xr 1
                images_xr_tmp_2 = image_preprocessing_fn(np.copy(images_xr_2))  # xr 2
                images_xr_tmp_3 = image_preprocessing_fn(np.copy(images_xr_3))  # xr 3
                images_xr_tmp_4 = image_preprocessing_fn(np.copy(images_xr_4))  # xr 4
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1
                # obtain true label
                labels = to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                images_adv = images
                images_adv = image_preprocessing_fn(np.copy(images_adv))
                grad_np = np.zeros(shape=batch_shape)
                amplification_np = np.zeros(shape=batch_shape)
                weight_np = np.zeros(shape=shape)
                # weight_np = np.zeros(shape=shape)
                weight_np1 = np.zeros(shape=shape1)
                weight_np2 = np.zeros(shape=shape2)
                weight_np3 = np.zeros(shape=shape3)
                weight_np4 = np.zeros(shape=shape4)  # the weight of five selected layers

                for i in tqdm(range(num_iter)):
                    if i==0:
                        if FLAGS.ens == 0:
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            w, feature = sess.run([weights_tensor, opt_operations[0]],
                                                  feed_dict={ori_input: images_tmp2, adv_input: images_tmp2,label_ph: labels})
                            weight_np =w[:FLAGS.batch_size]

                        # grad descent
                        images_tmp2 = image_preprocessing_fn(np.copy(images))  # direction 0
                        images_tmp2_1 = images_tmp2  # direction 1
                        images_tmp2_2 = images_tmp2  # direction 2
                        images_tmp2_3 = images_tmp2  # direction 3
                        images_tmp2_4 = images_tmp2  # direction 4

                        #  grad ascent
                        images_tmp2_up = image_preprocessing_fn(np.copy(images))
                        images_tmp2_1_up = images_tmp2
                        images_tmp2_2_up = images_tmp2
                        images_tmp2_3_up = images_tmp2
                        images_tmp2_4_up = images_tmp2

                        xr_logits_real, xr_logits_real_1, xr_logits_real_2,xr_logits_real_3, xr_logits_real_4 = sess.run([logits_xr_0, logits_xr_1,logits_xr_2,logits_xr_3, logits_xr_4],
                                                                                                       feed_dict={ori_input_xr_0:images_xr_tmp,
                                                                                                                  adv_input_xr_0:images_xr_tmp,
                                                                                                                  ori_input_xr_1: images_xr_tmp_1,
                                                                                                                  adv_input_xr_1: images_xr_tmp_1,
                                                                                                                  ori_input_xr_2: images_xr_tmp_2,
                                                                                                                  adv_input_xr_2: images_xr_tmp_2,
                                                                                                                  ori_input_xr_3: images_xr_tmp_3,
                                                                                                                  adv_input_xr_3: images_xr_tmp_3,
                                                                                                                  ori_input_xr_4: images_xr_tmp_4,
                                                                                                                  adv_input_xr_4: images_xr_tmp_4,
                                                                                                                  })


                        for l in range(int(FLAGS.ens)):


                            #  get MSE loss and grad
                            logits_MSE_loss_real, xr_path_gradient_real, logits_MSE_loss_real_1, xr_path_gradient_real_1, \
                            logits_MSE_loss_real_2, xr_path_gradient_real_2,logits_MSE_loss_real_3, xr_path_gradient_real_3, logits_MSE_loss_real_4, xr_path_gradient_real_4= \
                            sess.run([logits_MSE_loss, xr_path_gradient, logits_MSE_loss_xr_1, xr_path_gradient_xr_1, logits_MSE_loss_xr_2, xr_path_gradient_xr_2, \
                            logits_MSE_loss_xr_3, xr_path_gradient_xr_3,logits_MSE_loss_xr_4, xr_path_gradient_xr_4],feed_dict={ori_input_xr_0:images_tmp2, adv_input_xr_0:images_tmp2,
                                                                                                                xr_logits:xr_logits_real,
                                                                                                                ori_input_xr_1: images_tmp2_1,
                                                                                                                adv_input_xr_1: images_tmp2_1,
                                                                                                                xr_logits_1: xr_logits_real_1,
                                                                                                                ori_input_xr_2: images_tmp2_2,
                                                                                                                adv_input_xr_2: images_tmp2_2,
                                                                                                                xr_logits_2: xr_logits_real_2,
                                                                                                                ori_input_xr_3: images_tmp2_3,
                                                                                                                adv_input_xr_3: images_tmp2_3,
                                                                                                                xr_logits_3: xr_logits_real_3,
                                                                                                                ori_input_xr_4: images_tmp2_4,
                                                                                                                adv_input_xr_4: images_tmp2_4,
                                                                                                                xr_logits_4: xr_logits_real_4,
                                                                                                                })

                            print(f"grad descent:{logits_MSE_loss_real}, {logits_MSE_loss_real_1}, {logits_MSE_loss_real_2}, {logits_MSE_loss_real_3}, {logits_MSE_loss_real_4}")


                            #  add noise
                            act_temp = images_tmp2 + np.random.normal(size = images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_1 = images_tmp2_1 + np.random.normal(size = images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_2 = images_tmp2_2 + np.random.normal(size = images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_3 = images_tmp2_3 + np.random.normal(size = images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_4 = images_tmp2_4 + np.random.normal(size=images.shape, loc=0.0,
                                                                          scale=FLAGS.scale)
                            # direction 0
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp, adv_input: act_temp,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]


                            #  direction 1
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_1, adv_input: act_temp_1,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]

                            # 2
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_2, adv_input: act_temp_2,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]

                            # 3
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_3, adv_input: act_temp_3,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]

                            # 4
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_4, adv_input: act_temp_4,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]

                            #  grad descent next stage
                            images_tmp2 = images_tmp2 - FLAGS.lr * np.sign(xr_path_gradient_real)
                            images_tmp2 = inv_image_preprocessing_fn(images_tmp2)
                            images_tmp2 = np.clip(images_tmp2, 0, 255)
                            images_tmp2 = image_preprocessing_fn(images_tmp2)


                            images_tmp2_1 = images_tmp2_1 - FLAGS.lr * np.sign(xr_path_gradient_real_1)
                            images_tmp2_1 = inv_image_preprocessing_fn(images_tmp2_1)
                            images_tmp2_1 = np.clip(images_tmp2_1, 0, 255)
                            images_tmp2_1 = image_preprocessing_fn(images_tmp2_1)


                            images_tmp2_2 = images_tmp2_2 - FLAGS.lr * np.sign(xr_path_gradient_real_2)
                            images_tmp2_2 = inv_image_preprocessing_fn(images_tmp2_2)
                            images_tmp2_2 = np.clip(images_tmp2_2, 0, 255)
                            images_tmp2_2 = image_preprocessing_fn(images_tmp2_2)



                            images_tmp2_3 = images_tmp2_3 - FLAGS.lr * np.sign(xr_path_gradient_real_3)
                            images_tmp2_3 = inv_image_preprocessing_fn(images_tmp2_3)
                            images_tmp2_3 = np.clip(images_tmp2_3, 0, 255)
                            images_tmp2_3 = image_preprocessing_fn(images_tmp2_3)
                            # 4
                            images_tmp2_4 = images_tmp2_4 - FLAGS.lr * np.sign(xr_path_gradient_real_4)
                            images_tmp2_4 = inv_image_preprocessing_fn(images_tmp2_4)
                            images_tmp2_4 = np.clip(images_tmp2_4, 0, 255)
                            images_tmp2_4 = image_preprocessing_fn(images_tmp2_4)

                            #  grad ascent
                            logits_MSE_loss_real_up, xr_path_gradient_real_up, logits_MSE_loss_real_1_up, xr_path_gradient_real_1_up, \
                                logits_MSE_loss_real_2_up, xr_path_gradient_real_2_up, logits_MSE_loss_real_3_up, xr_path_gradient_real_3_up, logits_MSE_loss_real_4_up, xr_path_gradient_real_4_up = \
                                sess.run(
                                    [logits_MSE_loss, xr_path_gradient, logits_MSE_loss_xr_1, xr_path_gradient_xr_1,
                                     logits_MSE_loss_xr_2, xr_path_gradient_xr_2,
                                     logits_MSE_loss_xr_3, xr_path_gradient_xr_3, logits_MSE_loss_xr_4,
                                     xr_path_gradient_xr_4],
                                    feed_dict={ori_input_xr_0: images_tmp2_up, adv_input_xr_0: images_tmp2_up,
                                               xr_logits: xr_logits_real,
                                               ori_input_xr_1: images_tmp2_1_up,
                                               adv_input_xr_1: images_tmp2_1_up,
                                               xr_logits_1: xr_logits_real_1,
                                               ori_input_xr_2: images_tmp2_2_up,
                                               adv_input_xr_2: images_tmp2_2_up,
                                               xr_logits_2: xr_logits_real_2,
                                               ori_input_xr_3: images_tmp2_3_up,
                                               adv_input_xr_3: images_tmp2_3_up,
                                               xr_logits_3: xr_logits_real_3,
                                               ori_input_xr_4: images_tmp2_4_up,
                                               adv_input_xr_4: images_tmp2_4_up,
                                               xr_logits_4: xr_logits_real_4,
                                               })

                            print(
                                f"grad ascent:{logits_MSE_loss_real_up}, {logits_MSE_loss_real_1_up}, {logits_MSE_loss_real_2_up}, {logits_MSE_loss_real_3_up}, {logits_MSE_loss_real_4_up}")


                            # add noise
                            act_temp_up = images_tmp2_up + np.random.normal(size=images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_1_up = images_tmp2_1_up + np.random.normal(size=images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_2_up = images_tmp2_2_up + np.random.normal(size=images.shape, loc=0.0, scale=FLAGS.scale)
                            act_temp_3_up = images_tmp2_3_up + np.random.normal(size=images.shape, loc=0.0,
                                                                          scale=FLAGS.scale)  #
                            act_temp_4_up = images_tmp2_4_up + np.random.normal(size=images.shape, loc=0.0,
                                                                          scale=FLAGS.scale)  #
                            # grad ascent direction 0
                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_up, adv_input: act_temp_up,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]


                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_1_up, adv_input: act_temp_1_up,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]


                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_2_up, adv_input: act_temp_2_up,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]


                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_3_up, adv_input: act_temp_3_up,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]


                            w, w_1, w_2, w_3, w_4 = sess.run([weights_tensor, weights_tensor1, weights_tensor2,
                                                              weights_tensor3, weights_tensor4],
                                                             feed_dict={ori_input: act_temp_4_up, adv_input: act_temp_4_up,
                                                                        label_ph: labels})
                            weight_np = weight_np + w[FLAGS.batch_size:]
                            weight_np1 = weight_np1 + w_1[FLAGS.batch_size:]
                            weight_np2 = weight_np2 + w_2[FLAGS.batch_size:]
                            weight_np3 = weight_np3 + w_3[FLAGS.batch_size:]
                            weight_np4 = weight_np4 + w_4[FLAGS.batch_size:]

                            # grad ascent netx stage
                            images_tmp2_up = images_tmp2_up + FLAGS.lr * np.sign(xr_path_gradient_real_up)  #
                            images_tmp2_up = inv_image_preprocessing_fn(images_tmp2_up)  #
                            images_tmp2_up = np.clip(images_tmp2_up, 0, 255)
                            images_tmp2_up = image_preprocessing_fn(images_tmp2_up)  #

                            images_tmp2_1_up = images_tmp2_1_up + FLAGS.lr * np.sign(xr_path_gradient_real_1_up)  #
                            images_tmp2_1_up = inv_image_preprocessing_fn(images_tmp2_1_up)  #
                            images_tmp2_1_up = np.clip(images_tmp2_1_up, 0, 255)
                            images_tmp2_1_up = image_preprocessing_fn(images_tmp2_1_up)

                            images_tmp2_2_up = images_tmp2_2_up + FLAGS.lr * np.sign(xr_path_gradient_real_2_up)  #
                            images_tmp2_2_up = inv_image_preprocessing_fn(images_tmp2_2_up)  #
                            images_tmp2_2_up = np.clip(images_tmp2_2_up, 0, 255)
                            images_tmp2_2_up = image_preprocessing_fn(images_tmp2_2_up)

                            images_tmp2_3_up = images_tmp2_3_up + FLAGS.lr * np.sign(xr_path_gradient_real_3_up)  #
                            images_tmp2_3_up = inv_image_preprocessing_fn(images_tmp2_3_up)  #
                            images_tmp2_3_up = np.clip(images_tmp2_3_up, 0, 255)
                            images_tmp2_3_up = image_preprocessing_fn(images_tmp2_3_up)  #
                            # 4
                            images_tmp2_4_up = images_tmp2_4_up + FLAGS.lr * np.sign(xr_path_gradient_real_4_up)
                            images_tmp2_4_up = inv_image_preprocessing_fn(images_tmp2_4_up)  #
                            images_tmp2_4_up = np.clip(images_tmp2_4_up, 0, 255)
                            images_tmp2_4_up = image_preprocessing_fn(images_tmp2_4_up)  #




                        weight_np = -normalize(weight_np / 10, 2)   #  mean
                        weight_np1 = -normalize(weight_np1 / 10, 2)
                        weight_np2 = -normalize(weight_np2 / 10, 2)
                        weight_np3 = -normalize(weight_np3 / 10, 2)
                        weight_np4 = -normalize(weight_np4 / 10, 2)

                        # grad descent  xr 0
                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2, adv_input: images_tmp2, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size]
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size]
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size]
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size]
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size]

                        # grad descent xr 1

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_1, adv_input: images_tmp2_1, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        # grad descent xr 2

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_2, adv_input: images_tmp2_2, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        # grad descent xr 3

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_3, adv_input: images_tmp2_3, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        # grad descent xr 4

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_4, adv_input: images_tmp2_4, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        #  grad ascent xr 0
                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_up, adv_input: images_tmp2_up, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 将前面的所有都加起来
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        #  grad ascent xr 1

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_1_up, adv_input: images_tmp2_1_up, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        #  grad ascent xr 2

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_2_up, adv_input: images_tmp2_2_up, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        #  grad ascent xr 3

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_3_up, adv_input: images_tmp2_3_up, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4

                        #  grad ascent xr 4

                        feature_base_tmp, feature_base1_tmp, feature_base2_tmp, feature_base3_tmp, feature_base4_tmp = sess.run(
                            [opt_operations[0], opt_operations1[0],
                             opt_operations2[0], opt_operations3[0],
                             opt_operations4[0]],
                            feed_dict={ori_input: images_tmp2_4_up, adv_input: images_tmp2_4_up, label_ph: labels})

                        feature_base = feature_base_tmp[:FLAGS.batch_size] + feature_base  # 这里获取出来的直接就是特征了
                        feature_base1 = feature_base1_tmp[:FLAGS.batch_size] + feature_base1
                        feature_base2 = feature_base2_tmp[:FLAGS.batch_size] + feature_base2
                        feature_base3 = feature_base3_tmp[:FLAGS.batch_size] + feature_base3
                        feature_base4 = feature_base4_tmp[:FLAGS.batch_size] + feature_base4


                        feature_base = feature_base / 10  # mean
                        feature_base1 = feature_base1 / 10
                        feature_base2 = feature_base2 / 10
                        feature_base3 = feature_base3 / 10
                        feature_base4 = feature_base4 / 10
                    # optimize
                    images_adv, grad_np, amplification_np = sess.run([adv_input_update, noise, amplification_update],
                                                                     feed_dict={ori_input: images_tmp,
                                                                                adv_input: images_adv,
                                                                                weights_ph: weight_np,
                                                                                weights_ph1: weight_np1,
                                                                                weights_ph2: weight_np2,
                                                                                weights_ph3: weight_np3,
                                                                                weights_ph4: weight_np4,
                                                                                # weights_ph5: weight_np5,
                                                                                base_feature: feature_base,
                                                                                base_feature1: feature_base1,
                                                                                base_feature2: feature_base2,
                                                                                base_feature3: feature_base3,
                                                                                base_feature4: feature_base4,
                                                                                # base_feature5:feature_base5,
                                                                                label_ph: labels,
                                                                                accumulated_grad_ph: grad_np,
                                                                                amplification_ph: amplification_np})


                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)

                images_adv = inv_image_preprocessing_fn(images_adv)
                utils_IG2_xr_5.save_image(images_adv, names, FLAGS.output_dir)
                end_time = time.time()
                # 计算并打印运行时间
                elapsed_time = end_time - start_time
                print(f"running time:{elapsed_time:.6f} 秒")

if __name__ == '__main__':
    tf.app.run()
