import tensorflow as tf


def model_fn(features, labels, mode, params):
    
    image = features

    if isinstance(features, dict):
        image = features['image']
    
    isTraining = tf.constant(False)
    if mode == tf.estimator.ModeKeys.TRAIN:
        isTraining = tf.constant(True)

    #layer 1
    conv1_2 = tf.layers.conv2d(features, 64, 3, 1, 'same', name='conv1_2')
    relu1_2 = tf.nn.relu(conv1_2, name='relu1_2')
    conv1_2norm = tf.layers.batch_normalization(relu1_2, axis=3, name='conv1_2norm', training=isTraining)
    conv1_2norm_ss = tf.layers.conv2d(conv1_2norm, 64, 1, 2, 'same', name='conv1_2norm_ss')

    #layer 2
    conv2_1 = tf.layers.conv2d(conv1_2norm_ss, 128, 3, 1, 'same', name='conv2_1')
    relu2_1 = tf.nn.relu(conv2_1, name='relu2_1')
    conv2_2 = tf.layers.conv2d(relu2_1, 128, 3, 1, 'same', name='conv2_2')
    relu2_2 = tf.nn.relu(conv2_2, name='relu2_2')
    conv2_2norm = tf.layers.batch_normalization(relu2_2, axis=3, name='conv2_2norm', training=isTraining)
    conv2_2norm_ss = tf.layers.conv2d(conv2_2norm, 128, 1, 2, 'same', name='conv2_2norm_ss')

    #layer 3 
    conv3_1 = tf.layers.conv2d(conv2_2norm_ss, 256, 3, 1, 'same', name='conv3_1')
    relu3_1 = tf.nn.relu(conv3_1, name='relu3_1')
    conv3_2 = tf.layers.conv2d(relu3_1, 256, 3, 1, 'same', name='conv3_2')
    relu3_2 = tf.nn.relu(conv3_2, name='relu3_2')
    conv3_3 = tf.layers.conv2d(relu3_2, 256, 3, 1, 'same', name='conv3_3')
    relu3_3 = tf.nn.relu(conv3_3, name='relu3_3')
    conv3_3norm = tf.layers.batch_normalization(relu3_3, axis=3, name='conv3_3norm', training=isTraining)
    conv3_3norm_ss = tf.layers.conv2d(conv3_3norm, 256, 1, 2, 'same', name='conv3_3norm_ss')

    #layer 4
    conv4_1 = tf.layers.conv2d(conv3_3norm_ss, 512, 3, 1, 'same', name='conv4_1')
    relu4_1 = tf.nn.relu(conv4_1, name='relu4_1')
    conv4_2 = tf.layers.conv2d(relu4_1, 512, 3, 1, 'same', name='conv4_2')
    relu4_2 = tf.nn.relu(conv4_2, name='relu4_2')
    conv4_3 = tf.layers.conv2d(relu4_2, 512, 3, 1, 'same', name='conv4_3')
    relu4_3 = tf.nn.relu(conv4_3, name='relu4_3')
    conv4_3norm = tf.layers.batch_normalization(relu4_3, axis=3, name='conv4_3norm', training=isTraining)

    #layer 5
    conv5_1 = tf.layers.conv2d(conv4_3norm, 512, 3, 1, 'same', dilation_rate=2, name='conv5_1')
    relu5_1 = tf.nn.relu(conv5_1, name='relu5_1')
    conv5_2 = tf.layers.conv2d(relu5_1, 512, 3, 1, 'same', dilation_rate=2, name='conv5_2')
    relu5_2 = tf.nn.relu(conv5_2, name='relu5_2')
    conv5_3 = tf.layers.conv2d(relu5_2, 512, 3, 1, 'same', dilation_rate=2, name='conv5_3')
    relu5_3 = tf.nn.relu(conv5_3, name='relu5_3')
    conv5_3norm = tf.layers.batch_normalization(relu5_3, axis=3, name='conv5_3norm', training=isTraining)

    #layer 6
    conv6_1 = tf.layers.conv2d(conv5_3norm, 512, 3, 1, 'same', dilation_rate=2, name='conv6_1')
    relu6_1 = tf.nn.relu(conv6_1, name='relu6_1')
    conv6_2 = tf.layers.conv2d(relu6_1, 512, 3, 1, 'same', dilation_rate=2, name='conv6_2')
    relu6_2 = tf.nn.relu(conv6_2, name='relu6_2')
    conv6_3 = tf.layers.conv2d(relu6_2, 512, 3, 1, 'same', dilation_rate=2, name='conv6_3')
    relu6_3 = tf.nn.relu(conv6_3, name='relu6_3')
    conv6_3norm = tf.layers.batch_normalization(relu6_3, axis=3, name='conv6_3norm', training=isTraining)

    #layer 7
    conv7_1 = tf.layers.conv2d(conv6_3norm, 512, 3, 1, 'same', name='conv7_1')
    relu7_1 = tf.nn.relu(conv7_1, name='relu7_1')
    conv7_2 = tf.layers.conv2d(relu7_1, 512, 3, 1, 'same', name='conv7_2')
    relu7_2 = tf.nn.relu(conv7_2, name='relu7_2')
    conv7_3 = tf.layers.conv2d(relu7_2, 512, 3, 1, 'same', name='conv7_3')
    relu7_3 = tf.nn.relu(conv7_3, name='relu7_3')
    conv7_3norm = tf.layers.batch_normalization(relu7_3, axis=3, name='conv7_3norm', training=isTraining)

    #layer 8
    conv8_1 = tf.layers.conv2d_transpose(conv7_3norm, 256, 4, 2, 'same', name='conv8_1')
    conv3_3short = tf.layers.conv2d(conv3_3norm, 256, 3, 1, 'same', name='conv3_3short')
    relu8_1_comb = tf.nn.relu(conv8_1 + conv3_3short, name='relu8_1_comb')
    conv8_2 = tf.layers.conv2d(relu8_1_comb, 256, 3, 1, 'same', name='conv8_2')
    relu8_2 = tf.nn.relu(conv8_2, name='relu8_2')
    conv8_3 = tf.layers.conv2d(relu8_2, 256, 3, 1, 'same', name='conv8_3')
    relu8_3 = tf.nn.relu(conv3_3, name='relu8_3')
    conv8_3norm = tf.layers.batch_normalization(relu8_3, axis=3, name='conv8_3norm', training=isTraining)

    #layer 9
    conv9_1 = tf.layers.conv2d_transpose(conv8_3norm, 128, 4, 2, 'same', name='conv9_1')
    conv2_2short = tf.layers.conv2d(conv2_2norm, 128, 3, 1, 'same', name='conv2_2short')
    relu9_1_comb = tf.nn.relu(conv9_1 + conv2_2short, name='relu9_1_comb')
    conv9_2 = tf.layers.conv2d(relu9_1_comb, 128, 4, 1, 'same', name='conv9_2')
    relu9_2 = tf.nn.relu(conv9_2, name='relu9_2')
    conv9_2norm = tf.layers.batch_normalization(relu9_2, axis=3, name='conv9_2norm', training=isTraining)

    #layer 10
    conv10_1 = tf.layers.conv2d_transpose(conv9_2norm, 128, 4, 2, 'same', name='conv10_1')
    conv1_2short = tf.layers.conv2d(conv1_2norm, 128, 3, 1, 'same', name='conv1_2short')
    relu_10_1_comb = tf.nn.relu(conv1_2short, name='relu_10_1_comb')
    conv10_2 = tf.layers.conv2d(relu_10_1_comb, 128, 3, 1, 'same', name='conv10_2')
    relu10_2 = tf.nn.relu(conv10_2, name='relu10_2')


    #unary prediction
    conv10_ab = tf.layers.conv2d(conv10_2, 2, 1, 1, 'same', name='conv10_ab')
    pred_ab_1 = tf.nn.tanh(conv10_ab, name='conv10_ab') * tf.constant(100.0)


    loss = tf.losses.huber_loss(labels, pred_ab_1)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    return tf.estimator.EstimatorSpec(mode, 
                                      loss=loss,
                                      train_op=train,
                                      eval_metric_ops={
                                        'accuracy':
                                            tf.metrics.accuracy(
                                                labels=labels,
                                                predictions=pred_ab_1
                                            )
                })

