
## traffic signs MobileNet v2 - ResNet

## petrusel



import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import cv2
import math

tf.set_random_seed(1)
np.random.seed(1)
tf.keras.backend.clear_session()

path = 'D:\\AI\\traffic_signs_recognizer'
training_file =   path+"\\train.p"
validation_file = path+"\\valid.p"
testing_file =    path+"\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
# X_train, Y_train = train['features'], train['labels']
# X_val, Y_val = valid['features'], valid['labels']
# X_test, Y_test = test['features'], test['labels']  
 

X_train, Y_train = train['features'], train['labels']
X_test, Y_test = valid['features'], valid['labels']
X_val, Y_val = test['features'], test['labels'] 


X_all = np.concatenate([X_train, X_val], axis=0) 
Y_all = np.concatenate([Y_train, Y_val], axis=0)

num_classes = 43
lr = 0.01 
epochs = 20
batch_size = 64

def shuffle(x, y):
    num = np.random.permutation(x.shape[0])
    x = x[num]
    y = y[num]
    return x, y

def crop(img):
    im_c = img[4:29, 4:29, :] 
    im_c = cv2.resize(im_c, (32,32))
    return im_c

def adjust_gamma(image, gamma):
    # gamma_val = [0.5, 1.5, 2.5]
    invGamma = 1.0 / gamma #np.random.choice(gamma_val)
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def rotation(image):
    rows= image.shape[0]
    cols = image.shape[1]
    # rotation
    angle = [15,-15]
    rotatie = cv2.getRotationMatrix2D((cols/2,rows/2),np.random.choice(angle),1)
    img_rot = cv2.warpAffine(image,rotatie,(cols,rows))
    # crop
    img_crop = crop(img_rot)
    return img_crop

X_train_aug, Y_train_aug = X_all, Y_all

X_aug_1 = [] ### stocheaza doar datele augmentate 
Y_aug_1 = []

for i in range(0, 43):
    class_records = np.where(Y_train_aug==i)[0].size
    # max_records = 2500 # numarul minim de exemple al unei clase (0, 19, 37) pe setul de train
    # if class_records < max_records:
    ovr_sample =  class_records #max_records - class_records
    samples = X_train_aug[np.where(Y_train_aug==i)[0]]
    X_aug = []
    Y_aug = [i] * ovr_sample
    
    for x in range(ovr_sample):
        img = samples[x % class_records]
        # if np.random.rand(1) < 0.8:
        if np.mean(img) < 20:
            trans_img=adjust_gamma(img, 2.5)
        else:
            trans_img=rotation(img)
                
        X_aug.append(trans_img)
        
    X_train_aug = np.concatenate((X_train_aug, X_aug), axis=0)
    Y_train_aug = np.concatenate((Y_train_aug, Y_aug)) 
    
    Y_aug_1 = Y_aug_1 + Y_aug
    X_aug_1 = X_aug_1 + X_aug

def to_one_hot(Y_n):
    Y_new = np.zeros([len(Y_n), 43])
    for i in range(Y_new.shape[0]):
        Y_new[i, Y_n[i]] = 1
    return Y_new

Y_train = to_one_hot(Y_train_aug)
# Y_val = to_one_hot(Y_val)
Y_test = to_one_hot(Y_test)

def classes(Y_n):
    Y_nou = np.zeros(len(Y_n))
    for i in range(len(Y_nou)):
        Y_nou[i] = np.argmax(Y_n[i])
    return (Y_nou).astype('uint8')

def standard(img):
    return ((img - np.min(img))/(np.max(img)-np.min(img)))-0.5

X_train = standard(X_train_aug)
# X_val = standard(X_val)
X_test = standard(X_test)

with tf.device('/gpu:0'): # ruleaza pe GPU
    x_initializer = tf.contrib.layers.xavier_initializer()
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3]) 
    y_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes])
    # lr = tf.placeholder(tf.float32, shape=[]) #pentru a putea ajusta pasul de actualizare in timpul antrenarii
    is_training = tf.placeholder(tf.bool, shape=()) # pentru stratul de Batch Normalization

def mn_v2_reziduu(intrare, expand, squeeze, ksize, strides, padding, reziduu):
# --------------- 1x1 -> 3x3 -> 1x1
    k1x1_1 = 1
    c1x1_1 = int(intrare.get_shape()[3])
    h1x1_1 = expand

    W1x1_1 = tf.Variable(x_initializer([1,1,int(intrare.get_shape()[3]),expand]))
    b1x1_1 = tf.Variable(x_initializer([expand]))
    conv1x1_1 = tf.nn.relu6(tf.compat.v1.layers.batch_normalization(
                tf.add(tf.nn.conv2d(intrare, W1x1_1, strides=[1,1,1,1], padding="SAME"), b1x1_1), 
                training=is_training, momentum=0.9, trainable=True))
        # conv dw 3x3   BN + relu
    pc1x1_1 =  (k1x1_1*k1x1_1*c1x1_1+1)*h1x1_1
    cc1x1_1 = (k1x1_1*k1x1_1*c1x1_1+1)*h1x1_1*conv1x1_1.get_shape()[1]*conv1x1_1.get_shape()[2]


    k3x3 = ksize
    c3x3 = int(conv1x1_1.get_shape()[3])
    h3x3 = 1

    W3x3 = tf.Variable(x_initializer([ksize,ksize,int(conv1x1_1.shape[3]),1]))
    b3x3 = tf.Variable(x_initializer([expand]))
    conv3x3 = tf.nn.relu6(tf.compat.v1.layers.batch_normalization(
                tf.add(tf.nn.depthwise_conv2d(conv1x1_1, W3x3, strides=[1,strides,strides,1], padding=padding), b3x3), 
                training=is_training, momentum=0.9, trainable=True))
        # conv pw 1x1   BN
    pc3x3 =  (k3x3*k3x3*c3x3+1)*h3x3
    cc3x3 = (k3x3*k3x3*c3x3+1)*h3x3*conv3x3.get_shape()[1]*conv3x3.get_shape()[2]

    k1x1_2 = 1
    c1x1_2 = int(conv3x3.get_shape()[3])
    h1x1_2 = squeeze

    W1x1_2 = tf.Variable(x_initializer([1,1,int(conv3x3.get_shape()[3]),squeeze]))
    b1x1_2 = tf.Variable(x_initializer([squeeze]))
    conv1x1_2 = tf.compat.v1.layers.batch_normalization(
                tf.add(tf.nn.conv2d(conv3x3, W1x1_2, strides=[1,1,1,1], padding="SAME"), b1x1_2) + reziduu, 
                training=is_training, momentum=0.9, trainable=True)
    pc1x1_2 =  (k1x1_2*k1x1_2*c1x1_2+1)*h1x1_2
    cc1x1_2 = (k1x1_2*k1x1_2*c1x1_2+1)*h1x1_2*conv1x1_2.get_shape()[1]*conv1x1_2.get_shape()[2]

    m1_pc = pc1x1_1 + pc3x3 + pc1x1_2
    m1_cc = cc1x1_1 + cc3x3 + cc1x1_2
    print (' param', m1_pc, ' conn', m1_cc)
    return conv1x1_2
    
def mn_v1(intrare, expand, strides, padding):
        # conv dw 3x3   BN + relu
    shape_c3x3 = [3,3,int(intrare.get_shape()[3]),1] 
    W3x3 = tf.Variable(x_initializer(shape_c3x3))
    b3x3 = tf.Variable(x_initializer(shape_c3x3[-1:]))
    c3x3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(
                tf.add(tf.nn.depthwise_conv2d(intrare, W3x3, strides=[1,strides,strides,1], padding=padding), b3x3), 
                training=is_training, momentum=0.9, trainable=True))
        # conv pw 1x1   BN 
    shape_c1x1 = [1,1,int(c3x3.get_shape()[3]),expand] 
    W1x1 = tf.Variable(x_initializer(shape_c1x1))
    b1x1 = tf.Variable(x_initializer(shape_c1x1[-1:]))
    c1x1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(
                tf.add(tf.nn.conv2d(c3x3, W1x1, strides=[1,1,1,1], padding="SAME"), b1x1), 
                training=is_training, momentum=0.9, trainable=True))
    return c1x1
    

    # --------------------------------- C1 -----------------------------------------------------
    
    k1 = 3
    c1 = 3
    h1 = 64
    
    c1_W = tf.Variable(x_initializer([ 3, 3, 3, 64 ]))
    c1_b = tf.Variable(x_initializer([64]))
    conv1 = tf.add(tf.nn.conv2d(x_placeholder, c1_W, strides=[1,1,1,1], padding="VALID"), c1_b)
    c1_bn = tf.compat.v1.layers.batch_normalization(conv1, training=is_training, momentum=0.9, trainable=True)
    c1_bn_act = tf.nn.relu(c1_bn)
    pc1 =  (k1*k1*c1+1)*h1
    cc1 = (k1*k1*c1+1)*h1*int(c1_bn_act.get_shape()[1])*int(c1_bn_act.get_shape()[2])
    print('  param', pc1, '  conn', cc1 )
    print("\nC1:", c1_bn_act.get_shape())

    
    # ------------------------------- R1 ---------------------------------------
    r1 = mn_v1(intrare=c1_bn_act, expand=128, strides=1, padding="VALID")
    # ------------------------------ C2 - separabila ------------------------------------------------
    m1_s1 = mn_v2_reziduu(intrare=c1_bn_act, expand=128*2, squeeze=128, ksize=3, strides=1, padding="VALID", reziduu=r1) 
    print('m1_s1 + r', m1_s1.get_shape())
    
    # # ------------------------------ C3 - separabila ------------------------------------------------
    # m1_s2 = mn_v2_reziduu(intrare=m1_s1, expand=256, squeeze=128, ksize=3, strides=1, padding="SAME", reziduu=m1_s1) 
    # print('m1_s2 + r', m1_s2.get_shape())  
    
    # # ------------------------------ C4 - separabila ------------------------------------------------
    # m1_s3 = mn_v2_reziduu(intrare=m1_s2, expand=256, squeeze=128, ksize=3, strides=1, padding="SAME", reziduu=m1_s2) 
    # print('m1_s3 + r', m1_s3.get_shape())  
    
    # # ------------------------------ C5 - separabila ------------------------------------------------
    # m1_s4 = mn_v2_reziduu(intrare=m1_s3, expand=256, squeeze=128, ksize=3, strides=1, padding="SAME", reziduu=m1_s3) 
    # print('m1_s4 + r', m1_s4.get_shape())  
    
    
    # ------------------------------- R2 ---------------------------------------
    r2 = mn_v1(intrare=m1_s1, expand=256, strides=2, padding="SAME")   
    # ------------------------------ C2 - separabila ------------------------------------------------
    m2_s1 = mn_v2_reziduu(intrare=m1_s1, expand=256*2, squeeze=256, ksize=3, strides=2, padding="SAME", reziduu=r2) 
    print('m2_s1 + r', m2_s1.get_shape())
    
    
    # ------------------------------- R3 ---------------------------------------
    r3 = mn_v1(intrare=m2_s1, expand=256, strides=1, padding="VALID")
    # ------------------------------ C1 - separabila ------------------------------------------------
    m3_s1 = mn_v2_reziduu(intrare=m2_s1, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=r3) 
    print('m3_s1 + r', m3_s1.get_shape())
    
            
    # ------------------------------- R4 ---------------------------------------
    r4 = mn_v1(intrare=m3_s1, expand=384, strides=1, padding="VALID")
    # ------------------------------ C1 - separabila ------------------------------------------------
    m4_s1 = mn_v2_reziduu(intrare=m3_s1, expand=384*2, squeeze=384, ksize=3, strides=1, padding="VALID", reziduu=r4) 
    print('m4_s1 + r', m4_s1.get_shape())   
    
    # # ------------------------------ C2 - separabila ------------------------------------------------
    # m4_s2 = mn_v2_reziduu(intrare=m4_s1, expand=512, squeeze=256, ksize=3, strides=1, padding="SAME", reziduu=m4_s1) 
    # print('m4_s2 + r', m4_s2.get_shape())  
    
    # # ------------------------------ C3 - separabila ------------------------------------------------
    # m4_s3 = mn_v2_reziduu(intrare=m4_s2, expand=512, squeeze=256, ksize=3, strides=1, padding="SAME", reziduu=m4_s2) 
    # print('m4_s3 + r', m4_s3.get_shape())  
    
    # # ------------------------------ C4 - separabila ------------------------------------------------
    # m4_s4 = mn_v2_reziduu(intrare=m4_s3, expand=512, squeeze=256, ksize=3, strides=1, padding="SAME", reziduu=m4_s3) 
    # print('m4_s4 + r', m4_s4.get_shape()) 
    
    # ------------------------------- R5 ---------------------------------------
    r5 = mn_v1(intrare=m4_s1, expand=256, strides=2, padding="SAME")
    # ------------------------------ C5 - separabila ------------------------------------------------
    m5 = mn_v2_reziduu(intrare=m4_s1, expand=256*2, squeeze=256, ksize=3, strides=2, padding="SAME", reziduu=r5) 
    print('m5 + r', m5.get_shape()) 
    
    # ------------------------------- R5 ---------------------------------------
    r6 = mn_v1(intrare=m5, expand=256, strides=1, padding="VALID")
    # ------------------------------ C final - separabila ------------------------------------------------
    m6 = mn_v2_reziduu(intrare=m5, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=r6) 
    print('m6 + r', m6.get_shape()) 
    
    # ------------------------------ C final - separabila ------------------------------------------------
    m7 = mn_v2_reziduu(intrare=m6, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=0) 
    print('m7 + r', m7.get_shape()) 

    ### ----------------------------------- FLATTEN ---------------------------------------
    
    
    size = m7.get_shape()
    num_f = size[1:].num_elements()
    # flat = tf.reshape(m7, [-1, num_f])
    
    
    
    embedding_p = tf.reshape(m7, [-1, num_f])
        
    weights_n = tf.Variable(x_initializer([num_f, num_classes]))
       
    embedding_norm = tf.norm(embedding_p, axis=1, keep_dims=True)  
    embedding = tf.divide(embedding_p, embedding_norm)
    
    weights_norm = tf.norm(weights_n, axis=0, keep_dims=True) 
    weights = tf.divide(weights_n, weights_norm)
    
    
    m = 0.5
    s = 64.
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    
    mm = sin_m * m
    
    threshold = math.cos(math.pi - m)
    
    cos_t = tf.matmul(embedding, weights)
    s_cos_t = tf.multiply(s, cos_t)
    
    cos_t2 = tf.square(cos_t)
    sin_t2 = tf.subtract(1., cos_t2)
    
    sin_t = tf.sqrt(sin_t2)
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m))
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = y_placeholder
    inv_mask = tf.subtract(1., mask)
    
    arcface = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
    
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface, labels=tf.argmax(y_placeholder, axis=1)))
    
    # optimisation
    # optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_locking=False, 
    #                                                     use_nesterov=False).minimize(loss=cost)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(loss=cost)
    
    y_pred = tf.argmax(tf.nn.softmax(cos_t), axis=1)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y_placeholder, axis=1)), tf.float32))
    
    
    ## -------------------------------- FULLY CONNECTED - OUTPUT ------------------------------------
    
    # in1 = int(flat.get_shape()[1])
    # out1 = num_classes
    # shape_fc1 = [in1, out1]
    # fc1_W = tf.Variable(x_initializer(shape_fc1))
    # fc1_b = tf.Variable(x_initializer(shape_fc1[-1:]))
    
    # logit = tf.matmul(flat, fc1_W) + fc1_b
    # a = tf.nn.softmax(logit)
    # y_pred = tf.argmax(a, axis=1)
    # print("\nOutput: ", a.get_shape()[1:])

    pc_out = (256+1)*43
    print('  param', pc_out, '  conn', pc_out )

    
    # ## FUNCTIE COST
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit, labels = y_placeholder)
    # cost = tf.reduce_mean(cross_entropy)
    # # ALGORITM DE OPTIMIZARE
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(loss=cost)
    # # PERFORMANTA - ACURATETE
    # eq = tf.equal(y_pred, tf.argmax(y_placeholder, axis=1))
    # acc = tf.reduce_mean(tf.cast(eq, tf.float32))
    

    # ------------------- SESIUNE -------------------------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    
    num_batches_train = X_train.shape[0] // batch_size # numar de batchuri pentru train 
    # num_batches_val = X_val.shape[0] // batch_size # numar de batchuri pentru validare 
    num_batches_test = X_test.shape[0] // batch_size # numar de batchuri pentru test 
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops_n = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)


    with tf.control_dependencies(update_ops):
        
        t_train_start = time.time()
        
        ac_antrenare = np.empty(epochs)
        # ac_validare = np.empty(epochs)
        
        for e in range(epochs):
            
            start_optimizare = time.time()
            
            X_train[:], Y_train[:] = shuffle(X_train, Y_train)
            
            # ------------------------------ ANTRENARE ---------------------------------------------
            for i in range(num_batches_train):
                
                x_batch = X_train[i*batch_size:(i+1)*batch_size, :]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size, :]
                
                feed_dict_train = {x_placeholder:x_batch, y_placeholder:y_batch, is_training:True}            
                sess.run([optimizer, update_ops_n], feed_dict=feed_dict_train)
            # -------------------------------------------------------------------------------------- 
                    
            
            # ------------------------------ ACURATETE ANTRENARE -----------------------------------
            acc_train_total = 0
            for i in range(num_batches_train):
                
                x_batch = X_train[i*batch_size:(i+1)*batch_size, :]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size, :]
                
                feed_dict_train = {x_placeholder:x_batch, y_placeholder:y_batch, is_training:False}
                acc_train_batch = sess.run(acc, feed_dict=feed_dict_train)
                acc_train_total += acc_train_batch
                
            acc_train_total /= num_batches_train
            ac_antrenare[e] = acc_train_total
            #---------------------------------------------------------------------------------------
    
                
            # # ------------------------------ ACURATETE VALIDARE ---------------------------------------------
            # acc_val_total = 0
            # for i in range(num_batches_val):
                
            #     x_batch = X_val[i*batch_size:(i+1)*batch_size, :]
            #     y_batch = Y_val[i*batch_size:(i+1)*batch_size, :]
                
            #     feed_dict_val = {x_placeholder:x_batch, y_placeholder:y_batch, is_training:False}
            #     acc_val_batch = sess.run(acc, feed_dict=feed_dict_val)
            #     acc_val_total += acc_val_batch
                
            # acc_val_total /= num_batches_val
            # ac_validare[e] = acc_val_total
            # # -----------------------------------------------------------------------------------------------
    
            
            end_epoca = time.time()
            t_epoca = end_epoca - start_optimizare
            
            print('Epoca: ' + str(e) +
                  ' Ac.ant: ' + '{0:.4f}'.format(acc_train_total * 100) + ' %  ' +
                    # ' Ac.val: ' + '{0:.4f}'.format(acc_val_total * 100) + ' %  ' +
                  " Timp: " + '{0:.4f}'.format(t_epoca) + ' s')
            #---------------------------- SFARSIT EPOCA -------------------------------------------------
        t_train_stop = time.time()   
        timp_train = t_train_stop - t_train_start
        
        # ------------------------------ ACURATETE TESTARE ---------------------------------------------
        t_test_strat = time.time()
        acc_test_total = 0
        for i in range(num_batches_test):
            t_test_strat = time.time()
                
            x_batch = X_test[i*batch_size:(i+1)*batch_size, :]
            y_batch = Y_test[i*batch_size:(i+1)*batch_size, :]
                    
            feed_dict_test = {x_placeholder:x_batch, y_placeholder:y_batch,  is_training:False}
            acc_test_batch = sess.run(acc, feed_dict=feed_dict_test)
            acc_test_total += acc_test_batch
                
        acc_test_total /= num_batches_test
        t_test_stop = time.time()
        
        timp_test = t_test_stop - t_test_strat
        
        
        print('Acuratete antrenare: ', '{0:.4f}'.format(acc_train_total * 100), ' %')
        print('Acuratete testare: ', '{0:.4f}'.format(acc_test_total * 100), ' %')
        print('timp antrenare: ', timp_train)
        print('timp testare: ', timp_test)
        
        # def plot_acc(aca, acv):
        #     plt.plot(np.arange(1, len(aca)+1), aca, '*-' )
        #     plt.plot(np.arange(1, len(acv)+1), acv, 'o-' )
        #     plt.xlabel('epoca')
        #     plt.ylabel('acuratete')
        #     plt.legend(['ac. ant', 'ac. val'], loc='upper left')

        # plot_acc(ac_antrenare, ac_validare)
        
        # num_batches_train = X_train.shape[0] // batch_size
        # Y_pred_train = (np.empty(Y_train.shape[0])).astype('uint8')
        # for i in range(num_batches_train):  
            
        #     x_batch = X_train[i*batch_size:(i+1)*batch_size, :]
        #     y_batch = Y_train[i*batch_size:(i+1)*batch_size, :]
            
        #     feed_dict_train = {x_placeholder:x_batch, y_placeholder:y_batch,  is_training:False}
            
        #     Y_pred_train[i*batch_size:i*batch_size+batch_size] = sess.run(y_pred, feed_dict=feed_dict_train)
        
        # c_m = tf.confusion_matrix(labels = classes(Y_train), predictions = Y_pred_train)
        # conf_matrix = sess.run(c_m)
        # print(conf_matrix)
        
        # # confusion = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        # c = standard(conf_matrix)+0.5
        # fig= plt.figure(figsize=(600,600))
        # plt.rcParams.update({'font.size': 10})
        # plt.matshow(c)
        # # Make various adjustments to the plot.
        # plt.colorbar()
        # tick_marks = np.arange(num_classes)
        
        # labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
        #           '16','17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
        #           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']
        
        # plt.rcParams['figure.figsize'] = (12,12)
        # plt.yticks(tick_marks, labels)
        # #plt.set_xticklabels(labels)
        # plt.rcParams.update({'font.size': 10})
        # plt.xticks(tick_marks, labels, rotation=45)
        # plt.xlabel('Prezis')
        # plt.ylabel('Adevarat')
        # plt.show()
