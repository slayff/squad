
# coding: utf-8

# # L2 - Multilayer perceptron
# 
# ### Papers
# 1. [TensorFlow](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
# 
# ### TensorFlow
# 1. [Installing TensorFlow](https://www.tensorflow.org/install/)
# 2. [Basics of TensorFlow](https://www.tensorflow.org/get_started/get_started)
# 3. [Mnist with TensorFlow](https://www.tensorflow.org/get_started/mnist/pros)
# 4. [TensorFlow Mechanics](https://www.tensorflow.org/get_started/mnist/mechanics)
# 5. [Visualization](https://www.tensorflow.org/get_started/graph_viz)
# 
# 
# ### One more thing
# 1. [Jupyter tutorial](https://habrahabr.ru/company/wunderfund/blog/316826/)
# 2. [Plot.ly](https://plot.ly/python/)
# 3. [Widgets](http://jupyter.org/widgets.html)

# ### 1. Linear multi-classification problem
# 
# We have already learned binary linear classifier
# $$y = \text{sign}(w^Tx).$$
# There are [several approaches](https://en.wikipedia.org/wiki/Multiclass_classification) to solve the problem of multi-class classification. For example [reduction](https://en.wikipedia.org/wiki/Multiclass_classification#Transformation_to_Binary) of problem to binary classifier or [modification](https://en.wikipedia.org/wiki/Support_vector_machine#Multiclass_SVM) of the known model. However we are interested in approaches that is applied in neural networks.
# 
# For each class $c \in 1, \dots, |C|$ we have an individual row $w_i$ of matrix $W$. Then the probability of $x$ belonging to a particular class is equal to
# $$p_i = \frac{\exp(w_i^Tx)}{\sum_j \exp(w_j^Tx)}.$$
# This is nothing, but [softmax](https://en.wikipedia.org/wiki/Softmax_function) function of $Wx$.
# $$(p_1, \dots, p_{|C|}) = \text{softmax}(Wx).$$
# 
# If you look closely, $\text{softmax}$ is a more general variant of sigmoid. To see this, it suffices to consider the case $|C|=2$. As usual the training can be reduced to minimization of the empirical risk, namely, optimization problem
# $$\arg\min_W Q(W) = \arg\min_W -\frac{1}{\mathcal{l}}\sum_y\sum_i [y = i] \cdot \ln(p_i(W)).$$
# Actually, the maximization of the log-likelihood is written above.
# 
# #### Exercises
# 1. Find $\frac{dQ}{dW}$ in matrix form (hint: start with $\frac{dQ}{dw_i}$ for begining).
# 2. Please plot several mnist images (e.g using grid 5x5).
# 3. Train linear multi-label classifier for [mnist](https://www.kaggle.com/c/digit-recognizer) dataset with TensorFlow (possible, [this tutorial](https://www.tensorflow.org/get_started/mnist/pros) can help you).
# 4. Chek accuracy on train and validation sets.
# 5. Use a local [TensorBoard instance](https://www.tensorflow.org/get_started/graph_viz) to visualize resulted graph (no need to include in lab).

# ## Solutions
# -----
# 
# Let's find $\frac{dQ}{dW}$. Firstly, we rewrite the loss funtion in more convinient way:
# 
# $$Q(W) = -\frac{1}{\mathcal{l}}\sum_{j=1}^{l}\sum_{i=1}^{k} [y^{(j)} = i] \cdot \ln(\frac{\exp(w_i^Tx^j)}{\sum_l \exp(w_l^Tx^j)})$$
# 
# Here we use the following notation: $l$ is the number of elements in a set, $k$ is the number of labels and $y$ is the vector-string of labels such that $y^{(j)}$ corresponds to the element $j$. We assume that $W$ is a parameter-matrix, where vectors $w_i$ are stacked in rows. Therefore vectors $x_i$, which represent the attributes of $i$-th element, are stacked in cols in matrix $X$.
# 
# It's easier to come up with the solution by differentiating the function respectively to the $w_i$ row:
# 
# $$\frac{\partial Q}{\partial w_i} = -\frac{1}{\mathcal{l}} \big( \sum_{j=1}^{l}\big[y^{(j)} = i\big] \frac{\sum_l \exp(w_l^Tx^j)}{\exp(w_i^Tx^j)}  \cdot \frac{\sum_l \exp(w_l^Tx^j) \cdot \exp(w_i^Tx^{(j)})x^{(j)} - (\exp(w_i^Tx^{(j)})^2x^{(j)}} {(\sum_l \exp(w_l^Tx^j))^2} \big)$$
# 
# 
# $$\frac{\partial Q}{\partial w_i} = -\frac{1}{\mathcal{l}} \big( \sum_{j=1}^{l} x^{(j)}\big( [y^{(j)} = i] - \frac{\exp(w_i^Tx^j)}{\sum_l \exp(w_l^Tx^j)}\big)\big)$$
# 
# Actually, $\frac{\partial Q}{\partial w_i}$ is itself a vector such that it's $j$-th element is a partial derivative $\frac{\partial Q}{\partial w_{ij}}$. In order to rewrite the formula in a matrix form we introduce a new matrix $Y : Y_{ij} = \left\{
# \begin{aligned}
# 1&, \text{if j-th element has i-th label} \\
# 0&, \text{otherwise}
# \end{aligned}
# \right.$
# 
# Using $Y$, we can finally write the $\frac{dQ}{dW}$:
# 
# $$\frac{dQ}{dW} = -\frac{1}{\mathcal{l}} \big[ YX^T - PX^T\big]$$
# 
# Where $P$ is a matrix $WX$ with the softmax function being applied.

# In[2]:


import tensorflow as tf
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')

mnist_data = pd.read_csv('train.csv', sep=',')
data = mnist_data.values


# In[3]:


samplesN = 5
samples = np.random.randint(0, data.shape[0], samplesN**2)
f, ax  = plt.subplots(samplesN, samplesN)
f.set_size_inches(7, 7)
for i, s in enumerate(samples):
    ax[i // samplesN, i % samplesN].axis('off')
    ax[i // samplesN, i % samplesN].imshow(data[s, 1:].reshape(28, 28))


# Let's now implement simple linear multi-label classifier using `Tensorflow`:

# In[6]:


def NextBatch(X, Y, batch_size):
    begin = 0
    x_size = X.shape[0]
    while begin < x_size:
        end = min(begin + batch_size, x_size)
        yield X[begin:end], Y[begin:end]
        begin = end
        
mnist_labels = data[:, 0]
mnist_set = data[:, 1:]

#Let's preprocess data:
mnist_set = mnist_set - np.mean(mnist_set, axis=0, dtype=np.float64)
mnist_set /= 256.
#mnist_set = mnist_set / (np.std(mnist_set, axis=0, dtype=np.float64) + 1e-20)

x_train, x_test, y_train, y_test = train_test_split(mnist_set, mnist_labels,
                                                    test_size=0.15, random_state=30)


# In[23]:


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int32, shape=[None])
W = tf.Variable(tf.truncated_normal([784,10], stddev=0.05))
b = tf.Variable(tf.truncated_normal([10], stddev=0.05))
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[8]:


from IPython.display import clear_output


# In[53]:


BATCH_SIZE = 100
EPOCH_NUM = 10

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs_tensorboard/run_" + str(RUN_NUM) + "/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH_NUM):
        iter_num = 0
        x_train, y_train = shuffle(x_train, y_train)
        for x_batch, y_batch in NextBatch(x_train, y_train, BATCH_SIZE):
            train_step.run(feed_dict={x: x_batch, y_: y_batch})
            iter_num += 1
            train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
            print('step %d, epoch %d, training accuracy %g' % (iter_num, epoch, train_accuracy))
            clear_output()

    print('accuracy on test %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test}))
    print('accuracy on train %g' % accuracy.eval(feed_dict={x: x_train, y_: y_train}))
    writer.close()


# Let's briefly touch on themes of regularization. As was discussed before, there are different approaches. We focus on the modification of loss function.
# 
# $$\arg\min_W -\frac{1}{\mathcal{l}}\sum_y\sum_i [y = i] \cdot \ln(p_i(W)) + \lambda_1 L_1(W) + \lambda_2 L_2(W)$$
# 
# 1. $L_1(W) = sum_{i,j} |w_{i,j}|$ - sparsify weights (force to not use uncorrelated features)
# 2. $L_2(W) = sum_{i,j} w_{i,j}^2$ - minimize weights (force to not overfit)
# 
# #### Exercises
# 1. Train model again using both type of regularization.
# 2. Plot matrix of weights.
# 3. Which pixels have zero weights? What does it mean?
# 4. Have you improved accuracy on validation?

# In[87]:


RUN_NUM = 0


# In[123]:


g2 = tf.Graph()
LAMBDA_1 = 1e-5
LAMBDA_2 = 1e-5
with g2.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.int32, shape=[None])
    W = tf.Variable(tf.truncated_normal([784,10], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([10], stddev=0.05))
    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    y = tf.matmul(x,W) + b
    penalty_l1 = tf.reduce_sum(tf.abs(W))
    penalty_l2 = tf.reduce_sum(tf.abs(W**2))
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,
                                                                                  logits=y)) + \
                                    LAMBDA_1 * penalty_l1 + LAMBDA_2 * penalty_l2
    tf.summary.scalar("loss_function", cross_entropy)
    
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    merged = tf.summary.merge_all()


# In[130]:


BATCH_SIZE = 100
EPOCH_NUM = 10
saver = tf.train.Saver({'w_predicted': W})
with tf.Session(graph=g2) as sess:
    RUN_NUM += 1
    writer = tf.summary.FileWriter("logs_tensorboard/run_" + str (RUN_NUM) + "/", sess.graph)
    sess.run(tf.global_variables_initializer())
    iter_num = 0
    for epoch in range(EPOCH_NUM):
        x_train, y_train = shuffle(x_train, y_train)
        for x_batch, y_batch in NextBatch(x_train, y_train, BATCH_SIZE):
            train_step.run(feed_dict={x: x_batch, y_: y_batch})
            iter_num += 1
            summary = sess.run(merged, feed_dict={x: x_batch, y_:y_batch})
            writer.add_summary(summary, iter_num)
            train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
            print('step %d, epoch %d, training accuracy %g' % (iter_num, epoch, train_accuracy))
            clear_output()
            
    save_path = saver.save(sess, "logs_tensorboard/models/model.ckpt")
    print('accuracy on test %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test}))
    print('accuracy on train %g' % accuracy.eval(feed_dict={x: x_train, y_: y_train}))
    writer.close()


# In[181]:


with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "logs_tensorboard/models/model.ckpt")
    w_predicted = sess.run(W)
    samplesN = 10
    samples = range(samplesN + 2)
    f, ax  = plt.subplots(4, 3)
    f.set_size_inches(10, 10)
    
    for i in samples:
        ax[i // 3, i % 3].axis('off')
        if (i < 10):
            ax[i // 3, i % 3].imshow(w_predicted[:, i].reshape(28, 28), cmap='magma')
            ax[i // 3, i % 3].set_title('number %d' % i)
            


# Several test showed that small $\lambda$ leads to a strong underfitting with accuracy of about $0.2$. The parameter $\lambda = 0,00001$ gives approximately the same accuracy rate as without regularization (a bit smaller on both train and validation sets, to be precise).
# 
# As we have plot the "images" of weights, we can say why certain pixels are lighter and therefore have almost zero weight. These pixels (weights) are responsible for evaluating whether the number is in correct class - we can see, that the shape of light pixels resembles the numbers. If the test picture is the number of the current label, then we'll compute loss by multiplying pixels by weights and that loss will be adequately small.
# 
# On the other hand, we can see dark areas on these pictures. It's how our model knows that the current test picture has definetely another label - when computing loss, we will get reasonably high values.

# ### 2. Universal approximation theorem
# 
# What if we add more layers to our model? Namely, we train two matrix $W_2$ and $W_1$
# $$softmax(W_2\cdot(W_1x)).$$
# 
# At first glance adding more parameters helps to increase the generalizing ability of the model. Buy actually we have the same model $softmax(Wx)$, where $W = W_2\cdot W_1$. But everyting changes with adding ome more layer. Let's add nonlinear function $\sigma$ between $W_2$ and $W_1$
# 
# $$softmax(W_2\cdot \sigma(W_1x)).$$
# 
# Kurt Hornik showed in 1991 that it is not the specific choice of the nonlinear function, but rather the multilayer feedforward architecture itself which gives neural networks the potential of being universal approximators. The output units are always assumed to be linear. For notational convenience, only the single output case will be shown. The general case can easily be deduced from the single output case.
# 
# Let $\sigma(\cdot)$ be a nonconstant, bounded, and monotonically-increasing continuous function.
# Let $\mathcal{S}_m \subset \mathbb{R}^m$ denote any compact set. 
# Then, given any $\varepsilon > 0$ and any coninuous function $f$ on $\mathcal{S}_m$, there exist an integer $N$ and real constants $v_i$, $b_i$ amd real vectors $w_i$ that
# 
# $$\left| \sum _{i=1}^{N}v_{i}\sigma \left( w_i^{T} x+b_i \right) - f(x) \right| < \varepsilon, ~~~ \forall x \in \mathcal{S}_m.$$
# 
# The theorem has non-constructive proof, it meams that no estimates for $N$ and no method to find approximation's parameters.
# 
# #### Exercises
# 1. Let $\sigma$ – [heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) and $x \in \{0, 1\}^2$. Prove that $y = \sigma(wx + b)$ can approximate boolean function **OR** (hint: use constructive proof).
# 2. What about **AND** function?
# 3. Is it possible to implement **XOR**? Prove your words.
# 4. Prove that 2-layer network can implement any boolean function.
# 
# #### More useful facts:
# 1. A 2-layer network in in $\mathbb{R}^n$ allows to define convex polyhedron..
# 2. A 3-layer network in в $\mathbb{R}^n$ allows to define a not necessarily convex and not even necessarily connected area.

# ### 3. Backpropagation
# Backpropagation is a method used to calculate the error contribution of each layer after a batch of data. It is a special case of an older and more general technique called automatic differentiation. In the context of learning, backpropagation is commonly used by the gradient descent optimization algorithm to adjust the weight of layers by calculating the gradient of the loss function. This technique is also sometimes called backward propagation of errors, because the error is calculated at the output and distributed back through the network layers. The main motivation of method is simplify evaluation of gradient which is complex problem for multilayer nets.
# 
# We need the following notation. Let $(y^1,\dots,y^n) = f(x^1,\dots,x^n)$ is some differentiable function and $\frac{dy}{dx}$ is matrix
# $$\frac{dy}{dx} = \Big[ J_{ij} = \frac{\partial y^i}{\partial x^j} \Big]$$
# 
# Without violating the generality, we can assume that each layer is a function $x_{i} = f(x_{i-1}, w_i)$. As last layer we add loss function, so we can assume our multi-layer net as function $Q(x_0) = Q(f_n(f_{n-1}(\dots, w_{n-1}), w_n))$.
# 
# #### Forward step
# Propagation forward through the network to generate the output values. Calculation of the loss function.
# 
# #### Backward step
# Let's look at last layer. We can simply find $\frac{dQ}{dx_n}$. Now we can evaluate 
# 
# $$\frac{dQ}{dw_n} = \frac{dQ}{dx_n}\frac{dx_n}{dw_n} \text{ and } \frac{dQ}{dx_{n-1}} = \frac{dQ}{dx_n}\frac{dx_n}{dx_{n-1}}$$
# 
# Now we need calculate $\frac{dQ}{dw_{n-2}}$ и $\frac{dQ}{dx_{n-2}}$. But we have the same situation. We know $\frac{dQ}{dx_k}$, so can evaluate $\frac{dQ}{dw_k}$ and $\frac{dQ}{dx_{k-1}}$. Repeating this operation we find all the gradients. Now it's only remains to make a gradient step to update weights.
# 
# #### Exercises
# 1. Read more about [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
# 2. Train 2 layer net. Use sigmoid as nonlinearity.
# 3. Check accuracy on validation set.
# 4. Use [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) or LeakyReLu as nonlinearity. Compare accuracy and convergence with previous model.
# 5. Play with different architectures (add more layers, regularization and etc).
# 6. Show your best model.
# 7. How does quality change with adding layers. Prove your words, train model for 2, 3, 5, 7 and 10 layers.
# 8. Using backpropagation find optimal  digit 8 for your net.*

# In[19]:


def TrainModel(
    layers_num=2,
    layers_sizes=[784, 10],
    activation_f='sigmoid',
    l1_reg_coef=0.,
    l2_reg_coef=0.,
    batch_size=100,
    epoch_num=10
):
    if layers_num > len(layers_sizes):
        layers_sizes = [784]
        k = layers_num - 1
        while layers_sizes[-1] % 2 == 0 and k - 1:
            layers_sizes.append(layers_sizes[-1] // 2)
            k -= 1
        while k - 1:
            layers_sizes.append(layers_sizes[-1])
            k -= 1
        layers_sizes.append(10)
    elif len(layers_sizes) > layers_num:
        layers_num = len(layers_sizes)
            
    g_ = tf.Graph()
    with g_.as_default():
        activation_dict = {'sigmoid' : tf.sigmoid,
                           'relu' : tf.nn.relu,
                           'tanh' : tf.nn.tanh,
                           'leakyrelu': tf.nn.leaky_relu
                          }
        activation = activation_dict[activation_f]
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int32, shape=[None])
        layers_list = [x]
        for i in range(1, layers_num):
            layers_list.append(tf.layers.dense(layers_list[i - 1],
                                              layers_sizes[i - 1],
                                              activation=activation,
                                              kernel_initializer=tf.truncated_normal_initializer(
                                                  stddev=0.1),
                                              bias_initializer=tf.truncated_normal_initializer(
                                                  stddev=0.1),
                                              activity_regularizer=tf.contrib.layers.l1_l2_regularizer(
                                              l1_reg_coef, l2_reg_coef)
                                             ))
        y = tf.layers.dense(layers_list[-1],
                            layers_sizes[-1],
                            kernel_initializer=tf.truncated_normal_initializer(
                                        stddev=0.1),
                            bias_initializer=tf.truncated_normal_initializer(
                                        stddev=0.1),
                            activity_regularizer=tf.contrib.layers.l1_l2_regularizer(
                                        l1_reg_coef, l2_reg_coef)
                           )
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,
                                                                                      logits=y))
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session(graph=g_) as sess:
        sess.run(tf.global_variables_initializer())
        iter_num = 0
        for epoch in range(epoch_num):
            x_train_, y_train_ = shuffle(x_train, y_train)
            for x_batch, y_batch in NextBatch(x_train_, y_train_, batch_size):
                train_step.run(feed_dict={x: x_batch, y_: y_batch})
                iter_num += 1
                train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch})
                print('step %d, epoch %d, training accuracy %g' % (iter_num, epoch, train_accuracy))
                clear_output()
        print('Testing model with', activation_f, 'nonlinearity')
        print('Layers are used :', layers_sizes)
        print('accuracy on train %g' % accuracy.eval(feed_dict={x: x_train, y_: y_train}))
        print('accuracy on test %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test}))


# In[197]:


TrainModel(layers_num=2, activation_f='sigmoid', l1_reg_coef=1e-3, l2_reg_coef=1e-3)


# In[198]:


TrainModel(layers_num=2, activation_f='leakyrelu', l1_reg_coef=1e-3, l2_reg_coef=1e-3)


# In[199]:


TrainModel(layers_sizes=[784, 392, 196, 10], activation_f='leakyrelu', l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[203]:


TrainModel(layers_sizes=[784, 392, 100, 10], activation_f='tanh')


# In[204]:


TrainModel(layers_sizes=[784, 196, 10], activation_f='relu', batch_size=60, epoch_num=15)


# In[205]:


TrainModel(layers_sizes=[784, 392, 10], activation_f='leakyrelu', batch_size=60, epoch_num=20,
         l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[208]:


TrainModel(layers_sizes=[784, 392, 196, 10], activation_f='leakyrelu', batch_size=60, epoch_num=10)


# Let's test, how the size of the whole network affects the convergence and accuracy:

# In[10]:


TrainModel(layers_num=2, activation_f='leakyrelu', l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[11]:


TrainModel(layers_num=3, activation_f='leakyrelu', l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[12]:


TrainModel(layers_num=5, activation_f='leakyrelu', l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[9]:


TrainModel(layers_num=7, activation_f='leakyrelu', l1_reg_coef=1e-5, l2_reg_coef=1e-5)


# In[20]:


TrainModel(layers_sizes=[784, 784, 784, 784, 784, 784, 784, 784, 784, 10],
           activation_f='leakyrelu')


# ### 4. Autoencoders
# An autoencoder is an network used for unsupervised learning of efficient codings. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction. Also, this technique can be used to train deep nets.
# 
# Architecturally, the simplest form of an autoencoder is a feedforward net very similar to the multilayer perceptron (MLP), but with the output layer having the same number of nodes as the input layer, and with the purpose of reconstructing its own inputs. Therefore, autoencoders are unsupervised learning models. An autoencoder always consists of two parts, the encoder and the decoder. Encoder returns latent representation of the object (compressed representation, usuallu smaller dimension), but decoder restores object from this latent representation. Autoencoders are also trained to minimise reconstruction errors (e.g. MSE).
# 
# Various techniques exist to prevent autoencoders from learning the identity and to improve their ability to capture important information:
# 1. Denoising autoencoder - take a partially corrupted input.
# 2. Sparse autoencoder - impose sparsity on the hidden units during training (whilst having a larger number of hidden units than inputs).
# 3. Variational autoencoder models inherit autoencoder architecture, but make strong assumptions concerning the distribution of latent variables.
# 4. Contractive autoencoder - add an explicit regularizer in objective function that forces the model to learn a function that is robust to slight variations of input values.
# 
# #### Exercises
# 1. Train 2 layers autoencoder that compressed mnist images to $\mathbb{R}^3$ space.
# 2. For each digit plot several samples in 3D axis (use "%matplotlib notebook" mode or plotly). How do digits group?
# 3. Train autoencoder with more layers. What are results?
# 4. Use autoencoder to pretrain 2 layers (unsupervised) and then train the following layers with supervised method.
