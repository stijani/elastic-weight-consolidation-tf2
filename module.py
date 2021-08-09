import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
from tqdm import tqdm


def evaluate(model, test_x, test_y):
    acc = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    for imgs, labels in zip(test_x, test_y):
        preds = model.predict_on_batch(np.array([imgs]))
        acc.update_state(labels, preds)
    return round(100*acc.result().numpy(), 2)


def permute_task(train, test):
    train_shape, test_shape = train.shape, test.shape
    train_flat, test_flat = train.reshape((-1, 784)), test.reshape((-1, 784))
    idx = np.arange(train_flat.shape[1])
    np.random.shuffle(idx)
    train_permuted, test_permuted = train_flat[:, idx], test_flat[:, idx]
    return (train_permuted.reshape(train_shape), test_permuted.reshape(test_shape))


class Train:
    
    def __init__(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train_sgd(self, model, epochs, train_task, test_task_A, test_task_B=None):
        test_acc_A, test_acc_B = [], []
        for epoch in tqdm(range(epochs)):
            for batch in train_task:
                X, y = batch
                with tf.GradientTape() as tape:
                    pred = model(X)
                    loss = self.loss_fn(y, pred)
                grads = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            test_acc_A.append(evaluate(model, test_task_A[0], test_task_A[1]))
            if test_task_B:
                test_acc_B.append(evaluate(model, test_task_B[0], test_task_B[1]))
        return test_acc_A, test_acc_B

    def train_ewc(self, model, prior_weights, lambda_, fisher_matrix, epochs, train_task, test_task_A, test_task_B=None):
        test_acc_A, test_acc_B = [], []
        for epoch in tqdm(range(epochs)):
            for batch in train_task:
                X, y = batch
                with tf.GradientTape() as tape:
                    pred = model(X)
                    loss = self.loss_fn(y, pred)
                    total_loss = loss + self.compute_penalty_loss(model, prior_weights, fisher_matrix, lambda_)
                grads = tape.gradient(total_loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            test_acc_A.append(evaluate(model, test_task_A[0], test_task_A[1]))
            if test_task_B:
                test_acc_B.append(evaluate(model, test_task_B[0], test_task_B[1]))
        return test_acc_A, test_acc_B

    
    def compute_penalty_loss(self, model, prior_weights, fisher_matrix, lambda_):
        penalty = 0.
        for u, v, w in zip(fisher_matrix, model.weights, prior_weights):
            penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
        return 0.5 * lambda_ * penalty
    

class EWC:
    
    def __init__(self, prior_model, data_samples, lambda_, num_sample=30):
        self.prior_model = prior_model
        self.prior_weights = prior_model.weights
        self.lambda_ = lambda_
        self.num_sample = num_sample
        self.data_samples = data_samples
        self.fisher_matrix = self.compute_fisher()
        
    def compute_fisher(self):
        weights = self.prior_weights
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights], 
                           dtype=object
                          )
        for j in tqdm(range(self.num_sample)):
            idx = np.random.randint(self.data_samples.shape[0])
            with tf.GradientTape() as tape:
                logits = tf.nn.log_softmax(self.prior_model(np.array([self.data_samples[idx]])))
            grads = tape.gradient(logits, weights)
            for m in range(len(weights)):
                fisher_accum[m] += np.square(grads[m])
        fisher_accum /= self.num_sample
        return fisher_accum
    
    def get_fisher(self):
        return self.fisher_matrix
    
    def loss_penalty(self, current_weights):
        ewc_loss = 0.
        for u, v, w in zip(current_weights, self.prior_weights, self.fisher_matrix):
            ewc_loss += self.lambda_ * tf.reduce_sum(w * tf.math.square(u - v))
        return ewc_loss
         

class MLP3:
    
    def __init__(self, input_shape=(28, 28), hidden_layers_neuron_list=[200, 200], num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers_neuron_list = hidden_layers_neuron_list
        self.model = self.create_mlp()
        
    def create_mlp(self):
        model = Sequential([
                Flatten(input_shape=self.input_shape),
                Dense(self.hidden_layers_neuron_list[0], input_shape=self.input_shape, activation='relu'),
                Dense(self.hidden_layers_neuron_list[1], activation='relu'),
                Dense(self.num_classes)
        ])
        return model
    
    def get_uncompiled_model(self):
        return self.model
    
    def get_compiled_model(self, optimizer, loss_fn, metrics ):
        compiled_model = self.model
        compiled_model.compile(optimizer, loss_fn, metrics)
        return compiled_model

    
