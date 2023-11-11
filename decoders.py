import numpy as np
from numpy.linalg import inv as inv  # Used in kalman filter

from pearson_eval import pearson_mel

try:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass


def get_R2(y_test, y_test_pred):
    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list = []  # Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):  # Loop through outputs
        # Compute R2 for each output
        y_mean = np.mean(y_test[:, i])
        R2 = 1 - np.sum((y_test_pred[:, i] - y_test[:, i]) ** 2) / np.sum((y_test[:, i] - y_mean) ** 2)
        R2_list.append(R2)  # Append R2 of this output to the list
    R2_array = np.array(R2_list)
    return R2_array  # Return an array of R2s


class DNNDecoder(object):
    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, repsectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.loss_train, self.loss_valid = [], []
        self.cc_train, self.cc_valid = [], []
        self.r2_x_pos, self.r2_y_pos = [], []
        self.r2_x_vel, self.r2_y_vel = [], []
        self.r2_x_acc, self.r2_y_acc = [], []

        # If "units" is an integer, put it in the form of a vector
        try:  # Check if it's a vector
            units[0]
        except:  # If it's not a vector, create a vector of the number of units for each layer
            units = [units]
        self.units = units

        # Determine the number of hidden layers (based on "units" that the user entered)
        self.num_layers = len(units)

    def fit(self, X_flat_train, y_train, valid_data=None, logger=None):

        """
        Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add first hidden layer
        model.add(Dense(self.units[0], input_dim=X_flat_train.shape[1]))  # Add dense layer
        model.add(Activation('relu'))  # Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout != 0: model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0

        # Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers - 1):  # Loop through additional layers
            model.add(Dense(self.units[layer + 1]))  # Add dense layer
            model.add(Activation('relu'))  # Add nonlinear (tanh) activation
            if self.dropout != 0: model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0

        # Add dense connections to all outputs
        model.add(Dense(y_train.shape[1]))  # Add final dense layer (connected to outputs)

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # Set loss function and optimizer

        for i in range(self.num_epochs):
            logger.info(f'----- Epochs: [{i + 1}/{self.num_epochs}] -----')

            history = model.fit(X_flat_train, y_train, epochs=1, verbose=self.verbose,
                                validation_data=valid_data)  # Fit the model
            self.model = model

            self.loss_train.append(history.history['loss'][0])
            self.loss_valid.append(history.history['val_loss'][0])

            y_train_pred = self.predict(X_flat_train)
            y_valid_pred = self.predict(valid_data[0])
            cc_train, _ = pearson_mel(y_train, y_train_pred)
            cc_valid, _ = pearson_mel(valid_data[1], y_valid_pred)
            self.cc_train.append(cc_train)
            self.cc_valid.append(cc_valid)

            r2 = get_R2(valid_data[1], y_valid_pred)
            self.r2_x_pos.append(r2[0])
            self.r2_y_pos.append(r2[1])
            self.r2_x_vel.append(r2[2])
            self.r2_y_vel.append(r2[3])
            self.r2_x_acc.append(r2[4])
            self.r2_y_acc.append(r2[5])
            logger.info('----- Epochs [{} / {}], cc_train: {:.4f}, cc_valid: {:.4f}, r2_y_vel: {:.4f} -----'
                        .format(i + 1, self.num_epochs, self.cc_train[i], self.cc_valid[i], self.r2_y_vel[i]))

    def predict(self, X_flat_test):

        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted


class KFDecoder(object):
    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003
    (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    except the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris
    (https://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X_kf_train, y_train):
        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        # First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]
        A = X2 * X1.T * inv(X1 * X1.T)  # Transition matrix
        W = (X2 - A * X1) * (X2 - A * X1).T / (
                nt - 1) / self.C  # Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix
        params = [A, W, H, Q]
        self.model = params

    def predict(self, X_kf_test, y_test):
        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        # Extract parameters
        A, W, H, Q = self.model

        # # Q = Gaussian noise
        # Q = np.matrix(np.random.normal(0, 100, Q.shape))

        # # W = Gaussian noise
        # W = np.matrix(np.random.normal(0, 100, W.shape))

        # First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)

        # Initializations
        num_states = X.shape[0]  # Dimensionality of the state
        states = np.empty(X.shape)  # Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states, num_states]))
        P = np.matrix(np.zeros([num_states, num_states]))
        # # P
        # diagonal_elements = 100 * np.ones(num_states)
        # P = np.diag(diagonal_elements)
        # P = np.matrix(np.random.normal(0, 100, P.shape))

        state = X[:, 0]  # Initial state
        states[:, 0] = np.copy(np.squeeze(state))

        # Get predicted state for every time bin
        for t in range(X.shape[1] - 1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W
            state_m = A * state

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * inv(H * P_m * H.T + Q)  # Calculate Kalman gain
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (Z[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(state)  # Record state at the timestep

            #print(t)
        y_test_predicted = states.T
        return y_test_predicted


class LSTMDecoder(object):
    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.loss_train, self.loss_valid = [], []
        self.cc_train, self.cc_valid = [], []
        self.r2_x_pos, self.r2_y_pos = [], []
        self.r2_x_vel, self.r2_y_vel = [], []
        self.r2_x_acc, self.r2_y_acc = [], []

    def fit(self, X_train, y_train, valid_data=None, logger=None):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add recurrent layer
        model.add(LSTM(self.units, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=self.dropout,
                       recurrent_dropout=self.dropout))  # Within recurrent layer, include dropout
        if self.dropout != 0: model.add(Dropout(self.dropout))  # Dropout some units (recurrent layer output units)

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])  # Set loss function and optimizer
        for i in range(self.num_epochs):
            logger.info(f'----- Epochs: [{i + 1}/{self.num_epochs}] -----')

            history = model.fit(X_train, y_train, epochs=1, verbose=self.verbose,
                                validation_data=valid_data)  # Fit the model
            self.model = model

            self.loss_train.append(history.history['loss'][0])
            self.loss_valid.append(history.history['val_loss'][0])

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(valid_data[0])
            cc_train, _ = pearson_mel(y_train, y_train_pred)
            cc_valid, _ = pearson_mel(valid_data[1], y_valid_pred)
            self.cc_train.append(cc_train)
            self.cc_valid.append(cc_valid)

            r2 = get_R2(valid_data[1], y_valid_pred)
            self.r2_x_pos.append(r2[0])
            self.r2_y_pos.append(r2[1])
            self.r2_x_vel.append(r2[2])
            self.r2_y_vel.append(r2[3])
            self.r2_x_acc.append(r2[4])
            self.r2_y_acc.append(r2[5])
            logger.info('----- Epochs [{} / {}], cc_train: {:.4f}, cc_valid: {:.4f}, r2_y_vel: {:.4f} -----'
                        .format(i + 1, self.num_epochs, self.cc_train[i], self.cc_valid[i], self.r2_y_vel[i]))

    def predict(self, X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted
