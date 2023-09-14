
class drrpw_net(nn.Module):
    """End-to-end Dist. Robust RP with Wasserstein Distance learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, learnT=False, learnDelta=True, set_seed=None, cache_path='cache/'):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        drrpw_net: nn.Module object 
        """
        super(drrpw_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        self.trainT = learnT
        self.trainDelta = learnDelta

        # Prediction loss function
        # if pred_loss_factor is not None:
        #     self.pred_loss_factor = pred_loss_factor
        #     self.pred_loss = torch.nn.MSELoss()
        # else:
        #     self.pred_loss = None
        
        self.pred_loss = None

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Record the model design: nominal, base or DRO
        # Register 'delta' (ambiguity sizing parameter) for DR layer
        if self.trainDelta:
            ub = (1 - 1/n_obs) / 2
            lb = (1 - 1/n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = True
            self.delta_init = self.delta.item()

        self.model_type = 'dro'

        if self.trainT:
            Sigma_k = torch.rand(self.n_y, self.n_y)
            Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
            Sigma_k.add_(torch.eye(self.n_y))
            
            self.T = nn.Parameter(Sigma_k)
            self.T.requires_grad = True
            self.delta_init = 2

        # self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """
        Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Multiple predictions Y_hat from X
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        y_hat = Y_hat[-1]

        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        # solver_args = {'solve_method': 'ECOS', 'max_iters': 2000000, 'abstol': 1e-7}

        solver_args = {'solve_method': 'SCS'}

        # Covariance Matrix
        Q = np.cov(Y_hat.cpu().detach().numpy(), rowvar=False)


        # Optimization Layer
        # self.opt_layer = drrpw_nominal(n_y, n_obs, Q)

        # Optimize z per scenario
        # Determine whether nominal or dro model

        param = None
        if self.trainT:
            param = self.T
            self.opt_layer = drrpw_nominal_learnT(n_y, n_obs, Q)
            d = 0
            
        
        if self.trainDelta:
            param = self.delta
            self.opt_layer = drrpw_nominal_learnDelta(n_y, n_obs, Q)
            d = 1
        z_star, _ = self.opt_layer(param, solver_args=solver_args)

        softmax = torch.nn.Softmax(dim=d)
        z_star = softmax(z_star)
        
        # z_star = np.divide(z_star, np.sum(z_star))
        
        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad()
            
            for t, (x, y, y_perf) in enumerate(train_set):
                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                # print('---z_star---')
                # print(z_star)
                # print('---y_perf---')
                # print(y_perf)
                loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                
                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    print('Clamping')
                    param.data.clamp_(min=0.0001, max=0.9999)
                    print(param.item())

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0,'ones', 1.0)

                        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_path+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store trained gamma and delta values 
        self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                X_train.insert(0,'ones', 1.0)

                X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
            
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            self.delta_trained.append(self.delta.item())
            # self.delta_trained.append(2)

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]
