import numpy as np
import grad_optimizer
from utils import *
class ImageCaptioningTrainerEngine(object):
    """
    The `ImageCaptioningTrainerEngine` class is designed to manage the entire process of training image captioning models.
    It leverages stochastic gradient descent (SGD) with various optimization strategies provided in `grad_optimizer.py`.
    This trainer handles both training and validation datasets, enabling periodic evaluations to monitor model performance and check for overfitting.
    To use the trainer, create an instance of the `ImageCaptioningTrainerEngine` class by passing the model, dataset, and essential training
    parameters (e.g., learning rate and batch size) to its constructor. You can then call the `train()` method to initiate the optimization process and train the model.
    Upon completion, the `params` attribute of the model will store the parameter values that achieved the best
    on the validation dataset during training. The trainer also records the training process in variables like `loss_history`,
    `train_acc_history`, and `val_acc_history`, which capture loss values and accuracy metrics for both the training and validation datasets at each epoch.

    Example usage:
    trainer = ImageCaptioningTrainerEngine(
        model,
        data,
        update_opt='sgd',
        opt_config={'learning_rate': 1e-3},
        lr_decay=0.05,
        n_epoch=5,
        batch_size=50,
        print_all=1000
    )

    """
    def __init__(self, model, data, **kwargs):
        """
        Creates an instance of the `ImageCaptioningTrainerEngine` class.

        ### Required Parameters:
        - **`model`**: The model object designed for image captioning, adhering to the specified API.
        - **`data`**: A dictionary containing training and validation datasets, typically loaded using `load_coco_dataset`.

        ### Optional Parameters:
        - **`update_opt`**: The name of the optimization rule to use during training (default: `'sgd'`).
        - **`opt_config`**: A dictionary specifying hyperparameters for the optimization rule, such as learning rate and momentum. Note that `'learning_rate'` is a mandatory parameter for all update rules.
        - **`lr_decay`**: A scalar factor controlling the decay of the learning rate. After each epoch, the learning rate is multiplied by this value (default: `1.0`, meaning no decay).
        - **`batch_size`**: The number of samples per training minibatch (default: `100`).
        - **`n_epoch`**: The total number of training epochs (default: `10`).
        - **`print_all`**: The frequency (in iterations) at which training loss is printed (default: every `10` iterations).
        - **`ver`**: When `True`, displays training progress and statistics; when `False`, suppresses output (default: `True`).

        ### Behavior:
        The constructor validates all provided arguments, ensuring required parameters are supplied and formatted correctly. If any arguments are missing, invalid, or unrecognized, it raises an error. It also checks for the existence of the specified optimization rule in `grad_optimizer.py` and assigns the corresponding function to the trainer.

        ### Example:
        ```python
        trainer = ImageCaptioningTrainerEngine(
            model,
            data,
            update_opt='adam',
            opt_config={'learning_rate': 1e-1},
            batch_size=32,
            n_epoch=5
        )

        """
        self.model = model  # Assign the model object to the class instance.
        self.data = data  # Store the dataset (training and validation data) for later use.

        # Unpack keyword arguments with default values
        self.update_opt = kwargs.pop("update_opt", "sgd")  # Set the optimization update rule, default is 'sgd'.
        self.opt_config = kwargs.pop("opt_config", {})  # Store the configuration for the optimizer.
        self.lr_decay = kwargs.pop("lr_decay", 1.0)  # Set the learning rate decay factor, default is 1 (no decay).
        self.batch_size = kwargs.pop("batch_size", 100)  # Set the batch size, default is 100.
        self.n_epoch = kwargs.pop("n_epoch", 10)  # Set the number of epochs, default is 10.

        self.print_all = kwargs.pop("print_all", 10)  # Set how often to print the loss, default is every 10 iterations.
        self.ver = kwargs.pop("ver", True)  # If True, prints progress, else suppresses output during training.

        # Check for any unrecognized extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))  # Format the extra arguments for the error message.
            raise ValueError("Unrecognized arguments %s" % extra)  # Raise an error if extra arguments are passed.

        # Verify if the specified update rule exists in the grad_optimizer module
        if not hasattr(grad_optimizer, self.update_opt):
            raise ValueError('Invalid update_opt "%s"' % self.update_opt)  # Raise an error if the update rule is invalid.

        # Replace the string update rule with the actual function from the grad_optimizer module
        self.update_opt = getattr(grad_optimizer, self.update_opt)

        # Initialize/reset any additional attributes or variables
        self._reset()

    def _reset(self):
        """
        Reset optimization variables and history. This method should be called automatically
        during the training process and should not be called manually.
        """
        # Initialize tracking variables for optimization progress
        self.epoch = 0  # Track the current epoch number
        self.best_val_acc = 0  # Store the best validation accuracy encountered during training
        self.best_params = {}  # Keep a record of the model parameters that achieved the best validation accuracy
        self.loss_history = []  # List to track loss values at each training step
        self.train_acc_history = []  # List to track training accuracy over time
        self.val_acc_history = []  # List to track validation accuracy over time

        # Initialize optimizer configurations for each model parameter
        self.opt_configs = {}
        for p in self.model.params:
            # Create a new dictionary for each parameter's optimization configuration
            d = {k: v for k, v in self.opt_config.items()}
            self.opt_configs[p] = d  # Store the optimizer settings for each parameter

    def _step(self):
      """
      Perform a single step of gradient update. This method is automatically invoked by
      the train() function during training and should not be called manually.

      It includes the process of sampling a minibatch, computing the loss and gradients,
      and updating the model parameters using the specified optimization algorithm.
        """
      # Sample a minibatch of training data
      minibatch = get_coco_minibatch(
          self.data, batch_size=self.batch_size, split="train"
      )
      ground_truth_captions, image_features, urls = minibatch  # Extract ground_truth_captions, image_features, and URLs from the minibatch

      # Compute the loss and gradients for the current minibatch
      loss, grads = self.model.loss(image_features, ground_truth_captions)  # Call model's loss function
      self.loss_history.append(loss)  # Record the loss value for tracking progress

      # Update each model parameter using the gradients and optimizer configuration
      for p, weight in self.model.params.items():
          dweight = grads[p]  # Get the gradient for the current parameter
          config = self.opt_configs[p]  # Retrieve the optimizer configuration for the parameter

          # Use the update rule (e.g., SGD, Adam) to compute the new parameter and configuration
          next_w, next_cell_stateonfig = self.update_opt(weight, dweight, config)

          # Update the model's parameter and optimizer configuration
          self.model.params[p] = next_w
          self.opt_configs[p] = next_cell_stateonfig


    def _step(self):
        """
        This function in not meant to be called directly and is called during the training process automatically
        Performs a single gradient update step.
        During this step, a minibatch is sampled, the loss and gradients are computed, and the model parameters are updated using the designated optimization algorithm.
        """
        # Sample a minibatch of training data
        minibatch = get_coco_minibatch(
            self.data, minibatch_size=self.batch_size, dataset_split="train"
        )
        ground_truth_captions, image_features, urls = minibatch  # Unpack the minibatch components

        # Compute the loss and gradients for the current minibatch
        loss, grads = self.model.loss(image_features, ground_truth_captions)  # Call the loss function of the model
        self.loss_history.append(loss)  # Log the loss value for progress tracking

        # Update model parameters using the computed gradients and optimization configuration
        for param_name, current_param in self.model.params.items():
            grad = grads[param_name]  # Get the gradient corresponding to the parameter
            config = self.opt_configs[param_name]  # Retrieve the optimizer configuration for the parameter

            # Apply the optimization rule to compute the updated parameter and configuration
            updated_param, updated_config = self.update_opt(current_param, grad, config)

            # Save the updated parameter and optimizer configuration
            self.model.params[param_name] = updated_param
            self.opt_configs[param_name] = updated_config



    def train(self):
        num_train_samples = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train_samples // self.batch_size, 1)
        total_iterations = self.n_epoch * iterations_per_epoch

        for iteration in range(total_iterations):
            # Perform a single optimization step
            self._step()

            # Optionally print the current training loss
            if self.ver and iteration % self.print_all == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (iteration + 1, total_iterations, self.loss_history[-1])
                )

            # Check if the current iteration marks the end of an epoch
            if (iteration + 1) % iterations_per_epoch == 0:
                self.epoch += 1  # Increment epoch counter

                # Apply learning rate decay for all optimizer configurations
                for param_name in self.opt_configs:
                    self.opt_configs[param_name]["learning_rate"] *= self.lr_decay
