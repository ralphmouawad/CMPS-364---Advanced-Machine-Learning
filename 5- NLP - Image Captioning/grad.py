import numpy as np
from random import randrange

def eval_num_grad(f, x, h=0.001, ver=False):
    f_x = f(x)  # Compute the function value at the original point.
    grad = np.zeros_like(x)  # Initialize the gradient array with the same shape as x.

    # Iterate over all indices in x using a NumPy iterator.
    iter = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not iter.finished:
        # Get the current index.
        ix = iter.multi_index
        oldval = x[ix]  # Store the original value at the current index.

        # Evaluate the function at x + h.
        x[ix] = oldval + h  # Increment the value by h.
        f_xph = f(x)  # Compute f(x + h).

        # Evaluate the function at x - h.
        x[ix] = oldval - h  # Decrement the value by h.
        f_xmh = f(x)  # Compute f(x - h).

        # Restore the original value at the current index.
        x[ix] = oldval

        # Compute the partial derivative using the centered difference formula.
        grad[ix] = (f_xph - f_xmh) / (2 * h)  # Estimate the slope.

        # Optionally print the gradient at the current index if ver is enabled.
        if ver:
            print(ix, grad[ix])

        # Move to the next dimension.
        iter.iternext()

    return grad


def eval_num_grad_array(f, x, df, h=1e-5):
    # Initialize the gradient array with the same shape as x.
    grad = np.zeros_like(x)

    # Iterate over all indices in x using a NumPy iterator.
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index  # Get the current index.

        # Store the original value of x at the current index.
        oldval = x[ix]

        # Compute f(x + h) by incrementing the current element of x by h.
        x[ix] = oldval + h
        pos = f(x).copy()  # Copy the output to avoid unintended side effects.

        # Compute f(x - h) by decrementing the current element of x by h.
        x[ix] = oldval - h
        neg = f(x).copy()  # Copy the output to avoid unintended side effects.

        # Restore the original value of x at the current index.
        x[ix] = oldval

        # Compute the gradient using the chain rule and centered difference formula.
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)

        # Move to the next index.
        it.iternext()

    # Return the computed gradient.
    return grad

def eval_num_grad_blobs(f, inputs, output, h=1e-5):
    numeric_diffs = []  # List to store the numerical gradients for each input blob.

    # Iterate over each input blob in the inputs tuple.
    for input_blob in inputs:
        # Initialize a gradient array with the same shape as the input blob's `diffs`.
        diff = np.zeros_like(input_blob.diffs)

        # Create an iterator to traverse each element of the input blob's `vals`.
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            id_x = it.multi_index  # Get the current index.
            orig = input_blob.vals[id_x]  # Store the original value at the current index.

            # Compute f(inputs + h) by adding `h` to the current element of the input blob.
            input_blob.vals[id_x] = orig + h
            f(*(inputs + (output,)))  # Call the function with modified inputs.
            pos = np.copy(output.vals)  # Copy the output blob's values.

            # Compute f(inputs - h) by subtracting `h` from the current element of the input blob.
            input_blob.vals[id_x] = orig - h
            f(*(inputs + (output,)))  # Call the function with modified inputs.
            neg = np.copy(output.vals)  # Copy the output blob's values.

            # Restore the original value of the input blob at the current index.
            input_blob.vals[id_x] = orig

            # Calculate the gradient using the centered difference formula and the chain rule.
            diff[id_x] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            # Move to the next index in the input blob.
            it.iternext()

        # Append the computed gradient for the current input blob to the list.
        numeric_diffs.append(diff)

    # Return the list of numerical gradients for all input blobs.
    return numeric_diffs

def eval_num_grad_net(net, inputs, output, h=1e-5):
    return eval_num_grad_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )

def sparse_gradient_check(f, x, anal_grad, n_checks=10, h=1e-5):

    # Loop through a specified number of random checks.
    for i in range(n_checks):
        # Randomly select an index in the input array `x`.
        ix = tuple([randrange(m) for m in x.shape])

        # Store the original value of `x` at the selected index.
        oldval = x[ix]

        # Perturb the value at index `ix` by adding `h` and compute the function value at `x + h`.
        x[ix] = oldval + h
        f_xph = f(x)  # Evaluate the function at `x + h`.

        # Perturb the value at index `ix` by subtracting `h` and compute the function value at `x - h`.
        x[ix] = oldval - h
        f_xmh = f(x)  # Evaluate the function at `x - h`.

        # Restore the original value of `x` at the selected index.
        x[ix] = oldval

        # Compute the numerical gradient using the finite difference formula.
        grad_numerical = (f_xph - f_xmh) / (2 * h)

        # Retrieve the analytic gradient at the selected index.
        grad_analytic = anal_grad[ix]

        # Compute the relative error between the numerical and analytic gradients.
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )

        # Print the numerical gradient, analytic gradient, and the relative error.
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )