**Step 1: Shared Conventions and Understanding**

1. **Parameter format.** How parameters and gradients are stored, e.g. a dict `name → np.ndarray` with gradients in the same keys and shapes. Also whether optimizers update in place.  
2. **Optimizer API.** The exact method signatures, e.g. `step(params, grads)` and `reset()`, and whether you match PyTorch's formulas exactly (momentum convention, epsilon placement).  
3. **Model API.** The method signatures, e.g. `loss_and_grads(X, y) → (loss, grads)`, `predict(X)`, `init(seed)`. Plus the hidden layer sizes, activation function, and initialization scheme.  
4. **Loss definition.** Mean squared error with or without the ½ factor. PyTorch's `MSELoss` has no ½.  
5. **Data API.** The signatures, e.g. `get_splits(seed, standardize)` and `batches(X, y, batch_size, rng)`, and whether the target is standardized too.  
6. **Dtype policy.** float32 for experiments, float64 for gradient checks and parity tests.  
7. **Seed policy.** Separate random generators for weight initialization and for batch shuffling, and how seeds are passed around.  
8. **Log format.** The fields recorded per training step, e.g. step, loss, elapsed time, and a diverged flag.

**Step 2: Implementation**

**Person A (optimizers):**

1. Write the base class.  
2. Write the torch parity harness *before* the optimizers, from PyTorch's docs. Writing the test first gets you most of the benefit of having a separate test-writer.  
3. Implement SGD and momentum, then their parity and μ=0 equivalence tests.  
4. Implement Adam, then its parity test, first-step test, and the RMSProp equivalence test against `torch.optim.RMSprop`.  
5. Take over the baselines from Person C: closed-form or sklearn linear regression, and a sklearn `MLPRegressor` run to establish the RMSE ballpark.

**Person B (model):**

1. Write initialization, forward pass, and MSE loss, using the ½ convention agreed in the contract.  
2. Write the backward pass.  
3. Run finite-difference gradient checks in float64.  
4. Check parity against `nn.Sequential`: outputs, loss, and every gradient.  
5. Write a PyTorch end-to-end training script with the same architecture, initialization, and batches. You'll need it for the integration trajectory test, and B already owns the PyTorch model.

**Person C (data \+ harness):**

1. Write the data loading, the 70/15/15 split, and train-fit standardization, plus the unstandardized variant.  
2. Write the seeded batch iterator, with a determinism test: same seed, same batches.  
3. Write the training loop using the stand-in linear model and plain SGD, logging per-step loss, wall-clock time, and validation metrics.  
4. Add divergence detection.  
5. Set up the results storage format.

