**Comparing Different gradient optimizer on two layers’ MLP: SGD, Momentum and Adam**

Three-person team | Four Weeks | NumPy from scratch,sklearn/PyTorch as reference implementation

**Implementation**

- MLP regressor with two hidden layers to predict median house value from other features \[5\]  
- Feature standardization, mini-batch gradient descent equipped with customizable learning rate optimizers(SGD, SGD with momentum, AdaGrad(optional)\[4\], RMSProp(optional), Adam \[3\].）  
- Numerically stable formulations: float32 overflow guards on second-moment accumulators, audited bias-correction time step.


  
**Datasets**

- California\_housing: 20640 × 8, regression data \[1\]

**Evaluation**

- Fixed 70/15/15 train/validation/test splits, with final predictive metrics.  
- Identical initialization, batch composition, loss function across all optimizers.  
- The learning rate is set to a logarithmic grid to test the valid range for the initialization of learning rate of different optimizers.  
- Feature scale sensitivity: measure the degradation without feature standardization to see to what extent the optimizer can replace feature standardization.  
- Gradient descent step plot: loss vs. step and loss vs. time  
- Cost: compare seconds per step relative to SGD.  
- Report divergent configurations. Test all optimizers with multiple different random seeds to reduce the effect of seed noise.  
- Compare against the linear regression baseline and scikit-learns LinearRegression \[2\]; MLP outputs are sanity-checked against MLPRegressor / PyTorch nn.Sequential  
- 

**Tests and Quality**

- Finite-difference gradient checks: check the correctness of the MSE loss on every layer of MLP.  
- Single step gradient check for Adam: the parameter displacement for the first step should approximately equal the learning rate.  
- Model degradation equivalence: Adam with beta1=0 and without bias correction degrade to RMSProp, SGD with momentum degrade to SGD when momentum parameter \= 0\.  
- Check the value step by step comparing with pytorch.optim

**Team Plan**

Shaun Jin: optimizer family, base-class interface, update-rule derivations.  
Zhiyuan Zhang: MLP model layer, numerical stability, gradient checks.  
Fahim Mosharrof Ratul : experiment harness, sweeps, figures, datasets, documentation.

**Milestones and Deliverables**

**W1:**  
Implement base class of optimizers, implement SGD and momentum; MLP forward/backward; pass gradient check.

**W2:**   
Implement Adam,AdaGrad(optional), RMSProp(optional); equivalence(between optimizers) and parity at 1e-6(to torch.optim)  
End-to-end training of MLP with SGD, momentum, and Adam

**W3:**   
learning-rate and scale sweeps, all figures. 

**W4:**   
multi-seed reruns, full suite, report.

**Reference Implementation**

\[1\] [scikit-learn California Housing loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
\[2\] [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
\[3\] Kingma & Ba, Adam: A Method for Stochastic Optimization (2015)  
\[4\] Duchi, Hazan & Singer, Adaptive Subgradient Methods (AdaGrad, 2011\)  
\[5\] Rumelhart, Hinton & Williams, Learning representations by back-propagating errors, Nature(1986)

Resources: DON’T POST THIS PART

Useful datasets/links to them:  
[https\://claude.ai/artifact/GcRmXTT53SCgdzD4i3A5j5](https://claude.ai/artifact/GcRmXTT53SCgdzD4i3A5j5)

- RMSE, MAE  
- Compare weights, coefficients, predicted probabilities against baseline prediction mean, scikit-learn’s LinearRegression \[3\] & LogisticRegression \[4\]

