# RegressionCloud
---
**Created by:** Julian David Castiblanco Real
---
## Description
---
This laboratory explores the modeling of stellar luminosity through linear and polynomial regression techniques developed from first principles, without the use of pre-built machine learning libraries. The objective is to gain a deeper understanding of how prediction models are constructed by explicitly defining the hypothesis function, the cost function, and the gradient-based optimization process, using observational stellar data as a case study.

The work is conducted within the framework of a Machine Learning bootcamp and emphasizes the importance of understanding model construction and execution in cloud-based and enterprise-level systems, where machine learning is increasingly treated as a core architectural capability rather than a black-box tool.

**Prerequisitos:**
- Python 3.8 or higher (As a recomendation, the newer versions are better due to the compatibility with the creations of virtual environments in multiple IDEs)
- Jupyter Notebook or JupyterLab
- libraries:
  - ``numpy``
  - ``matplotlib``
 
**Execution:**
1. Clone Repository:
   ```
   git clone <Repository_URL>
   cd <Repository_name>
   ```

2. Setup the virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

3. Start running each block of code so you can see the results
---
### Part 1
For this part of the lab, we implement a simple linear regression model to study the relationship between stellar mass (M) and stellar luminosity (L) using a single input feature.
The model assumes that luminosity can be approximated as a linear function of mass, expressed as:

$$
\hat{L} = wM + b
$$

In this formulation, \(M\) denotes the stellar mass, \(w\) corresponds to the slope of the model, and \(b\) represents the intercept.
Model performance is evaluated using the Mean Squared Error (MSE), defined as:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{L}^{(i)} - L^{(i)})^2
$$

The gradients of the loss function with respect to both parameters were derived analytically and used to iteratively update the model through gradient descent. Two implementations were explored: a loop-based version for conceptual clarity and a vectorized version for computational efficiency.

Different learning rates were evaluated to analyze their effect on training stability and convergence speed. In addition, visualizing the cost function over a range of parameter values provided insight into the structure of the optimization problem.

Overall, the linear model successfully captures the broad increasing trend between mass and luminosity but fails to represent the rapid growth observed at higher masses. The convex shape of the cost surface confirms the presence of a single global optimum, while the learning rate experiments demonstrate the importance of proper hyperparameter selection for effective training.

### Results
This notebook implements linear regression from first principles to model stellar luminosity as a function of stellar mass. It includes dataset visualization, explicit definition of the linear model and Mean Squared Error (MSE) loss, analytical gradient derivation, and gradient descent optimization using both loop-based and vectorized implementations.

The analysis explores the effect of different learning rates on convergence behavior and visualizes the cost surface to illustrate the convex optimization landscape. While the model captures the general increasing trend between mass and luminosity, the results show clear underfitting at higher stellar masses due to the linear assumption.

To view the full implementation and results, see the notebook at the following link:
[Open Notebook 1](01_part1_linreg_1feature.ipynb)

---
### Part 2