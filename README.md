# Probability, Bayesian Inference, and Gradient Descent

## Group Members

| Name | Contribution |
|---|---|
| Isimbi Nelly Assoumpta | Part 1&3 |
| Sarah Gasaro | Part 2&3 |
| Divine Kuzo | Part 4&3|

---

# Project Overview

This project applies core probabilistic Machine Learning concepts, built from scratch in Python rather than through pre-built libraries.

The work is split into four parts:

1. Expectation-Maximization (EM) Algorithm
2. Bayesian Probability using IMDb Movie Reviews
3. Manual Gradient Descent Calculations
4. Gradient Descent Implementation in Python

The focus throughout was understanding how each algorithm actually works internally, rather than treating it as a black box.

---

# Project Structure

```
ML-formative3-Team18/
│
├── data/
│   ├── GaltonFamilies.csv
│   └── IMDB Dataset.csv
│
├── notebooks/
│   ├── Merged_notebook.ipynb
│   └── Part3_Manual_Calculations.pdf
│
├── src/
│   ├── em_heights.py
│   ├── bayes_sentiment.py
│   └── gradient_descent.py
│
├── submission/
│   └── contribution_tracker.pdf
│
└── README.md
```

---

# Part 1 – Expectation Maximization (EM)

## Objective

Separate a mixed, unlabeled dataset of Mother and Child heights into two underlying Gaussian distributions.

## Implementation

Built entirely from scratch:

- Gaussian Probability Density Function
- Expectation Step (soft responsibilities)
- Maximization Step (parameter updates)
- Log-Likelihood tracking
- Full EM optimization loop
- Posterior-based classification of a new height

No machine learning libraries were used for the core algorithm.

## Outputs

- Histogram of the mixed heights with fitted Gaussian curves overlaid
- Tracking table of μ, σ², π, and log-likelihood for the initial state and first two iterations
- Live classification of a test height into Child vs. Adult, with exact posterior probabilities

---

# Part 2 – Bayesian Probability

## Objective

Use Bayes' Theorem to calculate P(Positive | Keyword) on the IMDb Movie Reviews dataset (50,000 reviews, evenly split between positive and negative).

## Keywords

Positive-signal keywords:
- wonderful
- excellent
- brilliant

Negative-signal keywords:
- boring
- awful
- waste

## Probabilities Computed

For every keyword:
- Prior: P(Positive)
- Likelihood: P(keyword | Positive)
- Marginal: P(keyword)
- Posterior: P(Positive | keyword)

| Keyword | Prior | Likelihood | Marginal | Posterior |
|---|---|---|---|---|
| wonderful | 0.5000 | 0.1066 | 0.0650 | 0.8203 |
| excellent | 0.5000 | 0.1174 | 0.0725 | 0.8099 |
| brilliant | 0.5000 | 0.0755 | 0.0489 | 0.7711 |
| boring | 0.5000 | 0.0247 | 0.0623 | 0.1983 |
| awful | 0.5000 | 0.0136 | 0.0624 | 0.1093 |
| waste | 0.5000 | 0.0145 | 0.0731 | 0.0993 |

Positive-signal words carry a 77–82% posterior of appearing in a positive review, while negative-signal words carry only 10–20% — a clear, well-justified split.

Bayes' Theorem was implemented using basic Python only (the built-in csv module), with no external ML libraries.

---

# Part 3 – Manual Gradient Descent

## Objective

Manually compute three Gradient Descent updates for a two-feature linear regression model.

## Manual Calculations

For each iteration, the following was worked out by hand:

- Predictions (via matrix multiplication, not scalar values)
- Errors
- Mean Squared Error
- Gradient of MSE, with the chain rule shown explicitly
- Updated m₁, m₂, and b

Starting values: m = [-1, 2], b = 1, learning rate = 0.01

Result across three iterations: cost dropped from 61 → 6.845 → 2.724, while m and b progressively stabilized rather than diverging — confirming the parameters were moving in a direction that reduces error.

---

# Part 4 – Gradient Descent in Python

## Objective

Convert the Part 3 manual calculations into working code.

## Implementation

The program performs:

- Matrix-based prediction (X @ m + b)
- Cost computation (MSE)
- Gradient computation, using a custom derivative function
- Parameter updates
- Error tracking across iterations

A custom finite-difference derivative function was used in place of scipy.misc.derivative, since it has been removed from recent SciPy versions — this also keeps every calculation step visible rather than hidden inside a library call.

Matplotlib was used to visualize:

- Changes in m and b over iterations
- Error reduction over iterations

The final code output (m = [-1.3696, 1.0952], b = 0.9362) matches the Part 3 manual calculations exactly.

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- csv (built-in)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/s-gasaro/ML-formative3-Team18.git
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# Running the Project

Part 1:
```bash
python em_heights.py
```

Part 2:
```bash
python bayes_sentiment.py
```

Part 4:
```bash
python gradient_descent.py
```

---

# Results

The EM algorithm successfully separates the mixed height data into two distinct Gaussian distributions corresponding to Children and Adults.

Bayesian inference clearly distinguishes positive-signal from negative-signal keywords based on their posterior probabilities.

Gradient Descent demonstrates convergence, with the cost function dropping from 61 to under 3 across three iterations.

---

# Learning Outcomes

Through this project we gained hands-on understanding of:

- Gaussian distributions and mixture models
- Expectation-Maximization
- Bayesian probability and Bayes' Theorem
- Matrix multiplication
- Mean Squared Error
- Gradient Descent and numerical optimization

---

# References

- Galton Families Dataset
- IMDb Movie Reviews Dataset
- NumPy Documentation
- Matplotlib Documentation
