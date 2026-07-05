# Probability, Bayesian Inference, and Gradient Descent

## Group Members

Isimbi Nelly Assoumpta
Sarah Gasaro
Divine Kuzo

## Project Overview

This project covers four core Machine Learning concepts, implemented from scratch in Python:

1. Expectation-Maximization (EM) Algorithm
2. Bayesian Probability using IMDb Movie Reviews
3. Manual Gradient Descent Calculations
4. Gradient Descent in Python

The goal was to understand the math behind these algorithms rather than just calling a library function.

## EM Algorithm

We separated a dataset of parent and child heights into two Gaussian distributions, without using any labels. This included writing the E-step, M-step, log-likelihood calculation, and the full optimization loop from scratch, then using the trained model to classify a new height and give posterior probabilities.

## Bayesian Probability

Using the IMDb reviews dataset (50,000 reviews, half positive and half negative), we calculated P(Positive | keyword) for six chosen words.

Positive keywords: wonderful, excellent, brilliant
Negative keywords: boring, awful, waste

| Keyword | Prior | Likelihood | Marginal | Posterior |
|---|---|---|---|---|
| wonderful | 0.5000 | 0.1066 | 0.0650 | 0.8203 |
| excellent | 0.5000 | 0.1174 | 0.0725 | 0.8099 |
| brilliant | 0.5000 | 0.0755 | 0.0489 | 0.7711 |
| boring | 0.5000 | 0.0247 | 0.0623 | 0.1983 |
| awful | 0.5000 | 0.0136 | 0.0624 | 0.1093 |
| waste | 0.5000 | 0.0145 | 0.0731 | 0.0993 |

Positive words gave a posterior around 77-82%, while negative words gave a posterior around 10-20%. This confirms the keywords chosen actually carry sentiment signal. Everything was done with plain Python (no external libraries), using the built-in csv module.

## Manual Gradient Descent

We manually worked through three gradient descent updates for a small linear regression model, using matrix multiplication rather than treating values as scalars.

Starting values: m = [-1, 2], b = [1, 1], learning rate = 0.01

Across the three iterations, the error dropped sharply and m and b started to stabilize, showing the parameters were moving in the right direction to reduce error.

## Gradient Descent in Code

We converted the manual calculations into Python, using a small custom derivative function (since scipy.misc.derivative has been removed in newer SciPy versions) so every calculation step stays visible instead of being hidden inside a library call. Matplotlib was used to plot how m, b, and the error changed across iterations.

## Technologies Used

Python, NumPy, Matplotlib

## Running the Code

Part 2:

cd part2_bayes
python bayes_sentiment.py

Part 4:
cd part3_4_gradient_descent
python gradient_descent.py

## What We Learned

Gaussian distributions and mixture models, the EM algorithm, Bayes' theorem, matrix multiplication, mean squared error, and gradient descent as an optimization method.
