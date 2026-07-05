import pandas as pd
import numpy as np

# ---- LOAD DATA ----
data = pd.read_csv("GaltonFamilies.csv")
mother_heights = data["mother"].values
child_heights = data["childHeight"].values
all_heights = np.concatenate([mother_heights, child_heights])

print("Total combined data points:", len(all_heights))

# ---- HELPER FUNCTIONS ----

def normal_pdf(x, mu, sigma):
    """How likely is height x, if it came from a bell curve centered at mu with spread sigma."""
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent

def compute_log_likelihood(x, mu1, sigma1, pi1, mu2, sigma2, pi2):
    """A single score for how well our two bell curves fit ALL the data."""
    likelihood = pi1 * normal_pdf(x, mu1, sigma1) + pi2 * normal_pdf(x, mu2, sigma2)
    return np.sum(np.log(likelihood))

def e_step(x, mu1, sigma1, pi1, mu2, sigma2, pi2):
    """Calculate the probability each point belongs to group 1 vs group 2."""
    likelihood1 = normal_pdf(x, mu1, sigma1)
    likelihood2 = normal_pdf(x, mu2, sigma2)
    weighted1 = pi1 * likelihood1
    weighted2 = pi2 * likelihood2
    total = weighted1 + weighted2
    responsibility1 = weighted1 / total
    responsibility2 = weighted2 / total
    return responsibility1, responsibility2

def m_step(x, responsibility1, responsibility2):
    """Use the responsibilities to compute better mu, sigma, pi."""
    N1 = np.sum(responsibility1)
    N2 = np.sum(responsibility2)
    mu1 = np.sum(responsibility1 * x) / N1
    mu2 = np.sum(responsibility2 * x) / N2
    sigma1 = np.sqrt(np.sum(responsibility1 * (x - mu1) ** 2) / N1)
    sigma2 = np.sqrt(np.sum(responsibility2 * (x - mu2) ** 2) / N2)
    pi1 = N1 / len(x)
    pi2 = N2 / len(x)
    return mu1, sigma1, pi1, mu2, sigma2, pi2

# ---- INITIAL GUESSES (Iteration 0) ----
mu1, sigma1, pi1 = 60.0, 5.0, 0.5   # "kids" group
mu2, sigma2, pi2 = 68.0, 5.0, 0.5   # "moms" group

# ---- RUN THE EM LOOP ----
num_iterations = 20   # run enough times for it to settle down

print("\nIteration | mu1 (kids) | mu2 (moms) | sigma1 | sigma2 | pi1 | pi2 | Log-Likelihood")
print(f"0 (Init)  | {mu1:.3f}     | {mu2:.3f}     | {sigma1:.3f}  | {sigma2:.3f}  | {pi1:.3f} | {pi2:.3f} | -")

for i in range(1, num_iterations + 1):
    responsibility1, responsibility2 = e_step(all_heights, mu1, sigma1, pi1, mu2, sigma2, pi2)
    mu1, sigma1, pi1, mu2, sigma2, pi2 = m_step(all_heights, responsibility1, responsibility2)
    ll = compute_log_likelihood(all_heights, mu1, sigma1, pi1, mu2, sigma2, pi2)

    if i <= 2 or i == num_iterations:  # only print iteration 1, 2, and the final one (keeps output clean)
        print(f"{i}         | {mu1:.3f}     | {mu2:.3f}     | {sigma1:.3f}  | {sigma2:.3f}  | {pi1:.3f} | {pi2:.3f} | {ll:.3f}")

print("\nFinal converged values:")
print(f"Kids group   -> mean: {mu1:.2f}, std: {sigma1:.2f}, proportion: {pi1:.2%}")
print(f"Moms group   -> mean: {mu2:.2f}, std: {sigma2:.2f}, proportion: {pi2:.2%}")

# ---- LIVE DEMO: classify a random test height ----

def classify_height(height, mu1, sigma1, pi1, mu2, sigma2, pi2):
    """Given ONE height, return probability it's 'Child' vs 'Basketball Player/Pro'."""
    likelihood1 = normal_pdf(height, mu1, sigma1)
    likelihood2 = normal_pdf(height, mu2, sigma2)
    weighted1 = pi1 * likelihood1
    weighted2 = pi2 * likelihood2
    total = weighted1 + weighted2
    prob_child = weighted1 / total
    prob_pro = weighted2 / total
    return prob_child, prob_pro

# Ask the user (coach) to type in a height live during presentation
test_height = float(input("\nEnter a test height (in inches) to classify: "))

prob_child, prob_pro = classify_height(test_height, mu1, sigma1, pi1, mu2, sigma2, pi2)

print(f"\nFor height = {test_height} inches:")
print(f"P(Child)  = {prob_child:.4f} ({prob_child:.2%})")
print(f"P(Pro/Adult) = {prob_pro:.4f} ({prob_pro:.2%})")

if prob_child > prob_pro:
    print("=> Model's best guess: this is more likely a CHILD's height.")
else:
    print("=> Model's best guess: this is more likely a PRO/ADULT's height.")