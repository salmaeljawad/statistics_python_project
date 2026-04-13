import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from scipy import stats
from sklearn.linear_model import LinearRegression


########### Data

grades = [50, 60, 70, 80, 90, 100, 65, 75, 85]


############  Descriptive Statistics

print("Mean:", np.mean(grades))
print("Median:", np.median(grades))
print("Mode:", mode(grades))
print("Range:", max(grades) - min(grades))
print("Variance:", np.var(grades))
print("Standard Deviation:", np.std(grades))


############ Probability

passed = [g for g in grades if g >= 60]
prob_pass = len(passed) / len(grades)
print("Probability of Passing:", prob_pass)

############# Distribution

plt.hist(grades, bins=5)
plt.title("Grades Distribution")
plt.show()


#########  Sampling

sample = np.random.choice(grades, 5)
print("Sample:", sample)
print("Sample Mean:", np.mean(sample))


########## Central Limit Theorem

sample_means = []

for _ in range(500):
    sample = np.random.choice(grades, 5)
    sample_means.append(np.mean(sample))

plt.hist(sample_means)
plt.title("CLT - Sample Means Distribution")
plt.show()


#########  Confidence Interval

mean = np.mean(grades)
std = np.std(grades)

ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(grades)))
print("Confidence Interval:", ci)

#######  Hypothesis Testing

class_A = np.random.normal(70, 10, 30)
class_B = np.random.normal(75, 10, 30)

t_stat, p_value = stats.ttest_ind(class_A, class_B)

print("T-statistic:", t_stat)
print("P-value:", p_value)


######### Regression

hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
scores = np.array([50, 55, 65, 70, 75])

model = LinearRegression()
model.fit(hours, scores)

prediction = model.predict([[6]])
print("Predicted score for 6 hours:", prediction)