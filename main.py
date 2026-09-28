# ==========================================
# WEEK 5 - STATISTICS & PROBABILITY
# Statistics & ML Foundations
# ==========================================

import numpy as np
import pandas as pd
from scipy import stats

print("=" * 60)
print("WEEK 5 - STATISTICS & PROBABILITY FOR MACHINE LEARNING")
print("=" * 60)


# -------------------------------------------------
# 1. SAMPLE DATASET
# -------------------------------------------------

data = {
    "Group": ["A", "A", "A", "A", "A",
              "B", "B", "B", "B", "B"],
    "Sales": [120, 125, 130, 128, 127,
              135, 140, 138, 142, 145]
}

df = pd.DataFrame(data)

print("\n1. BUSINESS DATASET")
print("-" * 40)
print(df)


# -------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# -------------------------------------------------

sales = df["Sales"]

mean = np.mean(sales)
median = np.median(sales)
mode = stats.mode(sales, keepdims=True).mode[0]
variance = np.var(sales)
std_dev = np.std(sales)

print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 40)
print(f"Mean              : {mean:.2f}")
print(f"Median            : {median:.2f}")
print(f"Mode              : {mode:.2f}")
print(f"Variance          : {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")


# -------------------------------------------------
# 3. PROBABILITY
# -------------------------------------------------

print("\n3. PROBABILITY")
print("-" * 40)

total_customers = 100
successful_customers = 70

probability_success = successful_customers / total_customers

print(f"Total customers       : {total_customers}")
print(f"Successful customers  : {successful_customers}")
print(f"Probability of success: {probability_success:.2f}")


# -------------------------------------------------
# 4. NORMAL DISTRIBUTION
# -------------------------------------------------

print("\n4. NORMAL DISTRIBUTION")
print("-" * 40)

mean_sales = np.mean(sales)
std_sales = np.std(sales)

value = 135

z_score = (value - mean_sales) / std_sales

print(f"Mean       : {mean_sales:.2f}")
print(f"Std. Dev.  : {std_sales:.2f}")
print(f"Value      : {value}")
print(f"Z-score    : {z_score:.2f}")


# -------------------------------------------------
# 5. BINOMIAL DISTRIBUTION
# -------------------------------------------------

print("\n5. BINOMIAL DISTRIBUTION")
print("-" * 40)

n = 10
p = 0.7
k = 7

binomial_probability = stats.binom.pmf(k, n, p)

print(f"Number of trials       : {n}")
print(f"Probability of success : {p}")
print(f"Required successes     : {k}")
print(f"Probability            : {binomial_probability:.4f}")


# -------------------------------------------------
# 6. HYPOTHESIS TESTING - A/B TEST
# -------------------------------------------------

print("\n6. HYPOTHESIS TESTING - A/B TEST")
print("-" * 40)

group_a = df[df["Group"] == "A"]["Sales"]
group_b = df[df["Group"] == "B"]["Sales"]

t_stat, p_value = stats.ttest_ind(group_a, group_b)

print(f"Group A Mean : {group_a.mean():.2f}")
print(f"Group B Mean : {group_b.mean():.2f}")
print(f"T-statistic  : {t_stat:.4f}")
print(f"P-value      : {p_value:.4f}")

alpha = 0.05

if p_value < alpha:
    print("Result       : Statistically significant difference.")
else:
    print("Result       : No statistically significant difference.")


# -------------------------------------------------
# 7. CONFIDENCE INTERVAL
# -------------------------------------------------

print("\n7. 95% CONFIDENCE INTERVAL")
print("-" * 40)

confidence = 0.95
sample_mean = np.mean(sales)
standard_error = stats.sem(sales)

confidence_interval = stats.t.interval(
    confidence,
    len(sales) - 1,
    loc=sample_mean,
    scale=standard_error
)

print(f"Sample Mean : {sample_mean:.2f}")
print(f"95% CI      : ({confidence_interval[0]:.2f}, "
      f"{confidence_interval[1]:.2f})")


# -------------------------------------------------
# 8. CORRELATION
# -------------------------------------------------

print("\n8. CORRELATION")
print("-" * 40)

advertising = [10, 20, 30, 40, 50]
sales_data = [20, 35, 45, 60, 75]

correlation, correlation_p = stats.pearsonr(
    advertising,
    sales_data
)

print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value               : {correlation_p:.4f}")

if correlation > 0:
    print("Interpretation        : Positive correlation.")
elif correlation < 0:
    print("Interpretation        : Negative correlation.")
else:
    print("Interpretation        : No correlation.")


# -------------------------------------------------
# 9. FINAL BUSINESS CONCLUSION
# -------------------------------------------------

print("\n9. BUSINESS CONCLUSION")
print("-" * 40)

print(
    "The analysis shows that Group B has a higher average "
    "sales value than Group A. Statistical testing was used "
    "to check whether the difference between the two groups "
    "is significant. Descriptive statistics, probability, "
    "distribution analysis, confidence interval and correlation "
    "were also calculated."
)

print("\n" + "=" * 60)
print("WEEK 5 STATISTICS PROJECT COMPLETED")
print("=" * 60)
# -------------------------------------------------
# 10. DATA VISUALIZATION
# -------------------------------------------------

import matplotlib.pyplot as plt

print("\n10. DATA VISUALIZATION")
print("-" * 40)

# Create visualizations folder
import os
os.makedirs("visualizations", exist_ok=True)


# 1. Sales Comparison Bar Chart
plt.figure(figsize=(8, 5))

group_means = df.groupby("Group")["Sales"].mean()

plt.bar(group_means.index, group_means.values)

plt.title("Average Sales by Group")
plt.xlabel("Group")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.savefig("visualizations/average_sales_by_group.png")
plt.show()


# 2. Sales Distribution Histogram
plt.figure(figsize=(8, 5))

plt.hist(sales, bins=5, edgecolor="black")

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("visualizations/sales_distribution.png")
plt.show()


# 3. Advertising vs Sales Scatter Plot
plt.figure(figsize=(8, 5))

plt.scatter(advertising, sales_data)

plt.title("Advertising vs Sales")
plt.xlabel("Advertising")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("visualizations/advertising_vs_sales.png")
plt.show()


# 4. Box Plot of Sales
plt.figure(figsize=(8, 5))

plt.boxplot(sales)

plt.title("Sales Box Plot")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("visualizations/sales_boxplot.png")
plt.show()


print("All graphs have been created successfully.")
print("Graphs are saved inside the 'visualizations' folder.")


# -------------------------------------------------
# FINAL PROJECT SUMMARY
# -------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("""
This Week 5 project analyzed business sales data using
statistics and probability concepts.

The project included:
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Probability
- Normal distribution
- Binomial distribution
- Hypothesis testing
- P-value
- Confidence interval
- Correlation
- Data visualization

The analysis helps understand sales performance and
relationships between business variables.
""")

print("=" * 60)
print("WEEK 5 PROJECT FULLY COMPLETED")
print("=" * 60)