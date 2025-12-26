import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Dataset

print("First 5 rows:")
print(df.head())

# Rename columns if needed
# (adjust names based on dataset)
df = df.rename(columns={
    "job_category": "Gig_Category",
    "experience_level": "Experience_Level",
    "project_duration": "Completion_Time_Days"
})

# Remove missing values
df = df.dropna(subset=["Completion_Time_Days"])

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df["Completion_Time_Days"].describe())

# Box Plot – Gig Category
plt.figure()
sns.boxplot(
    x="Gig_Category",
    y="Completion_Time_Days",
    data=df
)
plt.title("Completion Time by Gig Category")
plt.xlabel("Gig Category")
plt.ylabel("Completion Time (Days)")
plt.show()

# Box Plot – Experience Level
plt.figure()
sns.boxplot(
    x="Experience_Level",
    y="Completion_Time_Days",
    data=df
)
plt.title("Completion Time by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Completion Time (Days)")
plt.show()

# Mean Comparison
mean_times = df.groupby("Experience_Level")["Completion_Time_Days"].mean()
print("\nMean Completion Time by Experience Level:")
print(mean_times)

# t-Test: Beginner vs Expert
beginner = df[df["Experience_Level"] == "Beginner"]["Completion_Time_Days"]
expert = df[df["Experience_Level"] == "Expert"]["Completion_Time_Days"]

t_stat, p_value = stats.ttest_ind(beginner, expert, equal_var=False)

print("\nT-Test (Beginner vs Expert)")
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))

# ANOVA – Gig Category
groups = [
    df[df["Gig_Category"] == cat]["Completion_Time_Days"]
    for cat in df["Gig_Category"].unique()
]

f_stat, p_val = stats.f_oneway(*groups)

print("\nANOVA (Gig Categories)")
print("F-statistic:", round(f_stat, 3))
print("p-value:", round(p_val, 4))

# Histogram
plt.figure()
plt.hist(df["Completion_Time_Days"], bins=10)
plt.title("Distribution of Completion Time")
plt.xlabel("Completion Time (Days)")
plt.ylabel("Frequency")
plt.show()
 