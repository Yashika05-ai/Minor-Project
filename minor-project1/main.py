# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create Simulated Dataset
np.random.seed(42)

n = 120

data = {
    "Freelancer_ID": np.arange(1, n + 1),
    "Gig_Category": np.random.choice(
        ["Design", "Writing", "Coding", "Data"], n
    ),
    "Experience_Level": np.random.choice(
        ["Beginner", "Intermediate", "Expert"], n
    ),
    "Completion_Time_Days": np.random.normal(10, 3, n).clip(2, 25)
}

df = pd.DataFrame(data)

#Descriptive Statistics
desc_stats = df["Completion_Time_Days"].describe()
print("Descriptive Statistics:")
print(desc_stats)

# Box Plot – Completion Time vs Gig Category
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

# Box Plot – Completion Time vs Experience Level
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

# Mean Completion Time Comparison
mean_times = df.groupby("Experience_Level")["Completion_Time_Days"].mean()
print("\nMean Completion Time by Experience Level:")
print(mean_times)

# Two-Sample t-Test (Beginner vs Expert)
beginner = df[df["Experience_Level"] == "Beginner"]["Completion_Time_Days"]
expert = df[df["Experience_Level"] == "Expert"]["Completion_Time_Days"]

t_stat, p_value = stats.ttest_ind(beginner, expert)

print("\nTwo-Sample t-Test (Beginner vs Expert)")
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))

# One-Way ANOVA (Gig Categories)
design = df[df["Gig_Category"] == "Design"]["Completion_Time_Days"]
writing = df[df["Gig_Category"] == "Writing"]["Completion_Time_Days"]
coding = df[df["Gig_Category"] == "Coding"]["Completion_Time_Days"]
data_gig = df[df["Gig_Category"] == "Data"]["Completion_Time_Days"]

f_stat, p_val = stats.f_oneway(design, writing, coding, data_gig)

print("\nOne-Way ANOVA (Gig Category)")
print("F-statistic:", round(f_stat, 3))
print("p-value:", round(p_val, 4))

# Histogram – Distribution of Completion Time
plt.figure()
plt.hist(df["Completion_Time_Days"], bins=10)
plt.title("Distribution of Completion Time")
plt.xlabel("Completion Time (Days)")
plt.ylabel("Frequency")
plt.show()

