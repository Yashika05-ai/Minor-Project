📊 Online Freelance Gig Completion Time Analysis

📌 Introduction
Online freelancing platforms such as Upwork and Freelancer host thousands of projects every day.  
Each project differs in complexity, category, and completion time.  
Understanding how long freelance gigs take to complete helps analyze **productivity trends** and **work efficiency**.

This project analyzes **freelance gig completion time** using **statistical techniques** as part of **Minor 1 – Statistics**.

 🎯 Aim of the Project
The main aim of this project is to:
- Analyze the time taken to complete freelance gigs
- Compare productivity across different categories and experience levels
- Apply statistical concepts learned in class to real-world data
 📂 Dataset Description

Data Source
- Dataset obtained from **Kaggle**
- Dataset Name: *Upwork Job Postings Dataset*
- Link:  
  https://www.kaggle.com/datasets/asaniczka/upwork-job-postings-dataset-2024-50k-records

 Nature of Dataset
- Real-world freelance platform data
- CSV (Comma Separated Values) format
- Contains more than **50,000 records**

 Important Columns Used
- `job_category` → Type of freelance work (Design, Coding, Writing, etc.)
- `experience_level` → Freelancer experience (Beginner, Intermediate, Expert)
- `project_duration` → Estimated project completion time (in days)

These columns are used to analyze completion time and productivity trends.

 🛠 Tools and Technologies Used
- **Python** – Programming language
- **VS Code** – Code editor
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical plots
- **SciPy** – Hypothesis testing

 📄 requirements.txt

pandas
numpy 
matplotlib
seaborn
scipy

▶️ How to Run the Project
 
Step 1: Clone the Repository
git clone https://github.com/Yashika05-ai/Minor-Project.git

Step 2: Install Required Libraries
pip install -r requirements.txt

Step 3: Run the Analysis
python src/analysis.py

