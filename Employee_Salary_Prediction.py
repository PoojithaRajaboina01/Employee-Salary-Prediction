#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn joblib')


# In[3]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# In[4]:


# Number of employees
n = 2000

np.random.seed(42)

employee_id = np.arange(1001, 1001 + n)

age = np.random.randint(21, 60, n)

gender = np.random.choice(["Male", "Female"], n)

education = np.random.choice(
    ["Bachelor", "Master", "PhD"],
    n,
    p=[0.6, 0.3, 0.1]
)

experience = np.random.randint(0, 31, n)

job_role = np.random.choice(
    ["Developer", "Data Analyst", "Manager", "Designer", "HR"],
    n
)

skills = np.random.randint(1, 11, n)

certifications = np.random.randint(0, 11, n)

company = np.random.choice(
    ["Startup", "Medium Scale", "MNC"],
    n
)

work_hours = np.random.randint(35, 61, n)


# In[5]:


salary = []

for i in range(n):

    education_bonus = {
        "Bachelor": 0,
        "Master": 80000,
        "PhD": 150000
    }[education[i]]

    company_bonus = {
        "Startup": 20000,
        "Medium Scale": 50000,
        "MNC": 100000
    }[company[i]]

    salary_value = (
        300000
        + experience[i] * 45000
        + skills[i] * 12000
        + certifications[i] * 8000
        + education_bonus
        + company_bonus
        + np.random.randint(-40000, 40000)
    )

    salary.append(salary_value)


# In[6]:


df = pd.DataFrame({
    "EmployeeID": employee_id,
    "Age": age,
    "Gender": gender,
    "EducationLevel": education,
    "YearsOfExperience": experience,
    "JobRole": job_role,
    "SkillsScore": skills,
    "Certifications": certifications,
    "CompanyType": company,
    "WorkHoursPerWeek": work_hours,
    "Salary": salary
})


# In[7]:


df.head()


# In[8]:


df.to_csv("employee_salary_dataset.csv", index=False)

print("Dataset saved successfully!")


# In[9]:


print(df.shape)


# In[10]:


df.info()


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df = df.drop_duplicates()


# In[17]:


print(df["Age"].min())
print(df["Age"].max())


# In[18]:


print(df["YearsOfExperience"].min())
print(df["YearsOfExperience"].max())


# In[19]:


print(df["SkillsScore"].min())
print(df["SkillsScore"].max())


# In[20]:


print(df["Certifications"].min())
print(df["Certifications"].max())


# In[21]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


# In[22]:


df["Gender"] = encoder.fit_transform(df["Gender"])

df["EducationLevel"] = encoder.fit_transform(df["EducationLevel"])

df["JobRole"] = encoder.fit_transform(df["JobRole"])

df["CompanyType"] = encoder.fit_transform(df["CompanyType"])


# In[23]:


df.head()


# In[24]:


df.to_csv("employee_salary_cleaned.csv", index=False)

print("Preprocessing completed successfully!")


# In[25]:


## 6. Exploratory Data Analysis (EDA)


# In[26]:


df.head(10)


# In[27]:


print("Rows and Columns:", df.shape)


# In[28]:


print(df.columns)


# In[29]:


df.describe()


# In[30]:


print("Average Salary:", round(df["Salary"].mean(), 2))


# In[31]:


print("Highest Salary:", df["Salary"].max())


# In[32]:


print("Lowest Salary:", df["Salary"].min())


# In[33]:


df.corr(numeric_only=True)


# In[34]:


education_map = {
    0: "Bachelor",
    1: "Master",
    2: "PhD"
}

temp = df.copy()
temp["EducationLevel"] = temp["EducationLevel"].map(education_map)

temp.groupby("EducationLevel")["Salary"].mean()


# In[35]:


company_map = {
    0: "MNC",
    1: "Medium Scale",
    2: "Startup"
}

temp["CompanyType"] = temp["CompanyType"].map(company_map)

temp.groupby("CompanyType")["Salary"].mean()


# In[36]:


job_map = {
    0: "Data Analyst",
    1: "Designer",
    2: "Developer",
    3: "HR",
    4: "Manager"
}

temp["JobRole"] = temp["JobRole"].map(job_map)

temp.groupby("JobRole")["Salary"].mean()


# In[38]:


## 7. Data Visualization


# In[39]:


plt.style.use("ggplot")


# In[40]:


plt.figure(figsize=(8,5))

plt.scatter(df["YearsOfExperience"], df["Salary"], alpha=0.6)

plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.show()


# In[41]:


job_map = {
    0: "Data Analyst",
    1: "Designer",
    2: "Developer",
    3: "HR",
    4: "Manager"
}

temp = df.copy()
temp["JobRole"] = temp["JobRole"].map(job_map)

average_salary = temp.groupby("JobRole")["Salary"].mean().sort_values()


# In[42]:


plt.figure(figsize=(8,5))

plt.bar(average_salary.index, average_salary.values)

plt.title("Average Salary by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Average Salary")

plt.xticks(rotation=20)

plt.show()


# In[43]:


plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()


# In[44]:


education_map = {
    0: "Bachelor",
    1: "Master",
    2: "PhD"
}

temp = df.copy()

temp["EducationLevel"] = temp["EducationLevel"].map(education_map)


# In[45]:


plt.figure(figsize=(8,5))

sns.boxplot(
    data=temp,
    x="EducationLevel",
    y="Salary"
)

plt.title("Salary Distribution by Education Level")

plt.show()


# In[46]:


plt.savefig("experience_vs_salary.png")


# In[49]:


## 8. Machine Learning Model


# In[50]:


X = df.drop(["EmployeeID", "Salary"], axis=1)

y = df["Salary"]


# In[51]:


print(X.head())
print(y.head())


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[53]:


print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# In[54]:


lr = LinearRegression()

lr.fit(X_train, y_train)


# In[55]:


lr_pred = lr.predict(X_test)


# In[56]:


lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression Results")
print("-------------------------")
print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)


# In[57]:


dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)


# In[58]:


dt_pred = dt.predict(X_test)


# In[59]:


dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree Results")
print("---------------------")
print("MAE :", dt_mae)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)


# In[60]:


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)


# In[61]:


rf_pred = rf.predict(X_test)


# In[62]:


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Results")
print("---------------------")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# In[63]:


results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        lr_mae,
        dt_mae,
        rf_mae
    ],
    "RMSE": [
        lr_rmse,
        dt_rmse,
        rf_rmse
    ],
    "R2 Score": [
        lr_r2,
        dt_r2,
        rf_r2
    ]
})

results


# In[64]:


best_model = results.loc[results["R2 Score"].idxmax()]

print("Best Model")
print(best_model)


# In[65]:


joblib.dump(rf, "salary_prediction_model.pkl")

print("Model saved successfully!")


# In[66]:


loaded_model = joblib.load("salary_prediction_model.pkl")

print(loaded_model)


# In[68]:


## 9. Interactive Salary Prediction System


# In[69]:


import joblib

model = joblib.load("salary_prediction_model.pkl")


# In[70]:


from sklearn.preprocessing import LabelEncoder

gender_encoder = LabelEncoder()
gender_encoder.fit(["Male", "Female"])
print("Gender:", dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_))))

education_encoder = LabelEncoder()
education_encoder.fit(["Bachelor", "Master", "PhD"])
print("Education:", dict(zip(education_encoder.classes_, education_encoder.transform(education_encoder.classes_))))

job_encoder = LabelEncoder()
job_encoder.fit(["Developer", "Data Analyst", "Manager", "Designer", "HR"])
print("Job Role:", dict(zip(job_encoder.classes_, job_encoder.transform(job_encoder.classes_))))

company_encoder = LabelEncoder()
company_encoder.fit(["Startup", "Medium Scale", "MNC"])
print("Company Type:", dict(zip(company_encoder.classes_, company_encoder.transform(company_encoder.classes_))))


# In[72]:


# Load model
model = joblib.load("salary_prediction_model.pkl")

print("===== Employee Salary Prediction =====")

# Numerical inputs
age = int(input("Enter Age: "))
experience = int(input("Enter Years of Experience: "))
skills = int(input("Enter Skills Score (1-10): "))
certifications = int(input("Enter Number of Certifications: "))
work_hours = int(input("Enter Work Hours Per Week: "))

# Encoded categorical inputs
print("\nGender: Male=1, Female=0")
gender = int(input("Enter Gender: "))

print("\nEducation: Bachelor=0, Master=1, PhD=2")
education = int(input("Enter Education Level: "))

print("\nJob Role:")
print("Data Analyst=0")
print("Designer=1")
print("Developer=2")
print("HR=3")
print("Manager=4")
job = int(input("Enter Job Role: "))

print("\nCompany Type:")
print("MNC=0")
print("Medium Scale=1")
print("Startup=2")
company = int(input("Enter Company Type: "))

# Create input list
user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "EducationLevel": [education],
    "YearsOfExperience": [experience],
    "JobRole": [job],
    "SkillsScore": [skills],
    "Certifications": [certifications],
    "CompanyType": [company],
    "WorkHoursPerWeek": [work_hours]
})

predicted_salary = model.predict(user_data)

print("\nPredicted Employee Salary")
print(f"₹ {predicted_salary[0]:,.2f}")


# In[ ]:




