# Employee Salary Prediction System

## Project Overview

The Employee Salary Prediction System is a Machine Learning project developed using Python. The system predicts an employee's estimated annual salary based on factors such as age, education level, years of experience, job role, skills score, certifications, company type, and work hours per week.

This project demonstrates the complete machine learning workflow, including synthetic data generation, data preprocessing, exploratory data analysis (EDA), visualization, model training, evaluation, and salary prediction.

---

## Features

* Synthetic dataset generation with 2000 employee records
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Data visualization using Matplotlib and Seaborn
* Multiple regression model implementation
* Model comparison using evaluation metrics
* Interactive salary prediction system
* Model saving using Joblib

---

## Dataset

The dataset contains the following attributes:

* EmployeeID
* Age
* Gender
* EducationLevel
* YearsOfExperience
* JobRole
* SkillsScore
* Certifications
* CompanyType
* WorkHoursPerWeek
* Salary (Target Variable)

The dataset was synthetically generated using Python libraries such as NumPy and Pandas with realistic salary variations based on experience, education, skills, certifications, and company type.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## Project Workflow

1. Import required libraries.
2. Generate a synthetic employee dataset.
3. Perform data preprocessing.
4. Conduct exploratory data analysis.
5. Visualize important patterns in the data.
6. Split the dataset into training and testing sets.
7. Train Linear Regression, Decision Tree Regressor, and Random Forest Regressor models.
8. Evaluate model performance using MAE, RMSE, and R² Score.
9. Select the best-performing model.
10. Build an interactive salary prediction system.

---

## Data Visualizations

The project includes:

* Experience vs Salary Scatter Plot
* Average Salary by Job Role Bar Chart
* Correlation Heatmap
* Salary Distribution by Education Level Box Plot

---

## Machine Learning Models

The following regression algorithms were implemented:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The Random Forest Regressor achieved the best overall performance and was selected for salary prediction.

---

## Project Structure

```text
Employee-Salary-Prediction/
│
├── Employee_Salary_Prediction.ipynb
├── employee_salary_dataset.csv
├── salary_prediction_model.pkl
├── employee_salary_prediction.py
├── README.md
```

---

## How to Run

1. Clone or download the repository.
2. Install the required libraries.
3. Open the Jupyter Notebook.
4. Run all notebook cells in order.
5. Execute the interactive prediction section.
6. Enter employee details to receive the predicted salary.

---

## Key Insights

* Salary generally increases with years of experience.
* Higher education levels are associated with higher salaries.
* Employees with better skills scores tend to receive higher salaries.
* Certifications positively influence salary.
* Company type also affects salary levels.

---

## Future Enhancements

* Develop a Streamlit web application.
* Deploy the model online.
* Use larger real-world datasets.
* Perform hyperparameter tuning for improved accuracy.
* Add feature importance visualization.

---

## Author

**Rajaboina Poojitha**

Machine Learning Internship Project
