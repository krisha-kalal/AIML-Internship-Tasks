import matplotlib.pyplot as plt

# 1. LINE PLOT: Stock Price over 5 Months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
stock_price = [100, 120, 115, 140, 135]

# plt.figure: Creates a new, blank window for chart.
plt.figure(figsize=(5, 3))
# plt.plot: Draws a Line Plot by connecting data points in sequence.
plt.plot(months, stock_price, marker='o', color='blue')
plt.title("Line Plot: Stock Price Trend")
# plt.xlabel: Adds to the horizontal X-axis at the bottom.
plt.xlabel("Months")
# plt.ylabel: Adds label to the vertical Y-axis on the left side.
plt.ylabel("Price ($)")
# plt.show: Compiles all configurations, pops open the window, and displays the visual graphs.
plt.show()

# 2. HISTOGRAM: Age distribution of a group
ages = [22, 25, 25, 30, 31, 31, 32, 40, 42, 45, 55, 60]
plt.figure(figsize=(5, 3))
# plt.hist: Draws a Histogram to show how frequently numbers fall into continuous ranges. 
plt.hist(ages, bins=4, color='purple', edgecolor='black')
plt.title("Histogram: Age Distribution")
plt.xlabel("Age Groups")
plt.ylabel("Number of People")
plt.show()

# 3. SCATTER PLOT: Study Hours vs Exam Score
study_hours = [1, 2, 3, 5, 6, 7, 9]
exam_scores = [50, 55, 65, 70, 85, 80, 95]
plt.figure(figsize=(5, 3))
# plt.scatter: Draws a Scatter Plot by placing individual, unconnected dots on the grid.
plt.scatter(study_hours, exam_scores, color='green')
plt.title("Scatter Plot: Study Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

# 4. BAR PLOT: Smartphone Sales by Brand
brands = ['Apple', 'Samsung', 'Google', 'Xiaomi']
sales = [500, 600, 150, 300]
plt.figure(figsize=(5, 3))
# plt.bar: Draws a Bar Plot using rectangular bars to compare different independent categories.
plt.bar(brands, sales, color='orange')
plt.title("Bar Plot: Smartphone Sales")
plt.xlabel("Brands")
plt.ylabel("Units Sold")
plt.show()

# 5. PIE CHART: Monthly Budget Allocation
categories = ['Rent', 'Food', 'Utilities', 'Savings']
budget = [40, 30, 15, 15]
plt.figure(figsize=(5, 3))
# plt.pie: Draws a Pie Chart to show parts-to-whole proportions as slices of a circle.
plt.pie(budget, labels=categories, autopct='%1.1f%%', colors=['red', 'orange', 'yellow', 'green'])
plt.title("Pie Chart: Budget Allocation")
plt.show()
