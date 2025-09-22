# Mini Capstone Project: Customer Purchase Analytics for ComponentX
# Run this on Google Colab

# Step 1: Install required packages
!pip install sqlalchemy pandas mlxtend matplotlib seaborn

# Step 2: Import Libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Step 3: Sample Data
customers = pd.DataFrame({
    'customer_id': ['C1', 'C2', 'C3'],
    'name': ['Aarav', 'Meera', 'Karan'],
    'city': ['Mumbai', 'Pune', 'Delhi'],
    'signup_date': ['2025-01-05', '2025-02-10', '2025-03-15']
})

products = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3'],
    'name': ['Arduino Uno', 'ESP32', 'Sensor Kit'],
    'category': ['Boards', 'Boards', 'Sensors'],
    'retail_price': [500, 450, 800]
})

orders = pd.DataFrame({
    'order_id': ['O1', 'O2', 'O3', 'O4'],
    'order_datetime': ['2025-09-10 12:30', '2025-09-11 15:45', '2025-09-12 17:00', '2025-09-12 18:30'],
    'customer_id': ['C1', 'C2', 'C1', 'C3'],
    'product_id': ['P1', 'P3', 'P2', 'P1'],
    'quantity': [2, 1, 3, 1],
    'unit_price': [500, 800, 450, 500],
    'discount': [0, 50, 0, 0]
})

# Step 4: Create SQLite database (acts as DW)
engine = create_engine('sqlite:///componentx_dw.db', echo=False)

# Create dimension tables
customers.to_sql('dim_customer', con=engine, if_exists='replace', index=False)
products.to_sql('dim_product', con=engine, if_exists='replace', index=False)

# Create date dimension table
dates = pd.DataFrame({
    'date_id': pd.date_range(start='2025-09-10', end='2025-09-12')
})
dates['year'] = dates['date_id'].dt.year
dates['month'] = dates['date_id'].dt.month
dates['day'] = dates['date_id'].dt.day
dates.to_sql('dim_date', con=engine, if_exists='replace', index=False)

# Step 5: Create fact_sales table
orders['order_datetime'] = pd.to_datetime(orders['order_datetime'])
orders['total_amount'] = orders['quantity'] * orders['unit_price'] - orders['discount']
orders['date_id'] = orders['order_datetime'].dt.floor('D')
orders.to_sql('fact_sales', con=engine, if_exists='replace', index=False)

print("Data Warehouse tables created successfully!")

# Step 6: Perform RFM Analysis
# Recency = days since last purchase
latest_date = orders['order_datetime'].max() + pd.Timedelta(days=1)
rfm = orders.groupby('customer_id').agg({
    'order_datetime': lambda x: (latest_date - x.max()).days,
    'order_id': 'count',
    'total_amount': 'sum'
}).reset_index()
rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# Simple segmentation
def rfm_segment(row):
    if row['Recency'] <= 2 and row['Frequency'] >= 2:
        return 'High-Value'
    else:
        return 'Low-Value'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)
print("\nRFM Table:")
print(rfm)

# Step 7: Association Rules (Market Basket)
basket = orders.pivot_table(index='order_id', columns='product_id', values='quantity', fill_value=0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])

# Step 8: Simple Visualizations
# Total Sales by Product
sales_by_product = orders.groupby('product_id')['total_amount'].sum().reset_index()
sales_by_product = sales_by_product.merge(products[['product_id','name']], on='product_id')
sns.barplot(x='name', y='total_amount', data=sales_by_product)
plt.title("Total Sales by Product")
plt.ylabel("Total Sales")
plt.xlabel("Product")
plt.show()

# Customer Segments Pie Chart
segment_count = rfm['Segment'].value_counts().reset_index()
segment_count.columns = ['Segment', 'Count']
plt.pie(segment_count['Count'], labels=segment_count['Segment'], autopct='%1.1f%%', startangle=140)
plt.title("Customer Segments")
plt.show()
