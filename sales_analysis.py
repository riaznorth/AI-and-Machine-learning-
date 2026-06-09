import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load datasets
print("Loading data files...")
sales_2020 = pd.read_csv('AdventureWorks Sales Data 2020.csv')
sales_2021 = pd.read_csv('AdventureWorks Sales Data 2021.csv')
sales_2022 = pd.read_csv('AdventureWorks Sales Data 2022.csv')

# Load supporting lookup tables
products = pd.read_csv('AdventureWorks Product Lookup.csv')
customers = pd.read_csv('AdventureWorks Customer Lookup.csv')
territories = pd.read_csv('AdventureWorks Territory Lookup.csv')

print("Data files loaded successfully!\n")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_sales_data(sales_df, year):
    """Preprocess sales data for analysis"""
    df = sales_df.copy()
    
    # Convert date columns to datetime
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['StockDate'] = pd.to_datetime(df['StockDate'])
    
    # Extract month and quarter for time-based analysis
    df['Month'] = df['OrderDate'].dt.month
    df['Quarter'] = df['OrderDate'].dt.quarter
    df['Year'] = year
    
    return df

# Preprocess all sales data
sales_2020 = preprocess_sales_data(sales_2020, 2020)
sales_2021 = preprocess_sales_data(sales_2021, 2021)
sales_2022 = preprocess_sales_data(sales_2022, 2022)

# Combine all sales data
all_sales = pd.concat([sales_2020, sales_2021, sales_2022], ignore_index=True)

# Merge with product information to get pricing
sales_with_products = all_sales.merge(
    products[['ProductKey', 'ProductName', 'ProductPrice', 'ProductCost']],
    on='ProductKey',
    how='left'
)

# Calculate revenue and profit
sales_with_products['Revenue'] = sales_with_products['OrderQuantity'] * sales_with_products['ProductPrice']
sales_with_products['Cost'] = sales_with_products['OrderQuantity'] * sales_with_products['ProductCost']
sales_with_products['Profit'] = sales_with_products['Revenue'] - sales_with_products['Cost']

print("=" * 80)
print("ADVENTURE WORKS SALES ANALYSIS: 2020, 2021, 2022")
print("=" * 80)

# ============================================================================
# 1. YEARLY SALES SUMMARY
# ============================================================================
print("\n1. YEARLY SALES SUMMARY")
print("-" * 80)

yearly_summary = sales_with_products.groupby('Year').agg({
    'OrderNumber': 'count',
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'Cost': 'sum',
    'Profit': 'sum'
}).rename(columns={'OrderNumber': 'Total_Orders'})

yearly_summary['Profit_Margin_%'] = (yearly_summary['Profit'] / yearly_summary['Revenue'] * 100).round(2)
yearly_summary['Avg_Order_Value'] = (yearly_summary['Revenue'] / yearly_summary['Total_Orders']).round(2)

print(yearly_summary.to_string())
print()

# ============================================================================
# 2. QUARTERLY ANALYSIS
# ============================================================================
print("\n2. QUARTERLY SALES ANALYSIS")
print("-" * 80)

quarterly_analysis = sales_with_products.groupby(['Year', 'Quarter']).agg({
    'OrderNumber': 'count',
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'Profit': 'sum'
}).rename(columns={'OrderNumber': 'Total_Orders'}).round(2)

print(quarterly_analysis.to_string())
print()

# Calculate quarter-over-quarter growth
print("\n3. QUARTER-OVER-QUARTER GROWTH ANALYSIS")
print("-" * 80)

for year in [2020, 2021, 2022]:
    print(f"\n{year} Revenue by Quarter:")
    year_data = quarterly_analysis.loc[year, 'Revenue']
    for quarter, revenue in year_data.items():
        print(f"  Q{quarter}: ${revenue:,.2f}")
    
    # Calculate growth rates
    if len(year_data) > 1:
        growth_rates = year_data.pct_change() * 100
        print(f"\n{year} QoQ Growth Rates:")
        for quarter in range(2, len(year_data) + 1):
            if quarter in growth_rates.index:
                growth = growth_rates[quarter]
                print(f"  Q{quarter-1} to Q{quarter}: {growth:+.2f}%")

# ============================================================================
# 4. MONTHLY ANALYSIS
# ============================================================================
print("\n4. MONTHLY SALES TRENDS")
print("-" * 80)

monthly_analysis = sales_with_products.groupby(['Year', 'Month']).agg({
    'OrderNumber': 'count',
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'Profit': 'sum'
}).rename(columns={'OrderNumber': 'Total_Orders'}).round(2)

for year in [2020, 2021, 2022]:
    print(f"\n{year} - Top 3 Months by Revenue:")
    year_data = monthly_analysis.loc[year].nlargest(3, 'Revenue')[['Revenue', 'Profit']]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month_idx, (revenue, profit) in year_data.iterrows():
        print(f"  {month_names[month_idx-1]}: Revenue=${revenue:,.2f}, Profit=${profit:,.2f}")

# ============================================================================
# 5. PRODUCT PERFORMANCE
# ============================================================================
print("\n5. TOP PRODUCTS BY REVENUE (ALL YEARS)")
print("-" * 80)

product_performance = sales_with_products.groupby('ProductName').agg({
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'Profit': 'sum'
}).sort_values('Revenue', ascending=False).round(2)

print("\nTop 10 Products:")
print(product_performance.head(10).to_string())

print("\n\nBottom 5 Products:")
print(product_performance.tail(5).to_string())

# ============================================================================
# 6. TERRITORY ANALYSIS
# ============================================================================
print("\n6. TERRITORY PERFORMANCE")
print("-" * 80)

# Merge with territory data
sales_with_territory = sales_with_products.merge(
    territories[['TerritoryKey', 'SalesTerritoryCountry', 'SalesTerritoryRegion']],
    on='TerritoryKey',
    how='left'
)

territory_analysis = sales_with_territory.groupby(['Year', 'SalesTerritoryCountry']).agg({
    'OrderNumber': 'count',
    'Revenue': 'sum',
    'Profit': 'sum'
}).rename(columns={'OrderNumber': 'Total_Orders'}).round(2)

print("\nRevenue by Territory and Year:")
print(territory_analysis.to_string())

# ============================================================================
# 7. YEAR-OVER-YEAR COMPARISON
# ============================================================================
print("\n7. YEAR-OVER-YEAR GROWTH COMPARISON")
print("-" * 80)

yoy_revenue = sales_with_products.groupby('Year')['Revenue'].sum()
yoy_profit = sales_with_products.groupby('Year')['Profit'].sum()
yoy_orders = sales_with_products.groupby('Year')['OrderNumber'].count()

print("\nRevenue Trend:")
for year in [2020, 2021, 2022]:
    print(f"  {year}: ${yoy_revenue[year]:,.2f}")

print("\nYear-over-Year Revenue Growth:")
revenue_growth_2021 = ((yoy_revenue[2021] - yoy_revenue[2020]) / yoy_revenue[2020] * 100)
revenue_growth_2022 = ((yoy_revenue[2022] - yoy_revenue[2021]) / yoy_revenue[2021] * 100)
print(f"  2020 to 2021: {revenue_growth_2021:+.2f}%")
print(f"  2021 to 2022: {revenue_growth_2022:+.2f}%")

print("\nProfit Trend:")
for year in [2020, 2021, 2022]:
    print(f"  {year}: ${yoy_profit[year]:,.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

# Figure 1: Revenue by Year
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Annual Revenue
ax1 = axes[0, 0]
years = yearly_summary.index
revenues = yearly_summary['Revenue']
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars1 = ax1.bar(years, revenues, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Annual Revenue Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=10)
ax1.set_xlabel('Year', fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1e6:.2f}M',
            ha='center', va='bottom', fontweight='bold')

# Plot 2: Annual Profit
ax2 = axes[0, 1]
profits = yearly_summary['Profit']
bars2 = ax2.bar(years, profits, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Annual Profit Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Profit ($)', fontsize=10)
ax2.set_xlabel('Year', fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1e6:.2f}M',
            ha='center', va='bottom', fontweight='bold')

# Plot 3: Profit Margin
ax3 = axes[1, 0]
profit_margins = yearly_summary['Profit_Margin_%']
bars3 = ax3.bar(years, profit_margins, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_title('Profit Margin by Year', fontsize=12, fontweight='bold')
ax3.set_ylabel('Profit Margin (%)', fontsize=10)
ax3.set_xlabel('Year', fontsize=10)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold')

# Plot 4: Orders Count
ax4 = axes[1, 1]
orders = yearly_summary['Total_Orders']
bars4 = ax4.bar(years, orders, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_title('Total Orders by Year', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Orders', fontsize=10)
ax4.set_xlabel('Year', fontsize=10)
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('sales_analysis_annual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sales_analysis_annual.png")

# Figure 2: Quarterly Trends
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Revenue by Quarter
ax1 = axes[0]
quarterly_revenue = sales_with_products.groupby(['Year', 'Quarter'])['Revenue'].sum().unstack()
quarterly_revenue.T.plot(kind='bar', ax=ax1, color=colors, width=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Quarterly Revenue Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=10)
ax1.set_xlabel('Quarter', fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax1.legend(title='Year', loc='upper left')
ax1.tick_params(axis='x', rotation=0)

# Plot 2: Profit by Quarter
ax2 = axes[1]
quarterly_profit = sales_with_products.groupby(['Year', 'Quarter'])['Profit'].sum().unstack()
quarterly_profit.T.plot(kind='bar', ax=ax2, color=colors, width=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Quarterly Profit Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Profit ($)', fontsize=10)
ax2.set_xlabel('Quarter', fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax2.legend(title='Year', loc='upper left')
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('sales_analysis_quarterly.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sales_analysis_quarterly.png")

# Figure 3: Top Products
fig, ax = plt.subplots(figsize=(12, 8))
top_products = product_performance.head(10).sort_values('Revenue')
top_products['Revenue'].plot(kind='barh', ax=ax, color='#2E86AB', edgecolor='black', linewidth=1.5)
ax.set_title('Top 10 Products by Revenue (2020-2022)', fontsize=12, fontweight='bold')
ax.set_xlabel('Revenue ($)', fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
for i, v in enumerate(top_products['Revenue']):
    ax.text(v, i, f' ${v/1e6:.2f}M', va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('sales_analysis_top_products.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sales_analysis_top_products.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. sales_analysis_annual.png - Annual revenue, profit, margin, and orders")
print("  2. sales_analysis_quarterly.png - Quarterly trends across years")
print("  3. sales_analysis_top_products.png - Top 10 products by revenue")
