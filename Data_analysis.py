import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load Excel file
file_path = "Data.xlsx"  # Ensure this file is in the working directory
xls = pd.ExcelFile(file_path)

# Read individual sheets
df_co2 = xls.parse('Data')  # Sheet for CO₂ emissions
df_life = xls.parse('Life expectancy')  # Sheet for life expectancy
df_gdp = xls.parse('GDP per capita')  # Sheet for GDP

# ----------------------------
# Step 1: Clean CO₂ emissions data
# ----------------------------
co2_df = df_co2.iloc[1:].copy()
co2_df.columns = ['Year', 'DRC_CO2', 'drop1', 'Year2', 'Burundi_CO2']
co2_df = co2_df[['Year', 'DRC_CO2', 'Burundi_CO2']]
co2_df = co2_df.dropna()
co2_df = co2_df.astype({'Year': int, 'DRC_CO2': float, 'Burundi_CO2': float})

# ----------------------------
# Step 2: Clean life expectancy data
# ----------------------------
life_df = df_life.iloc[1:].copy()
life_df.columns = ['Year', 'Burundi_Life', 'drop1', 'Year2', 'DRC_Life']
life_df = life_df[['Year', 'Burundi_Life', 'DRC_Life']]
life_df = life_df.dropna()
life_df = life_df.astype({'Year': int, 'Burundi_Life': float, 'DRC_Life': float})

# ----------------------------
# Step 3: Clean GDP data
# ----------------------------
gdp_df = df_gdp.iloc[1:].copy()
gdp_df.columns = ['Year', 'Burundi_GDP', 'drop1', 'Year2', 'DRC_GDP']
gdp_df = gdp_df[['Year', 'Burundi_GDP', 'DRC_GDP']]
gdp_df = gdp_df.dropna()
gdp_df = gdp_df.astype({'Year': int, 'Burundi_GDP': float, 'DRC_GDP': float})

# ----------------------------
# Step 4: Merge all datasets on Year
# ----------------------------
merged_df = pd.merge(pd.merge(gdp_df, life_df, on='Year'), co2_df, on='Year')

# The final dataset (merged_df) contains:
# - Year
# - GDP per capita (Burundi & DRC)
# - Life expectancy (Burundi & DRC)
# - CO₂ emissions per capita (Burundi & DRC)

# Figure 1: GDP vs Life Expectancy
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# Extract GDP and Life Expectancy data
x_drc = merged_df['DRC_GDP']
y_drc = merged_df['DRC_Life']
x_burundi = merged_df['Burundi_GDP']
y_burundi = merged_df['Burundi_Life']

# Fit linear regressions (ordinary least squares)
slope_drc, intercept_drc = np.polyfit(x_drc, y_drc, 1)
slope_burundi, intercept_burundi = np.polyfit(x_burundi, y_burundi, 1)

# Generate smooth x values for regression lines
x_vals = np.linspace(min(x_drc.min(), x_burundi.min()), max(x_drc.max(), x_burundi.max()), 100)

# Plotting
plt.figure(figsize=(7, 5))

# Scatter points
sns.scatterplot(x=x_drc, y=y_drc, color='blue', label='DR Congo')
sns.scatterplot(x=x_burundi, y=y_burundi, color='green', label='Burundi')

# Regression lines
plt.plot(x_vals, slope_drc * x_vals + intercept_drc, linestyle='--', color='blue', label=f'DR Congo trend (slope = {slope_drc:.2f})')
plt.plot(x_vals, slope_burundi * x_vals + intercept_burundi, linestyle='--', color='green', label=f'Burundi trend (slope = {slope_burundi:.2f})')

# Axis labels and title
plt.xlabel('GDP per capita (current US$)')
plt.ylabel('Life Expectancy at Birth (years)')
plt.title('GDP per Capita vs Life Expectancy')

# Add legend to explain symbols and trends
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig("figure1_gdp_life.png", dpi=300)


# Figure 2: GDP vs CO₂ Emissions (log-log scatter with regression)
plt.figure(figsize=(7, 5))

# Extract GDP and CO2 data
x_drc = merged_df['DRC_GDP']
y_drc = merged_df['DRC_CO2']
x_burundi = merged_df['Burundi_GDP']
y_burundi = merged_df['Burundi_CO2']

# Remove zeros to avoid log issues
mask_drc = (x_drc > 0) & (y_drc > 0)
mask_burundi = (x_burundi > 0) & (y_burundi > 0)

x_drc_log = np.log10(x_drc[mask_drc])
y_drc_log = np.log10(y_drc[mask_drc])
x_burundi_log = np.log10(x_burundi[mask_burundi])
y_burundi_log = np.log10(y_burundi[mask_burundi])

# Linear fit in log-log space
slope_drc, intercept_drc = np.polyfit(x_drc_log, y_drc_log, 1)
slope_burundi, intercept_burundi = np.polyfit(x_burundi_log, y_burundi_log, 1)

# Scatter plot
sns.scatterplot(x=x_drc, y=y_drc, label='DR Congo', color='blue')
sns.scatterplot(x=x_burundi, y=y_burundi, label='Burundi', color='green')

# Fitted lines
x_vals = np.logspace(np.log10(min(x_drc.min(), x_burundi.min())),
                     np.log10(max(x_drc.max(), x_burundi.max())), 100)

y_fit_drc = 10**(intercept_drc + slope_drc * np.log10(x_vals))
y_fit_burundi = 10**(intercept_burundi + slope_burundi * np.log10(x_vals))

plt.plot(x_vals, y_fit_drc, linestyle='--', color='blue', label=f'DR Congo trend (slope = {slope_drc:.2f})')
plt.plot(x_vals, y_fit_burundi, linestyle='--', color='green', label=f'Burundi trend (slope = {slope_burundi:.2f})')

# Log–log scale
plt.xscale('log')
plt.yscale('log')

# Axes labels and title
plt.xlabel('GDP per capita (log scale)')
plt.ylabel(r'$\mathrm{CO_2}$ emissions per capita (log scale)')
plt.title('Economic Growth and $\mathrm{CO_2}$ Emissions')
plt.legend()

# Save figure
plt.tight_layout()
plt.savefig("figure2_gdp_co2.png", dpi=300)


# Figure 3: Time Series Overview
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# GDP
axs[0].plot(merged_df['Year'], merged_df['DRC_GDP'], label='DR Congo', color='blue')
axs[0].plot(merged_df['Year'], merged_df['Burundi_GDP'], label='Burundi', color='green')
axs[0].set_ylabel('GDP per capita')
axs[0].set_title('GDP per Capita over Time')
axs[0].legend()

# Life Expectancy
axs[1].plot(merged_df['Year'], merged_df['DRC_Life'], label='DR Congo', color='blue')
axs[1].plot(merged_df['Year'], merged_df['Burundi_Life'], label='Burundi', color='green')
axs[1].set_ylabel('Life Expectancy (years)')
axs[1].set_title('Life Expectancy over Time')

# CO2 Emissions
axs[2].plot(merged_df['Year'], merged_df['DRC_CO2'], label='DR Congo', color='blue')
axs[2].plot(merged_df['Year'], merged_df['Burundi_CO2'], label='Burundi', color='green')
axs[2].set_ylabel('$\mathrm{CO_2}$ emissions per capita')
axs[2].set_title('$\mathrm{CO_2}$ Emissions over Time')
axs[2].set_xlabel('Year')

plt.tight_layout()
plt.savefig("figure3_time_series.png", dpi=300)