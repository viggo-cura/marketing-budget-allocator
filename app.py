import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Real-Time Marketing Budget Allocator", page_icon="💰", layout="wide")

# Application title
st.title("Real-Time Marketing Budget Allocator")

# Sidebar for inputs
st.sidebar.header("Budget Settings")

# Weekly budget input
total_weekly_budget = st.sidebar.number_input("Total Weekly Budget ($)", min_value=100, value=10000, step=100)

# Target ROAS inputs
st.sidebar.header("Target ROAS")

# Define the marketing channels including organic search
channels = ["Search", "YouTube", "Instagram", "TikTok", "Meta", "Display", "LinkedIn", "Organic Search"]

# Pre-set default values for a more realistic scenario
default_target_roas = {
    "Search": 1.8,  # Search typically has higher ROAS
    "YouTube": 1.4,
    "Instagram": 1.6,
    "TikTok": 1.5,
    "Meta": 1.6,
    "Display": 1.3,  # Display often has lower ROAS
    "LinkedIn": 1.2,  # B2B channels can have lower ROAS
    "Organic Search": 0.0  # No ROAS target for organic
}

# Also define value for Organic Search (not included in main channels list)
organic_search_revenue_multiplier = 1.5  # Relative to paid search revenue

# Default budget allocation (non-uniform)
default_budget_allocation = {
    "Search": 0.25,      # 25% to Search
    "YouTube": 0.15,     # 15% to YouTube
    "Instagram": 0.15,   # 15% to Instagram
    "TikTok": 0.12,      # 12% to TikTok
    "Meta": 0.18,        # 18% to Meta
    "Display": 0.08,     # 8% to Display
    "LinkedIn": 0.07,    # 7% to LinkedIn
    "Organic Search": 0.0 # 0% to Organic Search (no budget)
}

# Create inputs for each channel's target ROAS and initial budget allocation
target_roas = {}
initial_budget = {}
budget_sum = 0

for channel in channels:
    # Special handling for Organic Search
    if channel == "Organic Search":
        target_roas[channel] = 0.0  # No ROAS target for organic
        initial_budget[channel] = 0.0  # No budget for organic
        st.sidebar.markdown(f"**{channel}**: No budget needed (tracked revenue only)")
        continue
        
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        target_roas[channel] = st.number_input(
            f"{channel} Target ROAS", 
            min_value=0.1, 
            value=default_target_roas[channel],  # Use default value
            step=0.1
        )
    
    with col2:
        # Calculate default initial budget for this channel
        default_value = default_budget_allocation[channel] * total_weekly_budget
        
        # Ensure budget allocations sum to total budget
        remaining = total_weekly_budget - budget_sum
        max_value = min(remaining + initial_budget.get(channel, 0), total_weekly_budget)
        
        initial_budget[channel] = st.number_input(
            f"{channel} Budget ($)", 
            min_value=0.0,
            max_value=float(total_weekly_budget),
            value=float(default_value),  # Use default allocation
            step=100.0
        )
        budget_sum = sum(initial_budget.values())

# Show warning if budget doesn't match total
if abs(budget_sum - total_weekly_budget) > 1:
    st.sidebar.warning(f"Current allocation: ${budget_sum} (Target: ${total_weekly_budget})")
else:
    st.sidebar.success(f"Budget fully allocated: ${budget_sum}")

# Generate mock historical data
def generate_historical_data(channels, days=7):
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, 0, -1)]
    data = []
    
    # Define channel performance characteristics (some underperform)
    channel_performance = {
        "Search": {"min": 0.9, "max": 1.2},   # Search performs well
        "YouTube": {"min": 0.7, "max": 1.1},  # YouTube has mixed results
        "Instagram": {"min": 0.8, "max": 1.1}, # Instagram performs decently
        "TikTok": {"min": 0.7, "max": 1.3},   # TikTok is volatile
        "Meta": {"min": 0.85, "max": 1.15},   # Meta is stable
        "Display": {"min": 0.6, "max": 0.9},  # Display underperforms
        "LinkedIn": {"min": 0.5, "max": 0.8},  # LinkedIn underperforms
        "Organic Search": {"min": 0.0, "max": 0.0}  # Organic Search has no performance
    }
    
    # Define organic search multiplier relative to paid search
    organic_search_multiplier = 1.5  # Organic search revenue is 1.5x paid search
    
    # Track search revenue per day for organic calculation
    search_revenue_by_date = {}
    
    for date in dates:
        for channel in channels:
            # Skip Organic Search for now, we'll add it after processing all paid channels
            if channel == "Organic Search":
                continue
                
            daily_budget = initial_budget[channel] / 7  # Divide weekly budget by 7
            
            # Use channel-specific performance multipliers
            perf_range = channel_performance[channel]
            perf_multiplier = np.random.uniform(perf_range["min"], perf_range["max"])
            
            # First calculate the expected ROAS
            expected_roas = target_roas[channel] * perf_multiplier
            
            # If expected ROAS is above target, use 100% of the budget
            # Otherwise use a variable portion of the budget (70-100%)
            if expected_roas >= target_roas[channel]:
                # For channels with good ROAS, spend 100% of the budget
                spend = daily_budget
            else:
                # For underperforming channels, spend less (70-95% of budget)
                # The worse the performance, the less budget utilized
                performance_ratio = expected_roas / target_roas[channel]
                min_utilization = max(0.5, performance_ratio)  # At least 50% utilization
                max_utilization = min(0.95, 0.7 + performance_ratio * 0.25)  # Cap at 95%
                spend = daily_budget * np.random.uniform(min_utilization, max_utilization)
            
            # Calculate revenue based on spend and expected ROAS
            revenue = spend * expected_roas
            actual_roas = revenue / spend if spend > 0 else 0
            
            # Track Search revenue for organic calculation
            if channel == "Search":
                search_revenue_by_date[date] = revenue
            
            data.append({
                "Date": date,
                "Channel": channel,
                "Budget": daily_budget,
                "Spend": spend,
                "Revenue": revenue,
                "ROAS": actual_roas,
                "Target ROAS": target_roas[channel],
                "Budget Utilization": (spend / daily_budget * 100) if daily_budget > 0 else 0
            })
    
    # Add Organic Search data based on paid search performance
    for date in dates:
        # Base organic search revenue on paid search revenue for that day
        search_revenue = search_revenue_by_date.get(date, 0)
        organic_revenue = search_revenue * organic_search_multiplier * np.random.uniform(0.9, 1.1)
        
        data.append({
            "Date": date,
            "Channel": "Organic Search",
            "Budget": 0,  # Zero budget
            "Spend": 0,   # Zero spend
            "Revenue": organic_revenue,
            "ROAS": np.nan,  # ROAS is undefined (divide by zero)
            "Target ROAS": np.nan,  # No target ROAS
            "Budget Utilization": np.nan  # No utilization metric
        })
    
    return pd.DataFrame(data)

# Generate mock data for previous week
historical_data = generate_historical_data(channels)

# Display the historical data
st.header("Previous Week Performance")

tab1, tab2, tab3 = st.tabs(["Performance Summary", "Daily Breakdown", "Visualization"])

with tab1:
    # Aggregate data by channel
    summary = historical_data.groupby("Channel").agg({
        "Budget": "sum",
        "Spend": "sum",
        "Revenue": "sum"
    }).reset_index()
    
    summary["ROAS"] = summary["Revenue"] / summary["Spend"]
    summary["Budget Utilization"] = summary["Spend"] / summary["Budget"] * 100
    summary["Target ROAS"] = summary["Channel"].map(lambda c: target_roas.get(c, np.nan))
    summary["ROAS Performance"] = summary.apply(
        lambda row: (row["ROAS"] / row["Target ROAS"] * 100) if pd.notnull(row["Target ROAS"]) and row["Target ROAS"] > 0 else np.nan, 
        axis=1
    )
    
    # Format the summary table
    formatted_summary = summary.copy()
    formatted_summary["Budget"] = formatted_summary["Budget"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) and x > 0 else "$0.00")
    formatted_summary["Spend"] = formatted_summary["Spend"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) and x > 0 else "$0.00")
    formatted_summary["Revenue"] = formatted_summary["Revenue"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
    formatted_summary["ROAS"] = formatted_summary["ROAS"].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
    formatted_summary["Budget Utilization"] = formatted_summary["Budget Utilization"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    formatted_summary["Target ROAS"] = formatted_summary["Target ROAS"].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
    formatted_summary["ROAS Performance"] = formatted_summary["ROAS Performance"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    
    st.dataframe(formatted_summary, use_container_width=True)
    
    # Total performance (exclude organic search from ROAS calculation since it has no spend)
    paid_summary = summary[summary["Channel"] != "Organic Search"]
    total_spend = paid_summary["Spend"].sum()
    total_revenue = summary["Revenue"].sum()  # Include organic in total revenue
    paid_revenue = paid_summary["Revenue"].sum()  # Revenue from paid channels only
    total_roas = paid_revenue / total_spend if total_spend > 0 else 0  # ROAS for paid channels
    
    # Calculate what portion of revenue is from organic
    organic_revenue = summary[summary["Channel"] == "Organic Search"]["Revenue"].sum()
    organic_percentage = (organic_revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${total_spend:.2f}")
    col2.metric("Total Revenue", f"${total_revenue:.2f}")
    col3.metric("Overall ROAS", f"{total_roas:.2f}x")
    col4.metric("Organic Revenue", f"${organic_revenue:.2f}", f"{organic_percentage:.1f}% of total")

with tab2:
    # Daily view
    daily_view = st.checkbox("View by day")
    
    if daily_view:
        st.dataframe(
            historical_data.sort_values(["Date", "Channel"]),
            use_container_width=True
        )
    else:
        day_channel_view = historical_data.pivot_table(
            index="Date", 
            columns="Channel", 
            values="ROAS",
            aggfunc="mean"
        )
        
        st.dataframe(day_channel_view.style.highlight_max(axis=1), use_container_width=True)

with tab3:
    # Visualizations - exclude Organic Search since it doesn't have valid ROAS
    paid_summary_plot = summary[summary["Channel"] != "Organic Search"].copy()
    paid_summary_plot["ROAS Gap"] = paid_summary_plot["ROAS"] - paid_summary_plot["Target ROAS"]
    
    # Color based on whether ROAS is above or below target
    colors = ["green" if x >= 0 else "red" for x in paid_summary_plot["ROAS Gap"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    sns.barplot(x="Channel", y="ROAS", data=paid_summary_plot, ax=ax, color="lightblue")
    
    # Add target ROAS lines
    for i, (_, row) in enumerate(paid_summary_plot.iterrows()):
        ax.plot([i-0.4, i+0.4], [row["Target ROAS"], row["Target ROAS"]], 
                color="black", linestyle="--", alpha=0.7)
    
    ax.set_title("ROAS by Channel vs Target")
    ax.set_ylabel("ROAS")
    ax.grid(axis="y", alpha=0.3)
    
    st.pyplot(fig)
    
    # Add a revenue breakdown chart that includes Organic Search
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Sort by revenue and create a custom color map
    revenue_data = summary.sort_values("Revenue", ascending=False)
    colors = ['orange' if channel == 'Organic Search' else 'lightblue' for channel in revenue_data['Channel']]
    
    # Plot revenue by channel with custom colors
    sns.barplot(x="Channel", y="Revenue", data=revenue_data, ax=ax2, palette=colors)
    
    ax2.set_title("Revenue by Channel (Including Organic Search)")
    ax2.set_ylabel("Revenue ($)")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig2)

# Budget optimization for next week
st.header("Budget Optimization for Next Week")

# Algorithm for budget reallocation
def optimize_budget(df, total_budget, target_roas_dict):
    # Filter out organic search which has no budget
    df_paid = df[df["Channel"] != "Organic Search"].copy()
    
    # Group by channel and calculate performance metrics
    channel_perf = df_paid.groupby("Channel").agg({
        "Spend": "sum",
        "Revenue": "sum",
        "Budget": "sum"
    })
    
    channel_perf["ROAS"] = channel_perf["Revenue"] / channel_perf["Spend"]
    channel_perf["Target ROAS"] = pd.Series({k: v for k, v in target_roas_dict.items() if k != "Organic Search"})
    channel_perf["ROAS_Ratio"] = channel_perf["ROAS"] / channel_perf["Target ROAS"]
    channel_perf["Budget_Utilization"] = channel_perf["Spend"] / channel_perf["Budget"] * 100
    
    # Calculate ROAS efficiency score with improved formula
    # Channels performing above target get more budget, below target get less
    # Also account for budget utilization - channels that utilize their budget well should get more
    def calculate_score(row):
        roas_component = np.log(row["ROAS_Ratio"] + 0.5) * 2 if row["ROAS_Ratio"] > 0 else 0.1
        
        # If ROAS is above target, give a bonus
        if row["ROAS"] > row["Target ROAS"]:
            roas_bonus = 1.5  # 50% bonus for exceeding target
        else:
            roas_bonus = 1.0
            
        # Utilization penalty/bonus
        if row["Budget_Utilization"] > 95:  # High utilization is good
            util_factor = 1.2  # 20% bonus for high utilization
        elif row["Budget_Utilization"] < 70:  # Low utilization suggests problems
            util_factor = 0.8  # 20% penalty for low utilization
        else:
            util_factor = 1.0
            
        return max(0.2, roas_component * roas_bonus * util_factor)
    
    channel_perf["Score"] = channel_perf.apply(calculate_score, axis=1)
    
    # Ensure scores are positive and significant
    channel_perf["Score"] = channel_perf["Score"].apply(lambda x: max(0.2, x))
    
    # Normalize scores to create allocation percentages
    total_score = channel_perf["Score"].sum()
    channel_perf["Allocation"] = channel_perf["Score"] / total_score
    
    # Calculate new budgets
    channel_perf["New Budget"] = channel_perf["Allocation"] * total_budget
    
    # Ensure minimum budget for each channel (at least 50% of original)
    for channel in channel_perf.index:
        min_budget = channel_perf.loc[channel, "Budget"] * 0.5 / 7 * 7  # Convert to weekly
        if channel_perf.loc[channel, "New Budget"] < min_budget:
            channel_perf.loc[channel, "New Budget"] = min_budget
    
    # Adjust to ensure sum equals total budget
    budget_sum = channel_perf["New Budget"].sum()
    if budget_sum != total_budget:
        # Proportionally adjust budgets
        channel_perf["New Budget"] = channel_perf["New Budget"] * (total_budget / budget_sum)
    
    # Create optimization summary with current vs new budget
    current_budgets = df_paid.groupby("Channel")["Budget"].sum() / 7 * 7  # Weekly budget
    optimization_summary = pd.DataFrame({
        "Current Budget": current_budgets,
        "New Budget": channel_perf["New Budget"],
        "Change": channel_perf["New Budget"] - current_budgets,
        "Change %": (channel_perf["New Budget"] / current_budgets - 1) * 100,
        "Current ROAS": channel_perf["ROAS"],
        "Target ROAS": channel_perf["Target ROAS"]
    })
    
    return optimization_summary

# Get optimized budget
optimized_budget = optimize_budget(historical_data, total_weekly_budget, target_roas)

# Display optimization results
col1, col2 = st.columns([3, 1])

with col1:
    # Format the display of the optimized budget
    formatted_budget = optimized_budget.copy()
    formatted_budget["Current Budget"] = formatted_budget["Current Budget"].apply(lambda x: f"${x:.2f}")
    formatted_budget["New Budget"] = formatted_budget["New Budget"].apply(lambda x: f"${x:.2f}")
    formatted_budget["Change"] = formatted_budget["Change"].apply(lambda x: f"${x:.2f}")
    formatted_budget["Change %"] = formatted_budget["Change %"].apply(lambda x: f"{x:.1f}%")
    formatted_budget["Current ROAS"] = formatted_budget["Current ROAS"].apply(lambda x: f"{x:.2f}x")
    formatted_budget["Target ROAS"] = formatted_budget["Target ROAS"].apply(lambda x: f"{x:.2f}x")
    
    st.dataframe(formatted_budget, use_container_width=True)

with col2:
    # Calculate estimated revenue with new budget allocation
    current_revenue = historical_data["Revenue"].sum()
    
    # Simple estimation of new revenue - conservative multiplier of ROAS improvement
    avg_current_roas = historical_data["Revenue"].sum() / historical_data["Spend"].sum()
    
    # Apply a weighted average improvement based on new budget distribution
    perf_improvement = 0.05  # Assume 5% improvement from better allocation
    new_roas = avg_current_roas * (1 + perf_improvement)
    estimated_revenue = total_weekly_budget * new_roas
    
    revenue_increase = estimated_revenue - current_revenue
    revenue_increase_pct = (revenue_increase / current_revenue) * 100 if current_revenue > 0 else 0
    
    st.subheader("Projected Impact")
    st.metric("Current Weekly Revenue", f"${current_revenue:.2f}")
    st.metric("Projected Weekly Revenue", f"${estimated_revenue:.2f}", 
              delta=f"${revenue_increase:.2f} ({revenue_increase_pct:.1f}%)")

# Visualization of budget reallocation
fig, ax = plt.subplots(figsize=(10, 6))

# Extract data for plotting
plot_data = optimized_budget.reset_index()
plot_data["Original"] = plot_data["Current Budget"].astype(float)
plot_data["Optimized"] = plot_data["New Budget"].astype(float)

# Convert to long format for seaborn
plot_long = pd.melt(
    plot_data, 
    id_vars=["Channel"], 
    value_vars=["Original", "Optimized"],
    var_name="Budget Type", 
    value_name="Amount"
)

# Create the grouped bar chart
sns.barplot(x="Channel", y="Amount", hue="Budget Type", data=plot_long, ax=ax)

ax.set_title("Budget Reallocation Comparison")
ax.set_ylabel("Budget Amount ($)")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Add a button to apply the optimized budget
if st.button("Apply Optimized Budget"):
    st.success("Budget optimization applied for next week! The changes will take effect starting tomorrow.")
    # In a real application, this would update the database or settings 

# Timeline visualization (past 7 days + next 2 weeks forecast)
st.header("Timeline Forecast (Past week + Next 2 weeks)")

# Create tabs for different views
forecast_tab1, forecast_tab2, forecast_tab3 = st.tabs(["Revenue Forecast", "ROAS Forecast", "Budget Utilization"])

# Function to generate forecast data
def generate_forecast_data(historical_df, optimization_df, forecast_days=14):
    # Get the last date from historical data
    last_date = datetime.strptime(historical_df["Date"].max(), "%Y-%m-%d")
    
    # Create future dates
    future_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
    
    # Create data frames for current and optimized forecasts
    current_forecast = []
    optimized_forecast = []
    
    # Get channel-specific metrics
    channel_metrics = historical_df.groupby("Channel").agg({
        "Spend": "sum",
        "Revenue": "sum"
    })
    
    # Calculate ROAS only for channels with spend
    channel_metrics["ROAS"] = np.nan
    for channel in channel_metrics.index:
        if channel_metrics.loc[channel, "Spend"] > 0:
            channel_metrics.loc[channel, "ROAS"] = channel_metrics.loc[channel, "Revenue"] / channel_metrics.loc[channel, "Spend"]
    
    # Get organic search metrics if available
    has_organic = "Organic Search" in channel_metrics.index
    if has_organic:
        organic_revenue = channel_metrics.loc["Organic Search", "Revenue"]
        search_revenue = channel_metrics.loc["Search", "Revenue"] if "Search" in channel_metrics.index else 0
        organic_ratio = organic_revenue / search_revenue if search_revenue > 0 else 1.5
    else:
        organic_ratio = 1.5  # Default ratio if no historical data
    
    # Channel budgets
    current_budgets = optimization_df["Current Budget"].to_dict()
    optimized_budgets = optimization_df["New Budget"].to_dict()
    
    # Define channel performance improvements with optimization
    # Underperforming channels tend to get more improvement from optimization
    channel_optimization_improvement = {
        "Search": 1.10,      # 10% improvement
        "YouTube": 1.15,     # 15% improvement
        "Instagram": 1.12,   # 12% improvement
        "TikTok": 1.18,      # 18% improvement
        "Meta": 1.13,        # 13% improvement
        "Display": 1.20,     # 20% improvement
        "LinkedIn": 1.22,    # 22% improvement
    }
    
    # Generate forecasts for each channel and day
    for date in future_dates:
        current_search_revenue = 0
        optimized_search_revenue = 0
        
        for channel in current_budgets.keys():  # Only loop through paid channels
            # Current budget forecast
            daily_current_budget = current_budgets[channel] / 7  # Daily budget
            
            # Use historical ROAS with some variability
            if channel in channel_metrics.index and not pd.isna(channel_metrics.loc[channel, "ROAS"]):
                historical_roas = channel_metrics.loc[channel, "ROAS"]
                roas_variability = np.random.uniform(0.9, 1.1)
                current_roas = historical_roas * roas_variability
            else:
                current_roas = target_roas[channel] * np.random.uniform(0.8, 1.2)
            
            # Budget utilization based on ROAS performance
            if current_roas >= target_roas[channel]:
                # If ROAS is good, spend 100% of budget
                spend_current = daily_current_budget
            else:
                # Otherwise spend less (proportional to ROAS performance)
                performance_ratio = current_roas / target_roas[channel]
                min_utilization = max(0.6, performance_ratio)
                max_utilization = min(0.95, 0.7 + performance_ratio * 0.25)
                spend_current = daily_current_budget * np.random.uniform(min_utilization, max_utilization)
            
            revenue_current = spend_current * current_roas
            
            # Track Search revenue for organic calculation
            if channel == "Search":
                current_search_revenue = revenue_current
            
            current_forecast.append({
                "Date": date,
                "Channel": channel,
                "Budget": daily_current_budget,
                "Spend": spend_current,
                "Revenue": revenue_current,
                "ROAS": current_roas,
                "Forecast Type": "Current Budget",
                "Budget Utilization": (spend_current / daily_current_budget * 100) if daily_current_budget > 0 else 0
            })
            
            # Optimized budget forecast
            daily_optimized_budget = optimized_budgets[channel] / 7  # Daily budget
            
            # Apply channel-specific improvement factor
            optimized_roas = current_roas * channel_optimization_improvement.get(channel, 1.05)
            
            # Budget utilization based on optimized ROAS performance
            if optimized_roas >= target_roas[channel]:
                # If ROAS is good, spend 100% of budget
                spend_optimized = daily_optimized_budget
            else:
                # Otherwise spend less (proportional to ROAS performance)
                performance_ratio = optimized_roas / target_roas[channel]
                min_utilization = max(0.7, performance_ratio)  # Higher floor for optimized
                max_utilization = min(0.98, 0.8 + performance_ratio * 0.18)  # Higher ceiling
                spend_optimized = daily_optimized_budget * np.random.uniform(min_utilization, max_utilization)
            
            revenue_optimized = spend_optimized * optimized_roas
            
            # Track Search revenue for organic calculation
            if channel == "Search":
                optimized_search_revenue = revenue_optimized
            
            optimized_forecast.append({
                "Date": date,
                "Channel": channel,
                "Budget": daily_optimized_budget,
                "Spend": spend_optimized,
                "Revenue": revenue_optimized,
                "ROAS": optimized_roas,
                "Forecast Type": "Optimized Budget",
                "Budget Utilization": (spend_optimized / daily_optimized_budget * 100) if daily_optimized_budget > 0 else 0
            })
        
        # Add Organic Search forecasts which depend on Search channel performance
        # For current budget scenario
        organic_revenue_current = current_search_revenue * organic_ratio * np.random.uniform(0.9, 1.1)
        current_forecast.append({
            "Date": date,
            "Channel": "Organic Search",
            "Budget": 0,
            "Spend": 0,
            "Revenue": organic_revenue_current,
            "ROAS": np.nan,
            "Forecast Type": "Current Budget",
            "Budget Utilization": np.nan
        })
        
        # For optimized budget scenario
        # Organic search also slightly benefits from improved paid search
        organic_revenue_optimized = optimized_search_revenue * organic_ratio * np.random.uniform(0.95, 1.15)
        optimized_forecast.append({
            "Date": date,
            "Channel": "Organic Search",
            "Budget": 0,
            "Spend": 0,
            "Revenue": organic_revenue_optimized,
            "ROAS": np.nan,
            "Forecast Type": "Optimized Budget",
            "Budget Utilization": np.nan
        })
    
    # Combine historical and forecast data
    historical_copy = historical_df.copy()
    historical_copy["Forecast Type"] = "Actual"
    
    # Make sure historical data has Budget Utilization column
    if "Budget Utilization" not in historical_copy.columns:
        historical_copy["Budget Utilization"] = historical_copy.apply(
            lambda row: (row["Spend"] / row["Budget"] * 100) if row["Budget"] > 0 else np.nan, 
            axis=1
        )
    
    # Select only relevant columns from historical data
    historical_copy = historical_copy[["Date", "Channel", "Budget", "Spend", "Revenue", "ROAS", "Budget Utilization", "Forecast Type"]]
    
    # Combine all data
    combined_data = pd.concat([
        historical_copy, 
        pd.DataFrame(current_forecast), 
        pd.DataFrame(optimized_forecast)
    ])
    
    # Set proper data types
    combined_data["Date"] = pd.to_datetime(combined_data["Date"])
    
    return combined_data

# Generate forecast data
forecast_data = generate_forecast_data(historical_data, optimized_budget, forecast_days=14)

# Revenue Forecast Tab
with forecast_tab1:
    # Group by date and forecast type, sum revenue
    revenue_by_date = forecast_data.groupby(["Date", "Forecast Type"])["Revenue"].sum().reset_index()
    
    # Create revenue timeline plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data with different styles for actual vs forecast
    for forecast_type, color, linestyle in [
        ("Actual", "black", "-"), 
        ("Current Budget", "blue", "--"), 
        ("Optimized Budget", "green", "--")
    ]:
        data = revenue_by_date[revenue_by_date["Forecast Type"] == forecast_type]
        ax.plot(data["Date"], data["Revenue"], 
                label=forecast_type, 
                color=color, 
                linestyle=linestyle,
                marker="o" if forecast_type == "Actual" else None,
                linewidth=2)
    
    # Add vertical line to separate actual and forecast
    last_actual_date = forecast_data[forecast_data["Forecast Type"] == "Actual"]["Date"].max()
    ax.axvline(x=last_actual_date, color="gray", linestyle="-.", alpha=0.5)
    ax.text(last_actual_date, ax.get_ylim()[1]*0.9, "Forecast Start", 
            ha="center", va="center", backgroundcolor="white", fontsize=10)
    
    # Formatting
    ax.set_title("Revenue Timeline: Past Week + Next 2 Weeks Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis to show dates nicely
    fig.autofmt_xdate()
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Summary metrics
    st.subheader("Revenue Summary")
    
    actual_revenue = revenue_by_date[revenue_by_date["Forecast Type"] == "Actual"]["Revenue"].sum()
    current_forecast_revenue = revenue_by_date[revenue_by_date["Forecast Type"] == "Current Budget"]["Revenue"].sum()
    optimized_forecast_revenue = revenue_by_date[revenue_by_date["Forecast Type"] == "Optimized Budget"]["Revenue"].sum()
    
    improvement = optimized_forecast_revenue - current_forecast_revenue
    improvement_pct = (improvement / current_forecast_revenue) * 100 if current_forecast_revenue > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Past Week Revenue", f"${actual_revenue:.2f}")
    col2.metric("Forecast with Current Budget", f"${current_forecast_revenue:.2f}")
    col3.metric("Forecast with Optimized Budget", f"${optimized_forecast_revenue:.2f}", 
               delta=f"+${improvement:.2f} ({improvement_pct:.1f}%)")
    
    # Channel Revenue Breakdown
    st.subheader("Channel Revenue Breakdown")
    
    # Calculate revenue by channel for current vs optimized
    channel_revenue = forecast_data[forecast_data["Forecast Type"].isin(["Current Budget", "Optimized Budget"])].groupby(
        ["Channel", "Forecast Type"]
    )["Revenue"].sum().reset_index()
    
    # Pivot the data for easy comparison
    channel_revenue_pivot = channel_revenue.pivot(
        index="Channel", 
        columns="Forecast Type", 
        values="Revenue"
    )
    
    # Calculate change and percent change
    channel_revenue_pivot["Change"] = channel_revenue_pivot["Optimized Budget"] - channel_revenue_pivot["Current Budget"]
    channel_revenue_pivot["Change %"] = (channel_revenue_pivot["Change"] / channel_revenue_pivot["Current Budget"] * 100)
    
    # Format for display
    formatted_channel_revenue = channel_revenue_pivot.copy()
    formatted_channel_revenue["Current Budget"] = formatted_channel_revenue["Current Budget"].apply(lambda x: f"${x:.2f}")
    formatted_channel_revenue["Optimized Budget"] = formatted_channel_revenue["Optimized Budget"].apply(lambda x: f"${x:.2f}")
    formatted_channel_revenue["Change"] = formatted_channel_revenue["Change"].apply(lambda x: f"${x:.2f}")
    formatted_channel_revenue["Change %"] = formatted_channel_revenue["Change %"].apply(lambda x: f"{x:.1f}%")
    
    # Show the data table
    st.dataframe(formatted_channel_revenue, use_container_width=True)
    
    # Visualize channel revenue comparison 
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    plot_data = channel_revenue.pivot(
        index="Channel", 
        columns="Forecast Type", 
        values="Revenue"
    ).reset_index()
    
    # Melt the data for seaborn
    plot_data_melted = pd.melt(
        plot_data,
        id_vars=["Channel"],
        value_vars=["Current Budget", "Optimized Budget"],
        var_name="Budget Type",
        value_name="Revenue"
    )
    
    # Create the comparison bar chart
    sns.barplot(x="Channel", y="Revenue", hue="Budget Type", data=plot_data_melted, ax=ax2)
    
    ax2.set_title("Revenue by Channel: Current vs Optimized Budget")
    ax2.set_ylabel("Revenue ($)")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig2)

# ROAS Forecast Tab
with forecast_tab2:
    # Calculate daily ROAS by dividing total revenue by total spend for each day
    roas_by_date = forecast_data.groupby(["Date", "Forecast Type"]).agg({
        "Revenue": "sum",
        "Spend": "sum"
    }).reset_index()
    roas_by_date["ROAS"] = roas_by_date["Revenue"] / roas_by_date["Spend"]
    
    # Create ROAS timeline plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data with different styles for actual vs forecast
    for forecast_type, color, linestyle in [
        ("Actual", "black", "-"), 
        ("Current Budget", "blue", "--"), 
        ("Optimized Budget", "green", "--")
    ]:
        data = roas_by_date[roas_by_date["Forecast Type"] == forecast_type]
        ax.plot(data["Date"], data["ROAS"], 
                label=forecast_type, 
                color=color, 
                linestyle=linestyle,
                marker="o" if forecast_type == "Actual" else None,
                linewidth=2)
    
    # Add vertical line to separate actual and forecast
    last_actual_date = forecast_data[forecast_data["Forecast Type"] == "Actual"]["Date"].max()
    ax.axvline(x=last_actual_date, color="gray", linestyle="-.", alpha=0.5)
    ax.text(last_actual_date, ax.get_ylim()[1]*0.9, "Forecast Start", 
            ha="center", va="center", backgroundcolor="white", fontsize=10)
    
    # Formatting
    ax.set_title("ROAS Timeline: Past Week + Next 2 Weeks Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("ROAS")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis to show dates nicely
    fig.autofmt_xdate()
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Summary metrics
    st.subheader("ROAS Summary")
    
    # Calculate average ROAS for each period
    actual_roas = roas_by_date[roas_by_date["Forecast Type"] == "Actual"]["ROAS"].mean()
    current_forecast_roas = roas_by_date[roas_by_date["Forecast Type"] == "Current Budget"]["ROAS"].mean()
    optimized_forecast_roas = roas_by_date[roas_by_date["Forecast Type"] == "Optimized Budget"]["ROAS"].mean()
    
    improvement = optimized_forecast_roas - current_forecast_roas
    improvement_pct = (improvement / current_forecast_roas) * 100 if current_forecast_roas > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Past Week Avg ROAS", f"{actual_roas:.2f}x")
    col2.metric("Forecast with Current Budget", f"{current_forecast_roas:.2f}x")
    col3.metric("Forecast with Optimized Budget", f"{optimized_forecast_roas:.2f}x", 
               delta=f"+{improvement:.2f}x ({improvement_pct:.1f}%)")

# Budget Utilization Forecast Tab
with forecast_tab3:
    # Calculate daily budget utilization by channel
    util_by_date = forecast_data.groupby(["Date", "Forecast Type"])["Budget Utilization"].mean().reset_index()
    
    # Create budget utilization timeline plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data with different styles for actual vs forecast
    for forecast_type, color, linestyle in [
        ("Actual", "black", "-"), 
        ("Current Budget", "blue", "--"), 
        ("Optimized Budget", "green", "--")
    ]:
        data = util_by_date[util_by_date["Forecast Type"] == forecast_type]
        ax.plot(data["Date"], data["Budget Utilization"], 
                label=forecast_type, 
                color=color, 
                linestyle=linestyle,
                marker="o" if forecast_type == "Actual" else None,
                linewidth=2)
    
    # Add vertical line to separate actual and forecast
    last_actual_date = forecast_data[forecast_data["Forecast Type"] == "Actual"]["Date"].max()
    ax.axvline(x=last_actual_date, color="gray", linestyle="-.", alpha=0.5)
    ax.text(last_actual_date, ax.get_ylim()[1]*0.9, "Forecast Start", 
            ha="center", va="center", backgroundcolor="white", fontsize=10)
    
    # Add a horizontal line at 100% utilization
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5)
    ax.text(ax.get_xlim()[0], 100, "100% Utilization", 
            ha="left", va="bottom", color="red", fontsize=10)
    
    # Formatting
    ax.set_title("Budget Utilization: Past Week + Next 2 Weeks Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Budget Utilization (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Format x-axis to show dates nicely
    fig.autofmt_xdate()
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Add channel-specific utilization view
    st.subheader("Budget Utilization by Channel")
    
    # Calculate budget utilization by channel for each forecast type
    channel_util = forecast_data.groupby(["Channel", "Forecast Type"])["Budget Utilization"].mean().reset_index()
    
    # Create a pivot table
    channel_util_pivot = channel_util.pivot(index="Channel", columns="Forecast Type", values="Budget Utilization")
    
    # Add a column for ROAS performance
    channel_roas = forecast_data[forecast_data["Forecast Type"] == "Actual"].groupby("Channel")["ROAS"].mean()
    target_roas_series = pd.Series(target_roas)
    
    channel_util_pivot["ROAS Performance"] = (channel_roas / target_roas_series * 100)
    
    # Format the table
    formatted_util = channel_util_pivot.copy()
    for col in formatted_util.columns:
        formatted_util[col] = formatted_util[col].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(formatted_util, use_container_width=True)
    
    # Summary metrics
    st.subheader("Budget Utilization Summary")
    
    # Calculate average utilization for each period
    actual_util = util_by_date[util_by_date["Forecast Type"] == "Actual"]["Budget Utilization"].mean()
    current_forecast_util = util_by_date[util_by_date["Forecast Type"] == "Current Budget"]["Budget Utilization"].mean()
    optimized_forecast_util = util_by_date[util_by_date["Forecast Type"] == "Optimized Budget"]["Budget Utilization"].mean()
    
    improvement = optimized_forecast_util - current_forecast_util
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Past Week Avg Utilization", f"{actual_util:.1f}%")
    col2.metric("Forecast with Current Budget", f"{current_forecast_util:.1f}%")
    col3.metric("Forecast with Optimized Budget", f"{optimized_forecast_util:.1f}%", 
               delta=f"{improvement:.1f}%") 