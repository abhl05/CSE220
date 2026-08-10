import numpy as np
import matplotlib.pyplot as plt

def plot_branch_performance():
    # Provided Data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    north = np.array([120, 135, 128, 142, 150, 158, 162, 170, 168, 180, 185, 195])
    south = np.array([110, 118, 125, 130, 138, 145, 150, 155, 160, 168, 172, 180])
    central = np.array([100, 108, 115, 120, 130, 140, 148, 152, 158, 165, 170, 175])

    # Figure Settings
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Company Branch Performance Analysis (2025)", fontsize=16)

    # --- Plot 1 (Top Left): Monthly Profit Line Plot ---
    axs[0, 0].plot(months, north, 'b-', label='North')
    axs[0, 0].plot(months, south, 'g--', label='South')
    axs[0, 0].plot(months, central, 'r:', label='Central')
    axs[0, 0].set_title('Monthly Branch Profit (2025)')
    axs[0, 0].set_xlabel('Month')
    axs[0, 0].set_ylabel('Profit (in thousand dollars)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Plot 2 (Top Right): December Profit Comparison ---
    branches = ['North', 'South', 'Central']
    dec_profits = [north[-1], south[-1], central[-1]]
    axs[0, 1].bar(branches, dec_profits, color=['blue', 'green', 'red'])
    axs[0, 1].set_title('December Profit Comparison')
    axs[0, 1].set_xlabel('Branch')
    axs[0, 1].set_ylabel('Profit (in thousand dollars)')
    axs[0, 1].grid(axis='y')

    # --- Plot 3 (Bottom Left): North Branch Profit Distribution ---
    axs[1, 0].scatter(months, north, color='blue', s=60)
    axs[1, 0].set_title('North Branch Profit Distribution')
    axs[1, 0].set_xlabel('Month')
    axs[1, 0].set_ylabel('Profit (in thousand dollars)')
    axs[1, 0].grid(True)

    # --- Plot 4 (Bottom Right): Quarterly Branch Profit ---
    # this is a stacked bar chart, so we need to calculate the bottom for each stack
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Slicing the arrays and summing per quarter
    north_q = [np.sum(north[0:3]), np.sum(north[3:6]), np.sum(north[6:9]), np.sum(north[9:12])]
    south_q = [np.sum(south[0:3]), np.sum(south[3:6]), np.sum(south[6:9]), np.sum(south[9:12])]
    central_q = [np.sum(central[0:3]), np.sum(central[3:6]), np.sum(central[6:9]), np.sum(central[9:12])]

    axs[1, 1].bar(quarters, north_q, label='North', color='blue')
    axs[1, 1].bar(quarters, south_q, bottom=north_q, label='South', color='green') 
    # bottom parameter stacks the South bars on top of North bars
    
    # Calculate bottom offset for Central stack (North + South)
    bottom_central = np.add(north_q, south_q)
    axs[1, 1].bar(quarters, central_q, bottom=bottom_central, label='Central', color='red')
    
    axs[1, 1].set_title('Quarterly Branch Profit')
    axs[1, 1].set_xlabel('Quarter')
    axs[1, 1].set_ylabel('Profit (in thousand dollars)')
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y')

    # Arrange subplots neatly
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_branch_performance()