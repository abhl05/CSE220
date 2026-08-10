import numpy as np
import sys

def analyze_sales():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    s_salespersons = int(input_data[0])
    m_products = int(input_data[1])
    
    idx = 2
    
    # Parse Sales Matrix
    A = []
    for _ in range(s_salespersons):
        row = [int(x) for x in input_data[idx : idx + m_products]]
        A.append(row)
        idx += m_products
    A = np.array(A) # Convert list of lists to a numpy array for easier manipulation
    
    # Parse Target Sales Vector
    target_sales = np.array([int(x) for x in input_data[idx : idx + m_products]])
    idx += m_products
    
    # Parse K
    k = int(input_data[idx])
    
    # 1. Compute percentage matrix
    P = 100 * A / target_sales
    
    print("Percentage Matrix")
    for row in P:
        print(" ".join([f"{val:.2f}" for val in row])) # val:.2f formats the float to 2 decimal places
        
    # 2. Salesperson summary
    print("\nSalesperson Summary")
    sp_averages = []
    for i in range(s_salespersons):
        avg = np.mean(P[i])
        sp_averages.append(avg)
        best_prod = np.argmax(P[i]) 
        print(f"Salesperson {i}: Average = {avg:.2f} Best Product = {best_prod}")
        
    # 3. Product summary
    print("\nProduct Summary")
    for j in range(m_products):
        avg = np.mean(P[:, j])
        top_sp = np.argmax(P[:, j])
        print(f"Product {j}: Average = {avg:.2f} Top Salesperson = {top_sp}")
        
    # 4. Top K salespersons
    print(f"\nTop {k} Salespersons")
    # np.argsort returns ascending order, so we slice [::-1] to reverse it to descending
    # [::-1] reverses the order to get descending, [:k] takes the top k indices
    # here, argsort means we are sorting the indices of sp_averages based on their values
    top_k_indices = np.argsort(sp_averages)[::-1][:k]
    for sp_id in top_k_indices:
        print(sp_id)
        
    # 5. Grades count
    print("\nGrade Count")
    excellent = np.sum(P >= 90)
    good = np.sum((P >= 75) & (P < 90))
    average_grade = np.sum((P >= 60) & (P < 75))
    poor = np.sum(P < 60)
    
    print(f"Excellent: {excellent}")
    print(f"Good: {good}")
    print(f"Average: {average_grade}")
    print(f"Poor: {poor}")

if __name__ == "__main__":
    analyze_sales()