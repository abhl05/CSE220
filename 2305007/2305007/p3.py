class Employee:
    # Class Attribute
    company_name = "Global Tech" 
    
    def __init__(self, name, employee_id):
        self.name = name
        self.employee_id = employee_id
        self.salaries = []  
        
    def add_salary(self, salary):
        self.salaries.append(salary)
        
    def average_salary(self):
        if not self.salaries: return 0
        return sum(self.salaries) / len(self.salaries)
        
    def highest_salary(self):
        if not self.salaries: return 0
        return max(self.salaries)
        
    def annual_income(self):
        return sum(self.salaries)
        
    def __str__(self):
        return (f"Employee Name: {self.name}\n"
                f"Employee ID: {self.employee_id}\n"
                f"Average Salary: {self.average_salary():.2f}\n"
                f"Highest Salary: {self.highest_salary()}\n"
                f"Annual Income: {self.annual_income()}")

# Testing the implementation
if __name__ == "__main__":
    import random
    
    # 1. Create 5 Employee objects
    employees = [
        Employee("Alice", "E101"),
        Employee("Bob", "E102"),
        Employee("Charlie", "E103"),
        Employee("Diana", "E104"),
        Employee("Evan", "E105")
    ]
    
    # 2. Add 12 monthly salaries for each
    for emp in employees:
        for _ in range(12):
            emp.add_salary(random.randint(40000, 60000))
            
    # 3. Display info
    for emp in employees:
        print(emp)
        print("-" * 25)