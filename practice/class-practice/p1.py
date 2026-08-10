import sys

def manage_patient_records():
    # Read all standard input (you can also modify this to take a file path)
    input_data = sys.stdin.read().split() # reads all input and splits it into a list of strings
    if not input_data:
        return
    
    n_visits = int(input_data[0])
    idx = 1
    
    visits = []
    patients = {}
    
    # 1. Store every patient visit as a tuple
    for _ in range(n_visits):
        name = input_data[idx]
        severity = int(input_data[idx+1])
        treatment_time = int(input_data[idx+2])
        idx += 3
        
        visits.append((name, severity, treatment_time))
        
        # 2. Build dictionary to aggregate totals
        if name not in patients:
            patients[name] = {'total_time': 0, 'total_severity': 0} # dict of dict initialization
        
        patients[name]['total_time'] += treatment_time
        patients[name]['total_severity'] += severity
        
    # 3. Print the summary for every patient in alphabetical order
    sorted_patients = sorted(patients.keys()) # patients.keys() returns a list of the dictionary's keys, which are the patient names. sorted() sorts these names alphabetically.
    for name in sorted_patients:
        time = patients[name]['total_time']
        sev = patients[name]['total_severity']
        print(f"{name} {time} {sev}")
        
    # 4. Print the patient with the highest severity score
    # The lambda function uses (-severity, name) to ensure that we find the max severity
    # but in the event of a tie, the lexicographically smaller name is chosen.
    top_patient = min(patients.keys(), key=lambda k: (-patients[k]['total_severity'], k))
    
    print(f"TOP {top_patient} {patients[top_patient]['total_severity']}")

if __name__ == "__main__":
    manage_patient_records()