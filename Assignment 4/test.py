Employees=[]
#adding 4 employees names
Employees.append("John")
Employees.append("Sarah")
Employees.append("Tim")
Employees.append("Jane")
HR=[]
while True:
    name = input("Enter the employee name: ")
    if name=="STOP":
        break
    HR.append(name)

#finding common employees
common=[]
for name in HR:
    if name in Employees:
        common.append(name)
print("Count of common employees: ",len(common))
print("Common Employees: ",common)
updated_list=[]
#union of Employees and HR
updated_list=Employees+HR
#removing duplicates
updated_list=list(set(updated_list))
print("Updated list of employees: ",updated_list)
#sorting the list
updated_list.sort()
print("Sorted list of employees: ",updated_list)

#Q2
prices=[10,20,30,40,50]
discounted=[x-x*0.3 for x in prices]
print("Discounted prices: ",discounted)
print("Original prices: ",prices)
max_price=max(prices)
print("Maximum price: ",max_price)
