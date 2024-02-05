# Author: Annbjørg Lingaas Marthinsen
# Date: 02.02.2024
# Description: Standardproject, TMA4106; Solving the heat equation numerically on a closed spatial interval. 

# Import libraries:
import math
import numpy as np
from scipy.misc import derivative

""" Information given:
f´(x) = (f(x+h)-f(x))/ h
f(x) = e**x
"""

# Question 1) Repeat the experiment with ℎ = 0.01, ℎ = 0.001, etc., and see what happens. How small can ℎ before it no longer work?
def deriv(x, h):
    biggerThanZero = 0
    totalRuns = 0
    while True:
        try:
            y = (math.exp(x + h) - math.exp(x))/ h
            h /= 10
            totalRuns += 1
            if y > 0:
                biggerThanZero += 1
        except: 
            return biggerThanZero, totalRuns
        

# Initial values
x = 1.5
h = 0.1

# Solve the equation
btz,tr = deriv(x,h)
print("First equation. Total runs before a crash: " + str(tr) + "\nRuns with a results above zero: " + str(btz))    













# Question 2) Repeat the experiment from last question, but now use the formula (f(x + h) - f(x - h))/ 2h = f´(x). Use Taylor series to explain.
# Solution without using the Taylor series:
def NewFormula(x, h):
    bigger_than_zero = 0
    total_runs = 0
    while True:
        try:
            y = (math.exp(x + h) - math.exp(x - h)) / (2 * h)
            h /= 10
            total_runs += 1
            if y > 0:
                bigger_than_zero += 1
        except Exception as e:
            return bigger_than_zero, total_runs
            
    

# Initial values
x = 1.5
h = 0.1

# Solve the equation
btz, tr = NewFormula(x,h)
print("Question 2. Total runs before a crash: " + str(tr) + "\nRuns with a results above zero: " + str(btz)) 



''' #This was my first try, but could not get this code to work. 
# Solution using the Taylor series:
'' Source: "Kalkulus og lineær algebra" by Arne Hole. s.272. 
The taylor series: f(a) + f´(a)(x-a)+(f´´(a)/2!)*(x-a)**2+(f´´´(a)/3!)*(x-a)**3+...
The point of the Taylor series is to calculate parts of the series and hope to find a system.''

# Import libraries needed
import numpy as np
import sympy as smp

# Define the function and variables
def Taylor_series(x, h):
    bigger_than_zero = 0
    total_runs = 0
    running = True
    x_sym = smp.symbols('x')

    # Function f(x) and its derivatives
    def f(x):
        return (smp.exp(x + h) - smp.exp(x - h)) / (2 * h)

    f_deriv = smp.diff(f(x_sym), x_sym)
    f_double_deriv = smp.diff(f_deriv, x_sym)

    f_func = smp.lambdify(x_sym, f(x_sym), 'numpy')
    f_deriv_func = smp.lambdify(x_sym, f_deriv, 'numpy')
    f_double_deriv_func = smp.lambdify(x_sym, f_double_deriv, 'numpy')

    while running:
        try:
            # Taylor series expansion for f(x + h)
            fx_plus_h = f_func(x) + h * f_deriv_func(x) + (h**2 / 2) * f_double_deriv_func(x) + smp.O(h**3)

            # Taylor series expansion for f(x - h)
            fx_minus_h = f_func(x) - h * f_deriv_func(x) + (h**2 / 2) * f_double_deriv_func(x) + smp.O(h**3)

            # Symmetric difference formula using Taylor series expansions
            y = (fx_plus_h - fx_minus_h) / (2 * h)

            h /= 10
            total_runs += 1

            if smp.limit(y, h, 0, dir="-") > 0:
                bigger_than_zero += 1


        except Exception as e:
            running = False
            print(f"Question 2. Solution with Taylor Series. Total runs before a crash: {total_runs}\nRuns with a result above zero: {bigger_than_zero}\nError: {e}")

# Initial values
x = 1.5
h = 0.1

# Solve the equation
Taylor_series(x, h)
'''



# Solution using the Taylor series (SECOND TRY):
''' Source: "Kalkulus og lineær algebra" by Arne Hole. s.272. 
The taylor series: f(a) + f´(a)(x-a)+(f´´(a)/2!)*(x-a)**2+(f´´´(a)/3!)*(x-a)**3+...
The point of the Taylor series is to calculate parts of the series and hope to find a system.'''

# Import libraries needed
import sympy as smp

# Define the function and variables
def second_taylor(x, h):
    bigger_than_zero = 0
    total_runs = 0
    while True:
        try:
            y = (smp.exp(x + h) - smp.exp(x - h)) / (2 * h)
            h /= 10
            total_runs += 1
            if y > 0:
                bigger_than_zero += 1
        except Exception as e:
            return bigger_than_zero, total_runs

# Initial values
x = 1.5
h = 0.1

# Solve the equation using the new formula
btz_new, tr_new = secound_taylor(x, h)
print("Question 2. Total runs before a crash with Taylor series:" + str(tr_new) + "\nRuns with results above zero: " + str(btz_new))

# Define the function and its derivative
x_sym = smp.symbols('x')
f = (smp.exp(x_sym + h) - smp.exp(x_sym - h)) / (2 * h)
f_deriv = smp.diff(f, x_sym)

# Check if the derivative is positive for the given x
if f_deriv.subs(x_sym, x) > 0:
    print("The derivative is positive for the given x.")
else:
    print("The derivative is not positive for the given x.")
















# Question 4) Solve the heat equation using a explicit scheme
'''Source: Shameel Abdulla, Youtube. 
Explicit methods are used to solve partial differential equations.'''

# Import libraries 
import matplotlib.pyplot as plt

# Parameters
h = 0.025
k = 0.025

# Defining spatial and time vectors, and boundry and initial counditions
x = np.arange(0, 1+h, h)
t = np.arange(0, 0.1+k, k )

boundry = [0, 0]
initial = np.sin(np.pi*x)

# Creating matrix T
n = len(x)
m = len(t)

T = np.zeros((n,m))
T[0,:] = boundry[0]
T[-1, :] = boundry[1]
T[:, 0] = initial

# Defining the function
def ExplicitHeat(k,h,m,n):
    factor = k/h**2
    for j in range(1, m):
      for i in range(1, n-1):
            T[i,j] = factor*T[i-1, j-1] + (1-2*factor)*T[i, j-1] + factor * T[i+1, j-1]
    return T

T = T.round(3)
print(f"Question 4. The results using the explicit scheme is: ", T)

# Plot
for j in range(m):
    plt.plot(x, T[:, j], label=f'Time step {j}')

plt.legend()
plt.xlabel("Position [x]")
plt.ylabel("Temperature [Degree C]")
plt.title("Question 4. Explicit method")
plt.show()


















# Question 5) The requirement 𝑘/ℎ² ≥ 1/2 is quite restrictive, and that's why we have implicit methods. They are a bit more challenging to implement and somewhat computationally heavier, but they avoid such hair-raising conditions on ℎ and 𝑘. Implement implicitly.
'''Source: "Kalkulus og lineær algebra" by Arne Hole. s.462 and Shameel Abdulla, Youtube. 
The implicit method are to solve partial different equations. They method tells you to solve a system of equations at each step, 
where the unknowns correspond to the values of the dependent variable at different spatial locations. 
This method gives room for larger time steps compared to explicit methods.'''

# Parameters
h = 0.25
k = 0.25

# Define spatial and time vectors, and boundry and initial counditions
x = np.arange(0,1+h, h)
t = np.arange(0,1+k, k)
BoundryIm = [0,0]
InitialIm = np.sin(np.pi*x)

# Create Matrix T
n = len(x)
m = len(t)
T = np.zeros((n,m))
T[0,:] = BoundryIm[0]
T[-1,:] = BoundryIm[1]
T[:,0] = InitialIm

# Define the function
def ImplicitHeat():
    factor = k/h**2
    A = np.diag([1+2*factor] * (n - 2), 0) + np.diag([-factor] * (n - 3), -1) + np.diag([-factor] * (n - 3), 1)

    for j in range(1, m):
        b = T[1:-1,j-1].copy()
        b[0] = b[0] + factor*T[0, j]
        b[-1] = b[-1] + factor*T[-1,j]
        solution = np.linalg.solve(A,b)
        T[1: -1, j] = solution
        print(solution)

    T.round(3)
    return T

# Run the code
result3 = ImplicitHeat()
print(f"Question 5. The results with the implicit method is: ", result3)

# Plot
for j in range(m):
    plt.plot(x, T[:, j], label=f'Time step {j}')

plt.legend()
plt.xlabel("Position [x]")
plt.ylabel("Temperature [Degree C]")
plt.title("Question 5. Implicit method ")
plt.show()
















# Question 6) But why do we have Crank-Nicolson? Implement Crank-Nicolson and compare the three methods for the same ℎ and 𝑘 
# with the analytical solution of the heat equation, using, for example, initial conditions 𝑓(𝑥) = sin x.
''' Respond to first question: The reason why we have the Crank-Nicolsen method is because both the explicit and implicit methods are 
working okay alone, but a combination of them both gives us a even more accuracy. The Crank-Nicolsen method is a combination of them both.'''

# Define variables 
h = 0.1
k = 0.025

# Define spatial and time vectors, and boundry and initial counditions
x = np.arange(0, 1+h, h)
t = np.arange(0, 0.1+k, k).round(3)
BC = [0,0]
IC = np.sin(np.pi*x)

# Creating matrix T
n = len(x)
m = len(t)
T = np.zeros((n,m))
T[0,:] = BC[0]
T[-1,:] = BC[1]
T[:,0] = IC

# Define the function
factor = k/h**2

A = np.diag([2+2*factor]*(n-2),0) + np.diag([-factor]*(n-3) , -1) + np.diag([-factor]*(n-3), 1)
B = np.diag([2-2*factor]*(n-2),0) + np.diag([factor]*(n-3) , -1) + np.diag([-factor]*(n-3), 1)

def CrankN():
    for j in range(0, m-1):
        b = T[1:-1, j].copy()
        b = np.dot(B, b)
        b[0] = b[0] + factor*(T[0,j]+T[0,j+1])
        b[-1] = b[-1] + factor*(T[-1,j] + T[-1, j+1])
        solution = np.linalg.solve(A,b)
        T[1:-1, j+1] = solution
        solution = T.round(2)
        print(f"Question 6. The solution of with the Crank-Nicolsen method is: ", solution)

CrankN()

# Plot
for j in range(m):
    plt.plot(x, T[:, j], label=f'Time step {j}')

plt.legend()
plt.xlabel("Position [x]")
plt.ylabel("Temperature [Degree C]")
plt.title("Question 6. Crank-Nicolsen")
plt.show()
