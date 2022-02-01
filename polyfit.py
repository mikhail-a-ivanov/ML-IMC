import numpy as np
from numpy.linalg import inv
from numpy import matmul

def leastSquares(x_input, y_input, powers=[0, 1], start=0, quiet=True):
    """
    Performs a least-squares polynomial regression
    with an arbitrary number of terms as well as their degrees.
    """
    x = x_input[np.where(x_input > start)]
    y = y_input[np.where(x_input > start)]
    if not quiet:
        if len(x) != len(x_input):
            print(f'Starting from x = {x[0]}')
    X = []
    for power in powers:
        X.append(x**power)
    X = np.stack(X).T
    p = matmul(matmul(inv(matmul(X.T, X)), X.T), y) # regression coefficients
    if quiet:
        print(f'Min/Max polynomial degrees: {min(powers)} / {max(powers)}')
    else:
        print(f'Number of coefficients: {len(powers)}')
        print(f'Polynomial degrees: {powers}')
    
    y_fit = matmul(p, X.T) # predicted values
    SSE = np.sum((y - y_fit)**2) # squared estimate of errors
    TSS = np.sum((y - np.mean(y))**2) # total sum of squares
    R2 = 1 - SSE/TSS # coefficient of determination
    print(f'R^2 = {R2:.6f}')
    if not quiet:
        print(f'Optimized parameters: {p}')
    
    return((x, y_fit, p, R2))

def lmpTablePolynomial(name, powers, p, r_min=0.000001, r_max=15, 
                       r_boundary=3, Npoints=3001, force_constant=100):
    """
    Creates LAMMPS table potential file from a polynomial fit
    of a potential.
    
    It is assumed that the input potential is in kJ/mol.
    
    Quadratic fit is used for the [r_min, r_boundary) region.
    """
    cal_to_J = 4.184 # There is 4.184 J in 1 cal
    
    print(f'Writing {name}.table file...')
    print(f'N points = {Npoints}')
    print(f'Using polynomial fit for R in range[{r_boundary}, {r_max}]')
    print(f'Using quadratic extrapolation for R < {r_boundary} with k = {force_constant}')
    
    r = np.linspace(r_max/(Npoints - 1), r_max, (Npoints - 1))
    table_R = np.concatenate(([r_min], r)) # distance values
    
    table_N = np.arange(1, len(table_R) + 1) # table of indices
    
    table_R_poly = table_R[np.where(table_R >= r_boundary)[0]]
    
    table_R_extra = table_R[np.where(table_R < r_boundary)[0]]
    
    R_poly = []
    for power in powers:
        R_poly.append(table_R_poly**power)
    R_poly = np.stack(R_poly).T
    
    # predicted and extrapolated potential values in kcal/mol
    table_V_poly = np.matmul(p, R_poly.T)/cal_to_J
    table_V_extra = force_constant/2 * (table_R_extra - r_boundary)**2 + table_V_poly[0]
    
    table_V = np.concatenate((table_V_extra, table_V_poly))
    
    derivative_powers = powers - 1
    Rderivative_poly = []
    for power in derivative_powers:
        Rderivative_poly.append(table_R_poly**power)
    Rderivative_poly = np.stack(Rderivative_poly).T
    
    # predicted and extrapolated force values
    table_F_poly = np.matmul(-p*powers, Rderivative_poly.T)/cal_to_J
    table_F_extra = -force_constant * (table_R_extra - r_boundary) + table_F_poly[0]
    
    table_F = np.concatenate((table_F_extra, table_F_poly))
    
    header = f'{name}\nN {len(table_R)}\n'
    
    # Save table into the file
    table_data = np.stack((table_N, table_R, table_V, table_F)).T
    np.savetxt(f'{name}.table', table_data, fmt='%i  %.6f  %.6f  %.6f',
          header=header, comments='')
    
    return((header, table_N, table_R, table_V, table_F))
