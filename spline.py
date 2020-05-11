import numpy as np
import matplotlib.pyplot as plt

###
# Calculates Divided Difference Coefficients
# @param -> xCors is empty list of x coordinates
# @param -> yCors is list of empty y coordinates
# @ return -> list of equidistant xCors, the corresponding yCors, and divided
# difference coefficients
def Newton_Divided_Difference(xCors, yCors):
  xCors = np.linspace(-5, 5, 15)  # create 15 equidistant points in [-5,5]
  yCors = get_y_points(xCors)
  n = len(xCors)
  divided_diff_coef = yCors
  for i in range(1, n):
    for j in range(n-1, i-1, -1):
      divided_diff_coef[j] = float(yCors[j] - yCors[j-1]) / float(xCors[j]-xCors[j-i])
  return np.array(xCors), np.array(yCors), np.array(divided_diff_coef) 


###
# Calculates Divided Difference Coefficients
# @param -> xCors is np array of x coordinates
# @param -> yCors is np array of y coordinates
# @param -> div_diff_coef is np array of coefficients
# @ return -> list of equidistant xCors, the corresponding yCors, and divided
# difference coefficients
def Newton_Interpolation(xCors, yCors, div_diff_coef):
  n = len(xCors) - 1 # degree of polynomial
  ans = div_diff_coef[n]
  for i in range(1, n + 1):
    ans = a[n - i] + (x_data[n - k]) * p



def Spline_algrthm():
  return

# returns list of y coordinates
# @param -> xCors is the list of x coordinates
def get_y_points(xCors):
  sample = []
  for x in xCors:
    sample.append(polyFunc(x))
  return sample

###
# Function that computes 1/(1+x2)
# @param -> int xCor
# @ return -> pair of coordinates (xCor, yCor)
def polyFunc(xCor):
  yCor = 1/(1 + xCor**2)
  return yCor

#####################################
xCors = []
yCors = []
coor_and_coeff = Newton_Divided_Difference(xCors, yCors)
Newton_Interpolation(coor_and_coeff[0],coor_and_coeff[1], coor_and_coeff[2])
