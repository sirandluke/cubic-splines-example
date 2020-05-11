import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
import pandas as pd

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
  return xCors, yCors, np.array(divided_diff_coef)

###
# Calculates Cubic Spline Interpolation of Sampled points
# @param -> xCors is empty list of x coordinates
# @param -> yCors is list of empty y coordinates
# @ return -> list of equidistant xCors, the corresponding yCors, and divided
# difference coefficients
def Spline_Interpolation(xCors, yCors):
  xCors = np.linspace(-5, 5, 15)  # create 15 equidistant points in [-5,5]
  yCors = get_y_points(xCors)

  spline = interpolate.splrep(xCors, yCors) # returns spline curve as tuplee of coefficients 
  return xCors, yCors, spline

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
print("The Newton Interpolation Coefficients are: ")
for coeff in coor_and_coeff[2]:
  print(coeff)


xCors = []
yCors = []
coor_and_spline = Spline_Interpolation(xCors, yCors)
print("The Spline Interpolation Coefficients are: ")
print(coor_and_spline[2])


"""
# create line plot
df=pd.DataFrame({
                  'x': range(0,15),
                  'y1': coor_and_spline[1],
                  'y2': coor_and_coeff[2]
                })
        
#a = {'x': range(0,15), 'y1': coor_and_spline[1], 'y2': coor_and_coeff[2], 'y3': coor_and_spline[2]}
#df = pd.DataFrame.from_dict(a, orient='index')
#df.transpose()
plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2, label="f")   # add original curve to plot
plt.plot( 'x', 'y2', data=df, marker='', color='red', linewidth=2, label="p")  # add interpolating coefficients to plot
"""


x = np.linspace(-5, 5, 15) 
y = (coor_and_spline[1])
f2 = interp1d(x, y, kind='cubic')
plt.plot(x, y, 'o', color='red', linewidth=2, label="f")
plt.plot(x,coor_and_coeff[2], marker='', color='blue', linewidth=2, label="p" )
plt.plot(x, f2(x), '--', color='green', label="S")
plt.legend()
plt.show()
