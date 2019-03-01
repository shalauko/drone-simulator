# drone-simulator
Geometric Tracking Controller and Near-Hovering controller implemented in Python.

# Dependencies
- numpy
- scipy
- PIL
- matplotlib
- sys
- time
- csv
- control

Almoust all of these packages are preinstalled in Anaconda 1.9.6. Just install anaconda for Python 3 -> https://www.anaconda.com/

Additionally, you have to install only "control" package in version 0.8.1 at least -> https://python-control.readthedocs.io/en/0.8.1/

# How to use the code
After instalation required dependencies you have to run main.py in Python3. At begin of code set up which controller do you want to use and if GT, next choise full one or simplified. To choise which path generator should be used go to line 96 and uncomment one of showed lines.

For more details please read instructions inside scripts.

By default is set full Geometric Tracking Controller and RRT algorithm  for path.

# How to read the code inside:
- R 			- 	Rotation matrix 		[3x3]
- eta 			- 	Euler angles			[3x1]
- phi, theta, psi   	- 	roll, pith, yaw respectively	[scalar]
- omega			-	angular velocity in body frame	[3x1]
- u			-	steering thrust			[scalar]
- tau			-	steering torque			[3x1]
- f_ext			-	external forces			[3x1]
- tau_ext		-	external torques		[3x1]

All values started from "e_" are errors, e.g.:
- e_p 			-	position error
- e_p_dot		-	velosity error <=> error of first derivative of position <=> derivative of position error

All values with "_dot" are derivatives, e.g.:
- p			-	position
- p_dot			-	velocity <=> first derivative of position
- p_2dot		-	acceleration <=> second derivative of position
- p_3dot		-	jerk <=> third derivative of position
- p_4dot		-	snap <=> fourth derivative of position
- omega_dot		-	angular acceleration <=> first derivative of angular velocity
- etc.

All values with "_d" are desired values, e.g.:
- p_d_3dot		-	desired jerk
- psi_d			-	desired yaw
- etc.

All values with "_0" are initial values, e.g.:
- R_0			-	initial value of rotation matrix
- eta_0			-	initial values of Euler angles
- etc.