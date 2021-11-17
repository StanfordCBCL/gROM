import numpy as np
import sys
from stencil import *
import matplotlib.pyplot as plt

class Assembler:
    def __init__(self, resampled_geometry, stencil_size, training_data_fdr):
        self.resampled_geometry = resampled_geometry
        self.stencil_size = stencil_size
        self.training_data_fdr = training_data_fdr
        self.stencils_array = StencilsArray(resampled_geometry, stencil_size)
        self.stencils_array.load_models(training_data_fdr)
        self.bcs_indices = self.stencils_array.find_bcs_indices().astype(bool).squeeze()
        self.load_bounds()

    def load_bounds(self):
        self.bounds_pressure = np.load(self.training_data_fdr + '/bounds_pressure.npy')
        self.bounds_velocity = np.load(self.training_data_fdr + '/bounds_velocity.npy')

    def set_initial_conditions(self, pressure, velocity):
        self.initial_condition = self.stencils_array.generate_global_vector((pressure - self.bounds_pressure[0]) / (self.bounds_pressure[1] - self.bounds_pressure[0]),
                                                                            (velocity - self.bounds_velocity[0]) / (self.bounds_velocity[1] - self.bounds_velocity[0]))

    def evaluate_residual(self, vector, prev_vector, dt):
        residual = self.stencils_array.evaluate_models(vector, prev_vector, dt)
        # residual = models_output - vector
        residual[self.bcs_indices] = 0
        return residual

    def evaluate_jacobian(self, vector, prev_vector, dt):
        jacobian = self.stencils_array.evaluate_models_jacobian(vector, prev_vector, dt)
        # jacobian = self.stencils_array.evaluate_models_jacobian(vector, prev_vector, dt, False, jacobian)
        # jacobian = models_output - np.identity(models_output.shape[0])
        indices = np.where(self.bcs_indices)[0]
        for index in indices:
            jacobian[index,:] = jacobian[index,:] * 0
            jacobian[index,index] = 1
        return jacobian

    def set_exact_solutions(self, pressures, velocities):
        self.solutions = {}
        for t in pressures:
            self.solutions[t] = self.stencils_array.generate_global_vector((pressures[t] - self.bounds_pressure[0]) / (self.bounds_pressure[1] - self.bounds_pressure[0]),
                                                                           (velocities[t] - self.bounds_velocity[0]) / (self.bounds_velocity[1] - self.bounds_velocity[0]))
        print('hi')

    def solve(self, t0, T, dt):
        xs = [self.initial_condition]

        t = t0

        while t < T:
            t = np.round(t + dt, decimals = 6)
            print('time = ' + str(t))

            curx = np.copy(self.solutions[t]) # np.copy(xs[-1])
            # curx = np.copy(xs[-1])
            curx = self.apply_bcs(curx, t)

            curx = self.newtons_algorithm(curx,
                           lambda x: self.evaluate_residual(x, self.solutions[np.round(t-dt, decimals = 6)], dt),
                           lambda x: self.evaluate_jacobian(x, self.solutions[np.round(t-dt, decimals = 6)], dt),
                           tol = 1e-3,
                           maxit = 20)

            fig = plt.figure()
            ax = plt.axes()
            N = int(curx.size / 2)
            ax.plot(curx[0:N],'o-b')
            ax.plot(self.solutions[t][0:N],'o-r')
            ax.set_xlabel('arclength')
            ax.set_ylabel('pressure')
            ax.set_title(str(t))

            fig2 = plt.figure()
            ax2 = plt.axes()

            ax2.plot(curx[N:],'o-b')
            ax2.plot(self.solutions[t][N:],'o-r')
            ax2.set_xlabel('arclength')
            ax2.set_ylabel('flowrate')
            ax2.set_title(str(t))

            plt.show()

            xs.append(curx)

        self.xs = xs

    def apply_bcs(self, vector, t):
        vector[self.bcs_indices] = self.solutions[t][self.bcs_indices]
        return vector

    def newtons_algorithm(self, x0, res, jac, tol, maxit):
        x = x0
        R = res(x)
        err = np.linalg.norm(R)
        inerr = err

        numit = 0
        while err / inerr > tol and numit <= maxit:
            print('\t abs terr = ' + str(err) + ' rel err = ' + str(err/inerr))
            J = jac(x)
            x = x - np.linalg.solve(J, R)

            R = res(x)
            err = np.linalg.norm(R)
            numit = numit + 1
        print('\tfinal err = ' + str(err))

        if numit == maxit:
            sys.exit("Maximum number of Newton's iterations reached")
        return x
