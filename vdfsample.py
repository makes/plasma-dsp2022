import numpy as np

import vlsvtools

class VDFsampler:
    def __init__(self, n_spheres, n_points, r_max, distribution='radius'):
        self.n_spheres = n_spheres
        self.n_points = n_points
        self.r_max = r_max
        self.distribution = distribution
        self.spheres = self.__generate_spheres(n_spheres,
                                               n_points,
                                               r_max,
                                               distribution)

    # https://stackoverflow.com/a/26127012/287954
    @classmethod
    def fibonacci_sphere(samples=1000):
        points = np.empty((samples, 3))
        phi = np.pi * (3 - 5 ** 0.5)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = (1 - y * y) ** 0.5  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points[i] = np.array([x, y, z])

        return points

    def __generate_spheres(n_spheres, n_points, r_max, distribution):
        radii = np.linspace(r_max / n_spheres, r_max, n_spheres)
        if distribution == 'area':
            areas = 4 * np.pi * radii ** 2
            a_tot = np.sum(areas)
            points_per_sphere = np.trunc((areas / a_tot) * n_points).astype(int)
            points_per_sphere[:n_points - np.sum(points_per_sphere)] += 1
        elif distribution == 'radius':
            points_per_sphere = np.trunc((radii / np.sum(radii)) * n_points).astype(int)
            points_per_sphere[:n_points - np.sum(points_per_sphere)] += 1
        elif distribution == 'even':
            points_per_sphere = np.ones(n_spheres) * (n_points // n_spheres)
            points_per_sphere[-(n_points - np.sum(points_per_sphere)):] += 1
        else:
            raise ValueError(f'Unknown distribution {distribution}')
        spheres = []
        for n, radius in zip(points_per_sphere, radii):
            s = np.array(radius * VDFsampler.fibonacci_sphere(n))
            spheres.append(np.trunc(s).astype(int))
        return spheres

    def __sample_sphere(f, sphere):
        return f[sphere[:, 0], sphere[:, 1], sphere[:, 2]]