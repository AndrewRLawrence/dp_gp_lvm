"""
This module analyses the results for the concatenated walk with normal sequence for subject 35, sequence 1 from CMU and
walk with the legs swapped.
"""

from visualisation.plotters import plot_ard, plot_phi

from matplotlib2tikz import save as tikz_save
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sys import path

# Define paths.
absolute_path = [ap for ap in path if 'aistats_2019' in ap]
data_path = absolute_path[-1] + '/test/data/cmu_mocap/'
results_path = absolute_path[-1] + '/test/results/'

# Analyse results.
gpdp_file = results_path + 'gpdp_walk_legs_swapped_mocap_gpu_run_3.npz'
results = np.load(gpdp_file)
x_mean = results['x_mean']
x_u = results['x_u']
assignments = results['assignments']
ard_atoms = results['gamma_atoms']
ard_weights = results['ard_weights']
largest_ard = np.mean(ard_weights, axis=0).argsort()[-3:][::-1]
largest_clusters = np.mean(assignments, axis=0).argsort()[-3:][::-1]

num_output_dimensions, num_latent_dimensions = ard_weights.shape

# plot_file = 'weight_file.npz'
# np.savez(plot_file, ard_weights=ard_weights[::3, :], dp_assignments=assignments[::3, :])  # Subsample as DP mask was 3.

# print(ard_atoms[largest_clusters[0], largest_ard[2]])
# print(ard_atoms[largest_clusters[1], largest_ard[2]])
# print(ard_atoms[largest_clusters[2], largest_ard[2]])

latent_space = np.vstack((x_mean[:, largest_ard[0]], x_mean[:, largest_ard[1]], x_mean[:, largest_ard[2]])).T
ard_weights_3 = np.vstack((ard_weights[:, largest_ard[0]],
                           ard_weights[:, largest_ard[1]],
                           ard_weights[:, largest_ard[2]])).T

# print(np.mean(ard_weights_3[:, -1]))
# print(np.std(ard_weights_3[:, -1], ddof=1))
# print(np.max(ard_weights_3[:, -1]) - np.min(ard_weights_3[:, -1]))
scaler2 = StandardScaler()
x1 = scaler2.fit_transform(ard_weights_3[:, -1].reshape(-1, 1))
# print(x1)
# quit(0)

# Create plots.
frames = np.arange(x_mean.shape[0])

plot.figure(figsize=(12, 7))
ard_0 = plot.scatter(frames, x_mean[:, largest_ard[0]], c='C0')
ard_1 = plot.scatter(frames, x_mean[:, largest_ard[1]], c='C1')
ard_2 = plot.scatter(frames, x_mean[:, largest_ard[2]], c='k')
plot.legend(handles=[ard_0, ard_1, ard_2],
            labels=['X[{0}]'.format(largest_ard[0]), 'X[{0}]'.format(largest_ard[1]), 'X[{0}]'.format(largest_ard[2])])
plot.xlabel('Frame')
plot.ylabel('X')
plot.title('Mean of q(X[{0},{1},{2}]) ordered by frame number'.format(largest_ard[0], largest_ard[1], largest_ard[2]))
# if not isfile('cmu_all_x.pdf'):
#     plot.savefig('cmu_all_x.pdf', bbox_inches='tight')
# if not isfile('cmu_all_x.tex'):
#     tikz_save('cmu_all_x.tex')

plot.figure(figsize=(12, 7))
plot.scatter(frames, x_mean[:, largest_ard[0]], c='C0')
plot.xlabel('Frame')
plot.ylabel('X[{}]'.format(largest_ard[0]))
plot.title('Mean of q(X[{0}]) ordered by frame number'.format(largest_ard[0]))
# if not isfile('cmu_x{0}.pdf'.format(largest_ard[0])):
#     plot.savefig('cmu_x{0}.pdf'.format(largest_ard[0]), bbox_inches='tight')
# if not isfile('cmu_x{0}.tex'.format(largest_ard[0])):
#     tikz_save('cmu_x{0}.tex'.format(largest_ard[0]))

plot.figure(figsize=(12, 7))
plot.scatter(frames, x_mean[:, largest_ard[1]], c='C1')
plot.xlabel('Frame')
plot.ylabel('X[{}]'.format(largest_ard[1]))
plot.title('Mean of q(X[{0}]) ordered by frame number'.format(largest_ard[1]))
# if not isfile('cmu_x{0}.pdf'.format(largest_ard[1])):
#     plot.savefig('cmu_x{0}.pdf'.format(largest_ard[1]), bbox_inches='tight')
# if not isfile('cmu_x{0}.tex'.format(largest_ard[1])):
#     tikz_save('cmu_x{0}.tex'.format(largest_ard[1]))

plot.figure(figsize=(12, 7))
plot.scatter(frames, x_mean[:, largest_ard[2]], c='k')
plot.xlabel('Frame')
plot.ylabel('X[{}]'.format(largest_ard[2]))
plot.title('Mean of q(X[{0}]) ordered by frame number'.format(largest_ard[2]))
# if not isfile('cmu_x{0}.pdf'.format(largest_ard[2])):
#     plot.savefig('cmu_x{0}.pdf'.format(largest_ard[2]), bbox_inches='tight')
# if not isfile('cmu_x{0}.tex'.format(largest_ard[2])):
#     tikz_save('cmu_x{0}.tex'.format(largest_ard[2]))

# Plot ARD weights.
normalised_ard_weights = ard_weights / np.max(ard_weights)
minmax_scaler = MinMaxScaler()
# minmax_ard_weights = minmax_scaler.fit_transform(ard_weights.flatten().reshape(-1, 1))
# minmax_ard_weights = np.reshape(minmax_ard_weights, (num_output_dimensions, num_latent_dimensions))

mean_ard0 = np.mean(ard_weights[:, 0])
mean_ard1 = np.mean(ard_weights[:, 1])
mean_ard3 = np.mean(ard_weights[:, 3])
ratio_ard10 = np.mean(ard_weights[:, 1]) / np.mean(ard_weights[:, 0])
ratio_ard30 = np.mean(ard_weights[:, 3]) / np.mean(ard_weights[:, 0])

minmax_ard_weights0 = minmax_scaler.fit_transform(ard_weights[:, 0].reshape(-1, 1))
minmax_ard_weights1 = minmax_scaler.fit_transform(ard_weights[:, 1].reshape(-1, 1))
minmax_ard_weights3 = minmax_scaler.fit_transform(ard_weights[:, 3].reshape(-1, 1))
latent_factorisation = np.concatenate((minmax_ard_weights0.T,
                                       ratio_ard10 * minmax_ard_weights1.T,
                                       np.zeros((1, num_output_dimensions)),
                                       ratio_ard30 * minmax_ard_weights3.T,
                                       np.zeros((6, num_output_dimensions))), axis=0)

# latent_factorisation = np.concatenate((np.reshape(ard_weights[:, 0] / np.max(ard_weights[:, 0]), (-1, 1)),
#                                        np.reshape(ard_weights[:, 1] / np.max(ard_weights[:, 0]), (-1, 1)),
#                                        np.zeros((num_output_dimensions, 1)),
#                                        np.reshape(ard_weights[:, 3] / np.max(ard_weights[:, 0]), (-1, 1)),
#                                        np.zeros((num_output_dimensions, 6))), axis=1)

# standard_scaler = StandardScaler()
# normalised_ard_weights = np.square(np.transpose(standard_scaler.fit_transform(np.sqrt(ard_weights).T)))
# normalised_ard_weights = np.square(standard_scaler.fit_transform(np.sqrt(ard_weights)))

ard_fig = plot.figure(figsize=(18, 11))
# plot_ard(latent_factorisation.T, fig=ard_fig)
plot_ard(np.sqrt(latent_factorisation.T), fig=ard_fig)
# plot_ard(latent_factorisation, fig=ard_fig)
# plot_ard(normalised_ard_weights, fig=ard_fig)
# plot.title('Learned ARD Weights for each Joint')
plot.title('Latent Factorisation for each Joint')
ax = plot.gca()
plot.xlabel('')
ax.xaxis.set_ticks(np.arange(1, num_output_dimensions, 3))
ax.set_xticklabels([])
labels = ['Root', 'Left Hip', 'Left Femur', 'Left Tibia', 'Left Foot', 'Left Toes',
          'Right Hip', 'Right Femur', 'Right Tibia', 'Right Foot', 'Right Toes',
          'Lower Back', 'Upper Back', 'Thorax', 'Lower Neck', 'Upper Neck', 'Head',
          'Left Clavicle', 'Left Humerus', 'Left Radius', 'Left Wrist', 'Left Hand', 'Left Fingers', 'Left Thumb',
          'Right Clavicle', 'Right Humerus', 'Right Radius', 'Right Wrist', 'Right Hand', 'Right Fingers', 'Right Thumb']
ax.set_xticklabels(labels, rotation='vertical')
if not isfile('cmu_latent_factorisation.pdf'):
    plot.savefig('cmu_latent_factorisation.pdf', bbox_inches='tight')
if not isfile('cmu_latent_factorisation.tex'):
    tikz_save('cmu_latent_factorisation.tex')


# Show plots.
plot.show()

# # Create new 3D axes in the provided figure.
# fig = plot.figure()
# ax = Axes3D(fig)
#
#
# x_coordinates = x_mean[:, largest_ard[0]]
# y_coordinates = x_mean[:, largest_ard[1]]
# z_coordinates = x_mean[:, largest_ard[2]]
#
# ax.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
# ax.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
# ax.set_zlim(np.min(z_coordinates), np.max(z_coordinates))
#
# ax.set_xlabel('X[{}]'.format(largest_ard[0]))
# ax.set_ylabel('X[{}]'.format(largest_ard[1]))
# ax.set_zlabel('X[{}]'.format(largest_ard[2]))
#
# ax.scatter(x_coordinates, y_coordinates, z_coordinates, c='k', marker='o')
# ax.scatter(x_u[:, largest_ard[0]], x_u[:, largest_ard[1]], x_u[:, largest_ard[2]], c='b', marker='o')
#
# plot.show()

# ard_fig = plot.figure()
# plot_ard(ard_weights_3, fig=ard_fig)

# phi_fig = plot.figure()
# plot_phi(assignments, fig=phi_fig)

# plot.figure()
# plot.hist(latent_space[:, 2], bins=50)
# plot.show()

# # Create new 3D axes in the provided figure.
# fig = plot.figure()
# ax = Axes3D(fig)
#
# largest_ard = ard_weights.argsort()[-3:][::-1]
# x_coordinates = x_mean[:, largest_ard[0]]
# y_coordinates = x_mean[:, largest_ard[1]]
# z_coordinates = x_mean[:, largest_ard[2]]

# ax.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
# ax.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
# ax.set_zlim(np.min(z_coordinates), np.max(z_coordinates))
#
# ax.set_xlabel('X[{}]'.format(largest_ard[0]))
# ax.set_ylabel('X[{}]'.format(largest_ard[1]))
# ax.set_zlabel('X[{}]'.format(largest_ard[2]))
#
# ax.scatter(x_coordinates, y_coordinates, z_coordinates, c='k', marker='o')
# ax.scatter(x_u[:, largest_ard[0]], x_u[:, largest_ard[1]], x_u[:, largest_ard[2]], c='b', marker='o')
#
# plot.show()

# # Reset default graph before building new model graph. This speeds up script.
# tf.reset_default_graph()
# kernel = k_ard_rbf(gamma=ard_weights, alpha=signal_variance, beta=beta)
# session = tf.Session()
# plot_2d_latent_space(session, kernel, ard_weights, results['x_mean'], results['x_u'],
#                      var_img_dim=25, x_dims=None, on_threshold=0.05, fig=None, add_labels=True)
# plot.show()