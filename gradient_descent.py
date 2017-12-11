import time
import pandas
import matplotlib.pyplot as plt
import numpy as np

x_column = "weight (n)"
y_column = "mpg (n)"
iterations = 60000
result = []


def run():
	data = pandas.read_csv("mpg.csv")

	start_time = time.time()
	# starting point will not be random, but the line from the first point, to the mean point of the set
	m_current = (data[y_column].mean() - data[y_column][0]) / (data[x_column].mean() - data[x_column][0])
	b_current = data[y_column][0] - m_current * data[x_column][0]
	final_error = 0

	for i in range(iterations):
		[b_current, m_current, final_error] = step_gradient(b_current, m_current, data, 0.00000010405)
		result.append([m_current, b_current, final_error])
	res = pandas.DataFrame(data=result, columns=["m", "b", "error"])

	# x values for the trend lines in the graph
	x_val = np.array([i*(5300/1000) for i in range(1000)])

	# plotting final trend line
	y_val = m_current * x_val + b_current
	plt.plot(x_val, y_val, zorder=2, c='r', linewidth=2)

	# plotting trend line in each iteration
	# step size optimized to reduce excessive time
	for i in range(0, len(res), max(int(iterations/6000), 1)):
		y_val = res["m"][i] * x_val + res["b"][i]
		plt.plot(x_val, y_val, zorder=4, c='b', linewidth=.1)

	# base scatter graph (original data points)
	plt.scatter(data[x_column], data[y_column], zorder=1, s=3)
	plt.xlabel(x_column)
	plt.ylabel(y_column)
	plt.show()

	print("m:" + str(m_current) + "	b:" + str(b_current) + "  E:" + str(final_error))
	print("Execution time: %s seconds" % (time.time() - start_time))


def step_gradient(b_current, m_current, points, learning_rate):
	num_points = float(len(points) - 1)
	temp_points = points.copy()

	# base function of difference expected-obtained i.e. y-y' or y-(mx+b)
	delta = (temp_points[y_column]-m_current*temp_points[x_column]-b_current)
	# error calculated from MSE
	error = (delta**2).sum() / num_points
	# partial derivatives of error with respect to m and b respectively
	m_gradient = -(2 / num_points) * temp_points[x_column].dot(delta)
	b_gradient = -(2/num_points) * delta.sum()

	# updating m and b values from gradients
	new_m = m_current - (learning_rate * m_gradient)
	new_b = b_current - (learning_rate * b_gradient)

	return [new_b, new_m, error]


if __name__ == '__main__':
	run()
