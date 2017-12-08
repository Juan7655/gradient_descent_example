import time
import pandas as pd

x_column = "weight (n)"
y_column = "mpg (n)"
iterations = 60000


def run():
	data = pd.read_csv("mpg.csv")

	start_time = time.time()
	m_current = (data[y_column].mean()-data[y_column][0])/(data[x_column].mean()-data[x_column][0])
	b_current = data[y_column][0]-m_current*data[x_column][0]
	final_error = 0

	for i in range(iterations):
		[b_current, m_current, final_error] = step_gradient(b_current, m_current, data, 0.00000010405)
		print("m:" + str(m_current) + "	b:" + str(b_current) + "  E:" + str(final_error))
	print("Error: " + str(final_error))
	print("Execution time: %s seconds" % (time.time() - start_time))


def step_gradient(b_current, m_current, points, learning_rate):
	num_points = float(len(points) - 1)
	temp_points = points.copy()

	delta = (temp_points[y_column]-m_current*temp_points[x_column]-b_current)
	error = (delta**2).sum() / num_points
	m_gradient = -(2/num_points) * (temp_points[x_column] * delta).sum()
	b_gradient = -(2/num_points) * delta.sum()

	new_m = m_current - (learning_rate * m_gradient)
	new_b = b_current - (learning_rate * b_gradient)

	return [new_b, new_m, error]


if __name__ == '__main__':
	run()
