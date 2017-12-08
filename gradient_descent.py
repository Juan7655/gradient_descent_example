import time
import pandas as pd


def run():
	data = pd.read_csv("mpg.csv")

	start_time = time.time()
	m_current = (data["mpg (n)"].mean()-data["mpg (n)"][0])/(data["weight (n)"].mean()-data["weight (n)"][0])
	b_current = data["mpg (n)"][0]-m_current*data["weight (n)"][0]
	final_error = 0

	for i in range(60000):
		[b_current, m_current, final_error] = step_gradient(b_current, m_current, data, 0.00000010405)
		print("m:" + str(m_current) + "	b:" + str(b_current) + "  E:" + str(final_error))
	print("Error: " + str(final_error))
	print("Execution time: %s seconds" % (time.time() - start_time))


def step_gradient(b_current, m_current, points, learning_rate):
	N = float(len(points) - 1)
	temp_points = points.copy()

	delta = (temp_points["mpg (n)"]-m_current*temp_points["weight (n)"]-b_current)
	error = (delta**2).sum() / N
	m_gradient = -(2/N) * (temp_points["weight (n)"] * delta).sum()
	b_gradient = -(2/N) * delta.sum()

	new_m = m_current - (learning_rate * m_gradient)
	new_b = b_current - (learning_rate * b_gradient)

	return [new_b, new_m, error]


if __name__ == '__main__':
	run()
