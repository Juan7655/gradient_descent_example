def run():
	with open("results-20170918-230306.csv") as file:
		data = file.readlines()

	m_current = 0
	b_current = 0
	final_error = 0

	for i in range(600):
		[b_current, m_current, final_error] = step_gradient(b_current, m_current, data, 0.0001)
		print("m:" + str(m_current) + "	|	b:" + str(b_current))
	print("Error: " + str(final_error))

def step_gradient(b_current, m_current, points, learning_rate):
	m_gradient = 0
	b_gradient = 0
	N = float(len(points) - 1)
	condition = False
	error = 0

	print("m:" + str(m_current) + "	|	b:" + str(b_current))
	for line in points:
		if condition:
				text = line.split(",")
				x = int(text[1])
				y = int(text[2])
				function = (y - m_current * x - b_current)/N
				error += (function**2) * N
				m_gradient -= 2 * x * function
				b_gradient -= 2 * function
		else:
				condition = True
	new_m = m_current - (learning_rate * m_gradient)
	new_b = b_current - (learning_rate * b_gradient)

	return [new_b, new_m, error]

if __name__ == '__main__':
    run()