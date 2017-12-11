# Gradient descent example

In this code, I've used the gradient descent technique from machine learning to estimate the line of best fit 
for the given dataset points. The data file corresponds to the downloads and uninstalls from an Android application.
At first, the system is started with a line that goes from the first point to the center point
(`y = m*x + b` with `m = (y - y0) / (x - x0)` and `b = y - mx`).
Through 6000 steps, it gets pretty close to the actual line of best fit.

Expected value: **m=0.688318621674364, b=0.704989689999731**

Obtained result: **m=0.71661290544,  b=0.055726280524**

To run this code, you must have python installed. No other dependencies or libraries were used.
You just have to download the dataset file and run the code. The console will output values of m and b for each step.
Both files (code and dataset) must be in the same folder. Otherwise, you might have to change the path specified in the
code where the dataset is imported.
