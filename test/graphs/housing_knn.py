import matplotlib.pyplot as plt

data_x = [1, 2, 3, 5, 7, 9, 11, 13, 15]
data_y1 = [11.8145, 14.1755, 17.8826, 19.8877, 21.1720, 21.9965, 22.0676, 23.2193]
data_y2 = [11.3866,14.1291,17.4817,20.7933,22.0156, 23.0626, 23.4910, 24.6811,
]

plt.plot(data_x[0:8], data_y1[0:8] ,color ="red", label = "No Weighting")
plt.plot(data_x[0:8], data_y2[0:8] , color = "blue", label = "Distance Weighting")

plt.legend()
plt.xlabel("k")
plt.ylabel("Mean Squared Error")
plt.title("Distance Weighting VS No Weighting")

# plt.hlines(0, -.25, .75)
# plt.vlines(0, -1, 1)

# line_x = [-.15, -.1, 0, .1, .15]
# line_y = [3.54, 2.36, 0, -2.36, -3.54]
#
# plt.plot(line_x, line_y)
plt.show()
