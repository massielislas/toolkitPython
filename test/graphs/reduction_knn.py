import matplotlib.pyplot as plt

data_x = [
    0,
    93,
    324,
    667,
    5529,
    6237,
    7633,
    9271,
    10152
]
data_y1 = [
    0.8466,
    0.8471,
    0.8461,
    0.8458,
    0.8287,
    0.8264,
    0.8223,
    0.8186,
    0.8099,
]
data_y2 = [11.3866, 14.1291, 17.4817, 20.7933, 22.0156, 23.0626, 23.4910, 24.6811,
           ]

plt.plot(data_x, data_y1, color="red")

plt.legend()
plt.xlabel("Instances Removed")
plt.ylabel("Accuracy")
plt.title("Accuracy with Instances Removed")

# plt.hlines(0, -.25, .75)
# plt.vlines(0, -1, 1)

# line_x = [-.15, -.1, 0, .1, .15]
# line_y = [3.54, 2.36, 0, -2.36, -3.54]
#
# plt.plot(line_x, line_y)
plt.show()
