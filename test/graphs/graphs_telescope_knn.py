import matplotlib.pyplot as plt

data_x = [1, 2, 3, 5, 7, 9, 11, 13, 15]
data_y1 = [0.7933, 0.8099, 0.8143, 0.8180, 0.8195, 0.8228, 0.8239, 0.8128]
data_y2 = [0.7876, 0.7901, 0.8117, 0.8126, 0.8150, 0.8218, 0.8168, 0.8170]

plt.plot(data_x[0:8], data_y1[0:8] ,color ="red", label = "No Weighting")
plt.plot(data_x[0:8], data_y2[0:8] , color = "blue", label = "Distance Weighting")

plt.legend()
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Distance Weighting VS No Weighting")

# plt.hlines(0, -.25, .75)
# plt.vlines(0, -1, 1)

# line_x = [-.15, -.1, 0, .1, .15]
# line_y = [3.54, 2.36, 0, -2.36, -3.54]
#
# plt.plot(line_x, line_y)
plt.show()
