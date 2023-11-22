import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3])
print(a)
a = np.array([[1, 2], [3, 4]])
print(a)
a = np.array([1.+0.j, 1.-1.j, 2.+2.j], dtype=complex)
print(a)


plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
