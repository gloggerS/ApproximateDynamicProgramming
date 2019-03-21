which = lambda lst:list(np.where(lst)[0])
B = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
a = np.arange(3)
a[B[:, 1]]