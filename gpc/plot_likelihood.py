import matplotlib.pyplot as plt
likelihoods = [-3468696.42772856, -3467014.1675352, -3465621.54277245] 
plt.plot(likelihoods)
plt.ylabel('Likelihood')
plt.xlabel('Iteration')
plt.title('PolyA, K-means Initialization')
plt.show()
