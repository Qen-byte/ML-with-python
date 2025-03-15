import matplotlib.pyplot as plt
artistes=["Kendrick Lamar","Drake","J.Cole","Tems","Burna Boy"]
grammies_won=[5,6,3,2,1]

plt.bar(range(len(artistes)),grammies_won)

plt.title("Grammy Kings innit")
plt.ylabel("# of awards")

plt.xticks(range(len(artistes)),grammies_won)

plt.show()