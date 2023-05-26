import random
limits = 20
K = 50
if __name__ == "__main__":
    open('coordinates.txt', 'w').close()
    f = open("coordinates.txt", "a")
    for i in range(K):
        x = round(random.uniform(-limits, limits), 5)
        y = round(random.uniform(-limits, limits), 5)
        f.write(str(x) + " " + str(y)+"\n")




