import random
import math
import time
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def decoding(individual) :
    x = 0
    y = 0
    for i, k in zip(range(8), individual[:8]) :
        x += 2 ** i * k
    x = (x / 2 ** 8 * 60) - 30

    for i, k in zip(range(8), individual[8:]) :
        y += 2 ** i * k
    y = (y / 2 ** 8 * 60) - 30
    return x, y

def griewank(x,y) :
    return -((x*x + y*y)/4000 - math.cos(x)* math.cos(y/math.sqrt(2)) + 1) + 30

def myEval(individual):
    x, y = decoding(individual)
    return griewank(x,y),

def FitShareEval(individual, population) :
    inddecodingresult = decoding(individual)
    mapresult = list(map(decoding, population))
    result = []
    for i in mapresult :
        result.append(distance(inddecodingresult,i))
    result = list(map(usadofunc,result))
    return griewank(inddecodingresult[0],inddecodingresult[1])/sum(result),

toolbox.register("evaluate", myEval)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select",  tools.selTournament, tournsize=3)
seedval = int(time.time()*10000%10000)
random.seed(seedval)
print(seedval)

popresult = []
soonseo = []

mylist = []
for x in range(1) :
    mylist.append(x/10)

print(mylist)

for x in mylist :
    print(x)
    pop = toolbox.population(n=100)


    fitnesses = list(map(toolbox.evaluate, pop))

    bested = -1
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
    while max(fits) < 29.99 and  g < 100 :
        g = g + 1
        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        if g == 99 :
            soonseo.append(99)
            popresult.append(offspring)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5 :
                toolbox.mate(child1, child2)

        for mutant in offspring:
            if random.random() < x :
                toolbox.mutate(mutant)

        fitnesses = list(map(toolbox.evaluate, offspring))

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = tools.selBest(pop, 1)[0]
        prebest = best_ind.fitness.values
        if (prebest[0] > bested):
            bested = prebest[0]
            soonseo.append(g)
            popresult.append(list(map(toolbox.clone, toolbox.select(pop, len(pop)))))


i = 0
lastresult = []
for k in popresult :
    best_ind = tools.selBest(k, 1)[0]
    best_fit = -(best_ind.fitness.values[0]-30)#-1을 곱하고 30을 빼면 0이하가 되겠지 
    print("gene %s ind %s fit %s"% (soonseo[i], best_ind, best_fit), end = "")
    print("")
    if soonseo[i] == 99 :
        for o in k :
            print(decoding(o), -((myEval(o)[0])-30))
    i += 1


best_ind_axis_list = []
for k in popresult :
    best_ind_axis_list.append(list(map(decoding,k)))

i = 0
for k in best_ind_axis_list :
    tmpx = []
    tmpy = []
    for o in k :
        tmpx.append(o[0])
        tmpy.append(o[1])
    plt.plot(tmpx, tmpy, 'ro')
    plt.axis([-30, 30, -30, 30])
    plt.title("generation" + str(soonseo[i]))
    plt.show()
    i += 1