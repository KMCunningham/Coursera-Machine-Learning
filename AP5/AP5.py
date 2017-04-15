from random import Random
from time import time
from math import sin
from math import sqrt
from inspyred import ec
from inspyred.ec import terminators


def generate_Schwefel(random, args):
    size = args.get('num_inputs', 2)
    return [random.uniform(-500.00, 500.00) for i in range(size)]

def evaluate_Schwefel(candidates, args):
    fitness = []
    for cs in candidates:
        fit = (418.9829 * len(cs) - sum([(-x  * sin(sqrt(abs(x)))) for x in cs]))
        fitness.append(fit)
    return fitness


rand = Random()
rand.seed(int(time()))

es = ec.ES(rand)
es.terminator = terminators.evaluation_termination


final_pop = es.evolve(generator=generate_Schwefel,
                             evaluator=evaluate_Schwefel,
                             pop_size=100,
                             maximize=False,
                             bounder=ec.Bounder(-500, 500),
                             num_selected=100,
                             tournament_size=2,
                             mutation_rate=0.25,
                             max_evaluations=20000,
                             num_input=2)



# Sort and print the fittest individual, who will be at index 0.
final_pop.sort(reverse=True)
best = final_pop[0]
components = best.candidate
print('\nFittest individual:')
print(best)

