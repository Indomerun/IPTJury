import random
import numpy as np
from numpy.random import rand as rand
from copy import deepcopy as copy


#################################
#        Data structures        #
#################################
#
# teams = [Name;    TeamID = 0
#          Name;]   TeamID = 1
#
#
#                   Room0      Room1
# teamSchedule = [[TeamIDs], [TeamIDs], ...;  PF0
#                 [TeamIDs], [TeamIDs], ...;] PF1
#
#
# jurors = [Name, Affiliation(s);   JurorID = 0
#           Name, Affiliation(s);]  JurorID = 1
#
#
# jurorList = [jurorIDs;    PF0
#              jurorIDs;]   PF1
#
#
#                     TeamID=0  TeamID=1  TeamID=2
# judgementMatrix = [   int,      int,      int,      ...;  JurorID = 0
#                       int,      int,      int,      ...;] JurorID = 1
#                       int,      int,      int,      ...;] JurorID = 2
#


def import_teams(filename):
    teams = [line.rstrip('\n') for line in open(filename, 'r')]
    return teams, len(teams)


# Needs to be fixed later, for now assumes 18 teams.
# Should preferably import the team schedule externally
def import_teamSchedule():
    nPFs = 4
    PF = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9+0, 9+1, 9+2, 9+3, 9+4, 9+5, 9+6, 9+7, 9+8],
          [2, 6, 3, 4, 0, 7, 5, 1, 8, 9+2, 9+6, 9+3, 9+4, 9+0, 9+7, 9+5, 9+1, 9+8],
          [8, 2, 4, 1, 3, 7, 5, 0, 6, 9+8, 9+2, 9+4, 9+1, 9+3, 9+7, 9+5, 9+0, 9+6],
          [4, 1, 6, 2, 5, 7, 3, 0, 8, 9+4, 9+1, 9+6, 9+2, 9+5, 9+7, 9+3, 9+0, 9+8]]
    teamSchedule = [None] * nPFs
    for iPF in range(nPFs):
        teamSchedule[iPF] = [PF[iPF][0:3], PF[iPF][3:6], PF[iPF][6:9], PF[iPF][9:12], PF[iPF][12:15], PF[iPF][15:18]]
    return teamSchedule, nPFs


# Needs to be fixed later, for now assumes a single affiliation.
def import_jurors(filename, teams):
    jurors = []
    lines = [line.rstrip('\n') for line in open(filename, 'r')]
    for index, line in enumerate(lines):
        fields = line.split(';')
        name = fields[0] + ', ' + fields[1]
        affiliations = []
        for iAff in range(2, len(fields)):
            try:
                affiliation = teams.index(fields[iAff])
            except ValueError:
                affiliation = None
            affiliations += [affiliation]
        juror = [name, affiliations]
        jurors.append(juror)
    return jurors, len(jurors)


def print_imported(teams, teamSchedule, jurors):
    # Print team IDs and team names
    print('\nTeams:')
    for teamID, team in enumerate(teams):
        print('\t{:>2}: {}'.format(teamID, team))

    # Print the PF schedule
    print("\nPF Schedule:")
    for iPF, rooms in enumerate(teamSchedule):
        print("  PF {}:".format(iPF))
        for roomID, room in enumerate(rooms):
            ID_str = ', '.join(['{:>2}'.format(teamID) for teamID in room])
            str = ', '.join([teams[teamID] for teamID in room])
            print('\tRoom{}: '.format(roomID) + ID_str + '\t(' + str + ')')

    # Print the juror IDs, names and affiliations
    print("\nJurors, affiliations:")
    for jurorID, juror in enumerate(jurors):
        ID_str = ', '.join(['{:>2}'.format(affID) if affID is not None else '' for affID in juror[1]])
        str = ', '.join([teams[affID] if affID is not None else 'None' for affID in juror[1]])
        print('\t{:>2}: {}'.format(jurorID, juror[0]) + ', ' + ID_str + ' (' + str + ')')
    print()


# Needs to be fixed later, for now assumes ALL jurors for ALL PFs.
# Should be obtained from information provided in 'jurors'.
def generate_jurorList(jurors, nPFs):
    jurorList = [None] * nPFs
    for iPFs in range(nPFs):
        jurorList[iPFs] = list(range(len(jurors)))
    return jurorList


def get_jurorSchedule(individual, jurors, nJurors, nRooms):
    nPFs = len(individual)
    nRooms = nTeams // 3
    jurorSchedule = [[juror[0], [None] * nPFs] for juror in jurors]
    for iPF in range(nPFs):
        for iJuror, jurorID in enumerate(individual[iPF]):
            jurorSchedule[jurorID][1][iPF] = iJuror % nRooms
    return jurorSchedule


def save_jurorSchedule(filename, jurorSchedule):
    with open(filename, "w") as text_file:
        for juror in jurorSchedule:
            output = juror[0] + ';' + ';'.join([str(room) for room in juror[1]])
            print(output, file=text_file)


def calculateMatrix(individual, teamSchedule, jurors, nJurors, nTeams):
    nPFs = len(individual)
    nRooms = nTeams // 3
    judgementMatrix = np.zeros([nJurors, nTeams], dtype=int)
    for iPF in range(nPFs):
        for iJuror, jurorID in enumerate(individual[iPF]):
            room = iJuror % nRooms
            for iTeam in teamSchedule[iPF][room]:
                if iTeam in jurors[jurorID][1]:
                    judgementMatrix[jurorID][iTeam] += 10
                else:
                    judgementMatrix[jurorID][iTeam] += 1
    return judgementMatrix


def save_matrix(filename, matrix):
    with open(filename, "w") as text_file:
        for line in matrix:
            output = ' '.join([str(item) for item in line])
            print(output, file=text_file)


def print_matrix(matrix):
    sep1 = '  '
    sep2 = ' | '
    mat = matrix.copy()
    ncols = mat.shape[1]

    print()
    print((2+len(sep2))*' ' + ' '.join(['{:>2}'.format(item) for item in range(ncols)]))
    print((2+len(sep2))*' ' + (3*(mat.shape[1]-1)+1 + 2) * '-')
    for i, line in enumerate(mat):
        rowsum = sum([max(0, item-1) for item in line])
        print('{:>2}: | '.format(i)+sep1.join([str(item) for item in line]) + ' | ' + str(rowsum))
    print((2+len(sep2))*' ' + (3*(ncols-1)+1 + 2) * '-')
    mat[mat > 0] -= 1
    colsum = mat.sum(axis=0)
    print((3+len(sep2))*' ' + sep1.join([str(item) for item in colsum]))


###################################
#        Genetic Algorithm        #
###################################

def initializeIndividual(jurorList):
    nPFs = len(jurorList)
    individual = [None] * nPFs
    for iPFs in range(nPFs):
        individual[iPFs] = random.sample(jurorList[iPFs], len(jurorList[iPFs]))  # Random permutation of the juror list
    return individual


def initializePopulation(N, jurorList):
    population = [None] * N
    for i in range(N):
        population[i] = initializeIndividual(jurorList)
    return population


def evaluateIndividual(individual):
    judgementMatrix = calculateMatrix(individual, teamSchedule, jurors, nJurors, nTeams)
    judgementMatrix[judgementMatrix > 0] -= 1

    totalsum = (judgementMatrix**2).sum()
    teamSum = (judgementMatrix.sum(axis=0)**2).sum()

    return 1/(totalsum + teamSum)


def evaluatePopulation(population):
    nIndividuals = len(population)
    fitness = [None] * nIndividuals
    maximumFitness = (None, 0)
    for i, individual in enumerate(population):
        fitness[i] = evaluateIndividual(individual)
        if fitness[i] > maximumFitness[1]:
            maximumFitness = (i, fitness[i])
    return maximumFitness[0], fitness


def tournamentSelection(fitness, tournamentParameter):
    populationSize = len(fitness)

    iContestants = [None] * 2
    contestantFitness = [None] * 2
    for i in range(2):
        r = rand()
        iContestants[i] = int(r * populationSize)
        contestantFitness[i] = fitness[iContestants[i]]

    pairs = sorted(zip(contestantFitness, iContestants), key=lambda pair: pair[0], reverse=True)
    contestantFitness, iContestants = [p[0] for p in pairs], [p[1] for p in pairs]

    r = rand()
    if r < tournamentParameter:
        iSelected = iContestants[0]
    else:
        iSelected = iContestants[1]

    return iSelected


# Chooses cross1 with 20% probability and cross2 with 80% probability
def crossover(parent1, parent2, probability):
    """This method should cross 'parent1' with 'parent2' and return the resulting children."""
    r = rand()
    if r < 0.2:
        child1, child2 = cross1(parent1, parent2, probability)
    else:
        child1, child2 = cross2(parent1, parent2, probability)
    return child1, child2


# Crosses complete PF juror schedules (e.g. switches the PF2-PF3 schedules of the parents)
def cross1(parent1, parent2, probability):
    child1, child2 = copy(parent1), copy(parent2)

    crossoverPoint = int(rand() * nPFs)
    child1[:crossoverPoint], child2[crossoverPoint:] = child2[:crossoverPoint], child1[crossoverPoint:]
    return child1, child2


# Crosses single PF schedules (e.g. crosses PF2 of the parents)
def cross2(parent1, parent2, probability):
    child1, child2 = copy(parent1), copy(parent2)

    for iPFs in range(nPFs):
        r = rand()
        if (r < probability):
            child1[iPFs], child2[iPFs] = cyclicCrossover(child1[iPFs], child2[iPFs])
    return child1, child2


# Crosses two arrays, such that both remain permutations of the ordered array [0, 1, ..., n-1]
def cyclicCrossover(parent1, parent2):
    nJurors = len(parent1)
    child1 = [-1] * nJurors
    child2 = [-1] * nJurors
    pt = parent1.index(0)
    while child1[pt] == -1:
        child1[pt] = parent1[pt]
        pt = parent1.index(parent2[pt])

    pt = parent2.index(0)
    while child2[pt] == -1:
        child2[pt] = parent2[pt]
        pt = parent2.index(parent1[pt])

    for i in range(nJurors):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]

    return child1, child2


# Mutate individual by swapping the place of two jurors within the same PF schedule
def mutate(individual, probability):
    """This method should perform a mutation of 'individual' and return the result."""
    mutatedIndividual = copy(individual)

    for iPF in range(nPFs):
        r = rand()
        if r < probability[0]:
            for iJuror1 in range(nJurors):
                r = rand()
                if r < probability[1]:
                    iJuror2 = int(rand() * nJurors)
                    mutatedIndividual[iPF][iJuror1], mutatedIndividual[iPF][iJuror2] = mutatedIndividual[iPF][iJuror2], mutatedIndividual[iPF][iJuror1]
    return mutatedIndividual


#############################################
#     Import team and juror information     #
#############################################
teams, nTeams = import_teams('teams.txt')
teamSchedule, nPFs = import_teamSchedule()
jurors, nJurors = import_jurors('jurors.txt', teams)
jurorList = generate_jurorList(jurors, nPFs)
n = [nTeams, nPFs, nJurors]

print_imported(teams, teamSchedule, jurors)


#################################
#     Run Genetic Algorithm     #
#################################
populationSize = 30
nGenerations = 5000
tournamentParameter = 0.75
crossoverProbability = 0.8
mutationProbability = (1/nPFs, 1/nJurors)

population = initializePopulation(populationSize, jurorList)
for iGeneration in range(nGenerations):
    # Evaluation
    iBestIndividual, fitness = evaluatePopulation(population)
    if iGeneration % 100 == 0:
        print(iGeneration, 1 / fitness[iBestIndividual])

    tmpPopulation = [None] * populationSize
    # Selection
    for i in range(populationSize//2):
        i1 = tournamentSelection(fitness, tournamentParameter)
        i2 = tournamentSelection(fitness, tournamentParameter)

        # Crossover
        child1, child2 = crossover(population[i1], population[i2], crossoverProbability)
        tmpPopulation[2*i] = child1
        tmpPopulation[2*i + 1] = child2

    # Insert new, random Individual(s) at the end of the temporary population list
    for i in range(2+populationSize % 2):
        tmpPopulation[-1 - i] = initializeIndividual(jurorList)

    # Mutate all individuals in the temporary population list
    for i in range(populationSize):
        individual = tmpPopulation[i]
        tmpPopulation[i] = mutate(individual, mutationProbability)

    # Insert Best Individual at the beginning of the temporary population list
    tmpPopulation[0] = copy(population[iBestIndividual])

    # Update population
    population = tmpPopulation


###########################
#     Save the result     #
###########################
judgementMatrix = calculateMatrix(population[0], teamSchedule, jurors, nJurors, nTeams)
jurorSchedule = get_jurorSchedule(population[0], jurors, nJurors, nTeams)
save_matrix("judgementMatrix.txt", judgementMatrix)
save_jurorSchedule("jurorSchedule.txt", jurorSchedule)


##########################
# Print judgement matrix #
##########################
print_matrix(judgementMatrix)
print("Final fitness: {}".format(1/fitness[iBestIndividual]))
