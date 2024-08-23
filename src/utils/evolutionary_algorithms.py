import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

#GENETIC ALGORITHM

def fitness(chromosome, X_train, X_test, y_train, y_test, classifier):
    selected_features = [feature for feature, select in zip(X_train.columns, chromosome) if select == 1]
    if len(selected_features) == 0:
        return 0, []  # Penalize solutions with no features selected

    clf = classifier()  # Create classifier instance
    clf.fit(X_train[selected_features], y_train)
    y_pred = clf.predict(X_test[selected_features])
    score = accuracy_score(y_test, y_pred)
    return score, selected_features

def GeneticAlgorithm(X_train, y_train, X_val, y_val, population_size=100, num_generations=10, mutation_rate=0.1, save_path='Enter path here'):
    classifiers = [RandomForestClassifier, SVC, XGBClassifier]
    all_fitness_scores = []
    all_feature_lengths = []

    for classifier in classifiers:
        print(f"Classifier: {classifier.__name__}")
        fitness_scores = []
        population = np.random.randint(2, size=(population_size, X_train.shape[1]))
        
        for generation in range(num_generations):
            # Evaluate fitness of each chromosome
            scores_and_features = [fitness(chromosome, X_train, X_val, y_train, y_val, classifier) for chromosome in population]
            scores, features = zip(*scores_and_features)
            fitness_scores.append(max(scores))

            # Select parents based on fitness scores
            selected_indices = np.random.choice(range(population_size), size=population_size, replace=True, p=scores / np.sum(scores))
            parents = population[selected_indices]

            # Crossover
            crossover_point = np.random.randint(1, X_train.shape[1])
            offspring = np.zeros_like(parents)
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                offspring[i, :crossover_point] = parent1[:crossover_point]
                offspring[i, crossover_point:] = parent2[crossover_point:]
                offspring[i+1, :crossover_point] = parent2[:crossover_point]
                offspring[i+1, crossover_point:] = parent1[crossover_point:]

            # Mutation
            mutation_mask = np.random.rand(population_size, X_train.shape[1]) < mutation_rate
            offspring ^= mutation_mask

            # Replace old population with offspring
            population = offspring

            # Output best solution in current generation
            best_solution_idx = np.argmax(scores)
            best_fitness = scores[best_solution_idx]
            print(f"Generation {generation+1}: Best Fitness = {best_fitness}")

        # Output the selected features
        best_solution_idx = np.argmax(scores)
        selected_features = features[best_solution_idx]
        print("Selected Features:", selected_features)
        print("Number of Selected Features:", len(selected_features))
        print("\n")

        # Save the selected features
        with open(f"{save_path}/selected_features_GA_{classifier.__name__}.pkl", "wb") as f:
            pickle.dump(selected_features, f)

        # Store fitness scores and feature lengths for all classifiers
        all_fitness_scores.append(fitness_scores)
        all_feature_lengths.append(len(selected_features))

    # Plot fitness scores over generations for all classifiers
    plt.figure(figsize=(10, 6))
    for i, classifier in enumerate(classifiers):
        plt.plot(range(1, num_generations + 1), all_fitness_scores[i], label=f"{classifier.__name__} - Selected Features: {all_feature_lengths[i]}")
    plt.title("Fitness Scores over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GeneticAlgo.jpg")
    plt.show()


#PARTICLE SWARM OPTIMISATION
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = float('-inf')

def evaluate(chromosome, X_train, X_test, y_train, y_test, classifier):
    selected_features = [feature for feature, select in zip(X_train.columns, chromosome) if select == 1]
    if len(selected_features) == 0:
        return float('-inf'), 0  # Penalize solutions with no features selected

    clf = classifier()  # Create classifier instance
    clf.fit(X_train[selected_features], y_train)
    y_pred = clf.predict(X_test[selected_features])
    score = accuracy_score(y_test, y_pred)
    return score, len(selected_features)

def initialize_particles(population_size, num_features):
    particles = []
    for _ in range(population_size):
        position = np.random.randint(2, size=num_features)
        velocity = np.random.rand(num_features)
        particles.append(Particle(position, velocity))
    return particles

def update_velocity(particle, global_best_position, inertia_weight, cognitive_coefficient, social_coefficient):
    cognitive_component = cognitive_coefficient * np.random.rand() * (particle.best_position - particle.position)
    social_component = social_coefficient * np.random.rand() * (global_best_position - particle.position)
    new_velocity = inertia_weight * particle.velocity + cognitive_component + social_component
    return new_velocity

def update_position(particle):
    new_position = np.round(1 / (1 + np.exp(-particle.velocity)))
    return new_position

def ParticleSwarmOptimisation(X_train, y_train, X_val, y_val, population_size=100, num_generations=10, inertia_weight=0.5, cognitive_coefficient=1.5, social_coefficient=1.5, save_path="Enter path here"):
    classifiers = [RandomForestClassifier, SVC, XGBClassifier]
    all_fitness_scores = []
    all_feature_lengths = []

    for classifier in classifiers:
        print(f"Classifier: {classifier.__name__}")
        num_features = X_train.shape[1]
        particles = initialize_particles(population_size, num_features)
        global_best_score = float('-inf')
        global_best_position = None
        fitness_scores = []
        feature_lengths = []

        for generation in range(num_generations):
            for particle in particles:
                score, feature_length = evaluate(particle.position, X_train, X_val, y_train, y_val, classifier)
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()

            for particle in particles:
                new_velocity = update_velocity(particle, global_best_position, inertia_weight, cognitive_coefficient, social_coefficient)
                particle.velocity = new_velocity
                new_position = update_position(particle)
                particle.position = new_position

            fitness_scores.append(global_best_score)
            feature_lengths.append(feature_length)
            print(f"Generation {generation+1}: Best Fitness = {global_best_score}")

        selected_features = [feature for feature, select in zip(X_train.columns, global_best_position) if select == 1]
        print("Selected Features:", selected_features)
        print("Number of Selected Features:", len(selected_features))

        # Save the selected features
        with open(f"{save_path}/selected_features_PSO_{classifier.__name__}.pkl", "wb") as f:
            pickle.dump(selected_features, f)

        all_fitness_scores.append(fitness_scores)
        all_feature_lengths.append(len(selected_features))

    # Plot fitness scores over generations for all classifiers
    plt.figure(figsize=(10, 6))
    for i, classifier in enumerate(classifiers):
        plt.plot(range(1, num_generations + 1), all_fitness_scores[i], label=f"{classifier.__name__} - Selected Features: {all_feature_lengths[i]}")
    plt.title("Fitness Scores over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/PSOAlgo.jpg")
    plt.show()

# FIREFLY ALGORITHM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class Firefly:
    def __init__(self, position, attractiveness):
        self.position = position
        self.attractiveness = attractiveness

def evaluate(chromosome, X_train, X_test, y_train, y_test, classifier):
    selected_features = [feature for feature, select in zip(X_train.columns, chromosome) if select == 1]
    if len(selected_features) == 0:
        return float('-inf'), 0  # Penalize solutions with no features selected

    clf = classifier()  # Create classifier instance
    clf.fit(X_train[selected_features], y_train)
    y_pred = clf.predict(X_test[selected_features])
    score = accuracy_score(y_test, y_pred)
    return score, len(selected_features)

def initialize_fireflies(population_size, num_features):
    fireflies = []
    for _ in range(population_size):
        position = np.random.randint(2, size=num_features)
        attractiveness = 0  # Attractiveness can be initialized as 0
        fireflies.append(Firefly(position, attractiveness))
    return fireflies

def attractiveness(distance):
    # Define attractiveness function (e.g., based on distance)
    return 1 / (1 + distance)

def move_fireflies(current_firefly, other_firefly, attractiveness, beta, gamma):
    distance = np.linalg.norm(current_firefly.position - other_firefly.position)
    beta_component = beta * np.exp(-gamma * distance**2)
    new_position = current_firefly.position + attractiveness * (other_firefly.position - current_firefly.position) + beta_component
    new_position = np.round(1 / (1 + np.exp(-new_position)))  # Ensure binary positions
    return new_position

def FireflyAlgorithm(X_train, X_test, y_train, y_test, classifier, population_size=100, num_generations=10, beta=1.0, gamma=1.0):
    num_features = X_train.shape[1]
    fireflies = initialize_fireflies(population_size, num_features)
    fitness_scores = []
    feature_lengths = []

    for generation in range(num_generations):
        for current_firefly in fireflies:
            current_score, feature_length = evaluate(current_firefly.position, X_train, X_test, y_train, y_test, classifier)

            for other_firefly in fireflies:
                if current_score < other_firefly.attractiveness:  # Only move if the other firefly is brighter
                    current_firefly.position = move_fireflies(current_firefly, other_firefly, other_firefly.attractiveness, beta, gamma)

        # Update attractiveness based on fitness scores
        for current_firefly in fireflies:
            current_score, _ = evaluate(current_firefly.position, X_train, X_test, y_train, y_test, classifier)
            current_firefly.attractiveness = current_score

        best_firefly = max(fireflies, key=lambda x: x.attractiveness)
        best_score, best_feature_length = evaluate(best_firefly.position, X_train, X_test, y_train, y_test, classifier)
        fitness_scores.append(best_score)
        feature_lengths.append(best_feature_length)
        print(f"Generation {generation+1}: Best Fitness = {best_score}")

    selected_features = [feature for feature, select in zip(X_train.columns, best_firefly.position) if select == 1]
    print("Selected Features:", selected_features)
    print(len(selected_features))

    return fitness_scores, selected_features