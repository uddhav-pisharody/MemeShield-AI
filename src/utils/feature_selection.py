import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from evolutionary_algorithms import GeneticAlgorithm, FireflyAlgorithm, ParticleSwarmOptimization  

def apply_genetic_algorithm(features: pd.DataFrame, config):
    ga = GeneticAlgorithm(config.ga_params)
    selected_features = ga.fit_transform(features)
    return selected_features

def apply_firefly_algorithm(features: pd.DataFrame, config):
    fa = FireflyAlgorithm(config.fa_params)
    selected_features = fa.fit_transform(features)
    return selected_features

def apply_pso(features: pd.DataFrame, config):
    pso = ParticleSwarmOptimization(config.pso_params)
    selected_features = pso.fit_transform(features)
    return selected_features

def apply_feature_selection(features_csv: str, config):
    features = pd.read_csv(features_csv)
    selected_features_ga = apply_genetic_algorithm(features, config)
    selected_features_fa = apply_firefly_algorithm(features, config)
    selected_features_pso = apply_pso(features, config)

    # Return or save selected features
    return selected_features_ga, selected_features_fa, selected_features_pso