import numpy as np
LARVAE_INFLOW_THRESHOLD = 2000 

class FishPopulation:
    def __init__(self):
        self.age_classes = [1000, 500] + [100] * 11 # age 0 to 12; 0 = larvae, 1–4 = juvenile, 5+ = adult
        self.larvae_survival_rate = 0.3
        self.adult_spawn_rate = 1.2
        self.carrying_capacity = 10000  # Max sustainable population
        self.mortality_density_independent = [0.4, 0.3, 0.2, 0.1, 0.05] + [0.1] * 8
        self.reproduction_rates = [0.0] * 5 + [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.migration_rate_age_0 = 0.25

    def grow(self, lake_water, may_inflow): # leslie matrix equation
        new_population = [0] * 13
        adult_population = sum(self.age_classes[5:])
        juvenile_population = sum(self.age_classes[0:5])
        
        alpha = self.adult_spawn_rate # Birth rate
        sigma = 0.00001 # Density dependence strength
        gamma_juv = 0.00000000000001 # Juvenile crowding mortality

        I_t = int(may_inflow - LARVAE_INFLOW_THRESHOLD) if may_inflow > LARVAE_INFLOW_THRESHOLD else 0
        density_term = np.exp(-sigma * adult_population) # density dependent spawning
        larvae_from_adults = sum([ alpha * density_term * self.age_classes[i] for i in range(5, 13)])
        new_population[0] = int(I_t + larvae_from_adults)

        # age progression with mortality
        for age in range(12, 0, -1):  # age 12 down to age 1
            prev = self.age_classes[age - 1]
            mortality = self.mortality_density_independent[age - 1]
            survival = prev * (1 - mortality)

            if 1 <= age - 1 <= 4:
                crowding_penalty = gamma_juv * (juvenile_population ** 2)
                survival -= crowding_penalty
                survival = max(survival, 0)

            # Age 12 fish die
            new_population[age] = int(survival)

        if lake_water <= 100:
            new_population = [0 for _ in new_population]
        elif lake_water < 500:
            new_population = [int(n * 0.8) for n in new_population]
        
        self.age_classes = new_population

    def harvest(self, target_catch):
        adult_indices = range(5, 13)  # age classes 5 to 12
        catch = 0

        for _ in range(target_catch):
            adults_total = sum(self.age_classes[i] for i in adult_indices)
            if adults_total == 0:
                break

            # random adult fish, weighted by availability
            weights = np.array([self.age_classes[i] for i in adult_indices])
            probabilities = weights / weights.sum()

            chosen = np.random.choice(adult_indices, p=probabilities)
            if self.age_classes[chosen] > 0:
                self.age_classes[chosen] -= 1
                catch += 1

        return catch
