import json
import pygambit

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
def cv_irrigation(total_fields, water_field, total_water, yield_field, cost_per_field,
                  consumption_cost, stress_threshold, stressed_yield,
                  authority_budget, n_farmers):

    WATER_PER_FIELD_YEARLY = water_field * 12

    affordable_total = min(total_fields, max(0, authority_budget) // cost_per_field) # budget cap
    water_possible_total = total_water // WATER_PER_FIELD_YEARLY # water cap

    actual_total_fields = int(min(affordable_total, water_possible_total)) # actual fields

    # stress threshold
    is_stressed = actual_total_fields > stress_threshold
    yield_factor = stressed_yield if is_stressed else yield_field

    # calculation
    total_yield = actual_total_fields * yield_factor
    total_irrigation = actual_total_fields * cost_per_field
    total_consumption = consumption_cost * n_farmers

    payoff = total_yield - total_irrigation - total_consumption

    return payoff

def dv_irrigation(uf_fields, df_fields, water_field, total_water, yield_field, cost_per_field,
                  consumption_cost, stress_threshold, stressed_yield,
                  uf_budget, df_budget, uf_fish_income, df_fish_income):
    
    WATER_PER_FIELD_YEARLY = water_field * 12

    # limits
    uf_affordable = min(uf_fields, uf_budget // cost_per_field)
    df_affordable = min(df_fields, df_budget // cost_per_field)

    uf_water = min(uf_affordable * WATER_PER_FIELD_YEARLY, total_water)
    df_water = max(total_water - uf_water, 0)

    uf_actual_fields = min(uf_affordable, uf_water // WATER_PER_FIELD_YEARLY)
    df_actual_fields = min(df_affordable, df_water // WATER_PER_FIELD_YEARLY)

    # Stress threshold applied to actual irrigated fields
    is_stressed = (uf_actual_fields + df_actual_fields) > stress_threshold
    yield_factor = stressed_yield if is_stressed else yield_field

    # calculations
    uf_yield = uf_actual_fields * yield_factor + uf_fish_income
    df_yield = df_actual_fields * yield_factor + df_fish_income

    uf_cost = uf_actual_fields * cost_per_field + consumption_cost 
    df_cost = df_actual_fields * cost_per_field + consumption_cost 

    uf_payoff = uf_yield - uf_cost
    df_payoff = df_yield - df_cost

    return (uf_payoff, df_payoff)

import math
from typing import Tuple


def dv_irrigation_smooth(uf_fields, df_fields, water_field, total_water, yield_field, cost_per_field,
                  consumption_cost, stress_threshold, stressed_yield,
                  uf_budget, df_budget, uf_fish_income, df_fish_income,
                  tau_minmax=0.1, tau_sig=0.1):
    """
    Smooth continuous approximation of dv_irrigation.
    uf_fields, df_fields: strategies (requested irrigated fields, can be continuous).
    tau_minmax: smoothing parameter for min/max/clip.
    tau_sig: smoothing parameter for stress threshold.
    """

    # ---- smooth helpers ----
    def softplus(x, t):
        z = x / t
        if z > 20:  # stable branch
            return x
        if z < -20:
            return 0.0
        return t * math.log1p(math.exp(z))

    def relu_smooth(x, t):
        return softplus(x, t)  # ≈ max(x,0)

    def softmax2(a, b, t):
        m = max(a, b)
        return t * math.log(math.exp((a - m)/t) + math.exp((b - m)/t)) + m

    def softmin2(a, b, t):
        return -softmax2(-a, -b, t)

    def clip_smooth(x, lo, hi, t):
        return softmin2(hi, softmax2(lo, x, t), t)

    def sigmoid(x, t):
        z = x / t
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    # ---- constants ----
    WATER_PER_FIELD_YEARLY = water_field * 12.0
    N = total_water / max(WATER_PER_FIELD_YEARLY, 1e-12)

    # strategies are requested fields (continuous, but clipped to physical max)
    su = clip_smooth(uf_fields, 0.0, uf_fields, tau_minmax)
    sd = clip_smooth(df_fields, 0.0, df_fields, tau_minmax)

    # ---- budget-limited affordability ----
    Bu_cap = uf_budget / max(cost_per_field, 1e-12)
    Bd_cap = df_budget / max(cost_per_field, 1e-12)

    au = softmin2(su, Bu_cap, tau_minmax)
    ad = softmin2(sd, Bd_cap, tau_minmax)

    # ---- upstream allocation priority ----
    fu = softmin2(au, N, tau_minmax)
    rem = relu_smooth(N - fu, tau_minmax)
    fd = softmin2(ad, rem, tau_minmax)

    # ---- stress sigmoid ----
    total_fields = fu + fd
    I = sigmoid(total_fields - stress_threshold, tau_sig)
    y = (1.0 - I) * yield_field + I * stressed_yield

    # ---- payoffs ----
    uf_yield = fu * y + uf_fish_income
    df_yield = fd * y + df_fish_income

    uf_cost = fu * cost_per_field + consumption_cost
    df_cost = fd * cost_per_field + consumption_cost

    uf_payoff = uf_yield - uf_cost
    df_payoff = df_yield - df_cost

    return (uf_payoff, df_payoff)

  
def generate_cv_matrix(n, m, water_field, total_water, yield_field, cost_per_field,
                       consumption_cost, stress_threshold, stressed_yield,
                       authority_budget, n_farmers):

    matrix = []
    for per_farmer in range(m, m + n):  # 1..10 fields per farmer
        total_fields = per_farmer * n_farmers  # system-wide total
        payoff = cv_irrigation(
            total_fields, water_field, total_water, yield_field, cost_per_field,
            consumption_cost, stress_threshold, stressed_yield,
            authority_budget, n_farmers
        )
        matrix.append([(payoff, 0.0)])
    return matrix

def generate_dv_matrix(n, m, water_field, total_water, yield_field, cost_per_field,
                    consumption_cost, stress_threshold, stressed_yield,
                    uf_budget, df_budget, uf_fish_income, df_fish_income):
    matrix = []
    for uf in range(m, m + n): # UF strategies: m, m+1...
        row = []
        for df in range(m, m + n):
            row.append(dv_irrigation(
                uf, df,
                water_field, total_water, yield_field, cost_per_field,
                consumption_cost, stress_threshold, stressed_yield,
                uf_budget, df_budget, uf_fish_income, df_fish_income
            ))
        matrix.append(row)
    return matrix

def print_matrix(matrix, label_a="UF", label_b="DF"):
    header = f"{label_a}\\{label_b}" + "".join(f"{j+1:>8}" for j in range(len(matrix[0])))
    print(header)
    for i, row in enumerate(matrix):
        line = f"{i+1:>5} " + "".join(f"{p[0]:>4}/{p[1]:<3}" for p in row)
        print(line)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

def load_game(file_path, index=0):
    with open(file_path, 'r') as f: # change index as needed
        all_data = json.load(f)
    
    data = all_data[index]  # get game at specified index

    player_labels = data["players"]
    strategies_a = data["strategies"][player_labels[0]]
    strategies_b = data["strategies"][player_labels[1]]

    # payoff matrix: rows = A's strategies, cols = B's
    n_rows = len(strategies_a)
    n_cols = len(strategies_b)
    payoff_matrix = [[(0, 0) for _ in range(n_cols)] for _ in range(n_rows)]

    for key, value in data["payoffs"].items():
        s1, s2 = key.strip("()").split(",")
        i = strategies_a.index(s1.strip())
        j = strategies_b.index(s2.strip())
        payoff_matrix[i][j] = tuple(value)

    return payoff_matrix, player_labels, strategies_a

def solve_game(payoffs, player_labels=("Player A", "Player B"), strategy_labels=None):
    n_rows = len(payoffs)
    n_cols = len(payoffs[0]) if n_rows > 0 else 0

    game = pygambit.Game.new_table([n_rows, n_cols])
    game.players[0].label = player_labels[0]
    game.players[1].label = player_labels[1]

    # generate strategy labels if not given
    strategy_labels = strategy_labels or [f"Choice {i+1}" for i in range(max(n_rows, n_cols))]

    for i in range(n_rows):
        game.players[0].strategies[i].label = strategy_labels[i]
    for j in range(n_cols):
        game.players[1].strategies[j].label = strategy_labels[j]

    # assign payoffs
    p1, p2 = game.players
    for i in range(n_rows):
        for j in range(n_cols):
            game[i, j][p1] = payoffs[i][j][0]
            game[i, j][p2] = payoffs[i][j][1]

    # solve game
    result = pygambit.nash.enummixed_solve(game, rational=False)
    equilibria = []

    for profile in result.equilibria:
        eq = {
            player_labels[0]: [float(profile[p1][s]) for s in p1.strategies],
            player_labels[1]: [float(profile[p2][s]) for s in p2.strategies]
        }
        equilibria.append(eq)

    return equilibria