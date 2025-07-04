import json
import pygambit

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
def dv_irrigation(uf_fields, df_fields, water_field, total_water, yield_field, cost_per_field,
                  consumption_cost, stress_threshold, stressed_yield,
                  uf_budget, df_budget, uf_fish_income, df_fish_income):
    
    WATER_PER_FIELD_YEARLY = water_field * 12

    # UF withdraws first
    uf_water = min(uf_fields * WATER_PER_FIELD_YEARLY, total_water)
    df_water = max(total_water - uf_water, 0)

    df_actual_fields = min(df_fields, df_water // WATER_PER_FIELD_YEARLY)

    is_stressed = (uf_fields + df_fields) > stress_threshold
    yield_factor = stressed_yield if is_stressed else yield_field

    uf_yield = uf_fields * yield_factor + df_budget + df_fish_income
    df_yield = df_actual_fields * yield_factor + uf_budget + uf_fish_income

    uf_cost = uf_fields * cost_per_field + consumption_cost 
    df_cost = df_fields * cost_per_field + consumption_cost 

    uf_payoff = uf_yield - uf_cost
    df_payoff = df_yield - df_cost

    return (uf_payoff, df_payoff)

# generate full matrix
def generate_matrix(n, m, water_field, total_water, yield_field, cost_per_field,
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