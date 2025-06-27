import json
import pygambit

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
def dv_irrigation(uf_fields, df_fields, total_water=2, yield_per_field=4, cost_per_field=1, consumption_cost=1,stress_threshold=3,stressed_yield=2):

    total_fields = uf_fields + df_fields

    # threshold stress
    if total_fields > stress_threshold:
        actual_yield = stressed_yield
    else:
        actual_yield = yield_per_field

    # UF withdraws first
    uf_water = min(uf_fields, total_water)
    df_water = max(total_water - uf_fields, 0)
    df_actual_fields = min(df_fields, df_water)

    uf_yield = uf_water * actual_yield
    df_yield = df_actual_fields * actual_yield

    uf_cost = uf_fields * cost_per_field + consumption_cost
    df_cost = df_fields * cost_per_field + consumption_cost

    uf_payoff = uf_yield - uf_cost # doesn't input current budget and fish income
    df_payoff = df_yield - df_cost

    #print (uf_payoff, df_payoff)
    return (uf_payoff, df_payoff)

# generate full matrix
def generate_matrix(n=2, m=1): # input matrix size and number of fields
    matrix = []
    for uf in range(m, m + n): # UF strategies: m, m+1...
        row = []
        for df in range(m, m + n):
            row.append(dv_irrigation(uf, df))
        matrix.append(row)
        print(row)
    return matrix
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

# run with matrices.json
#payoffs, players, strategy_labels = load_game("matrices.json")
#equilibria = solve_game(payoffs, player_labels=players, strategy_labels=strategy_labels)

# run with CPR
payoffs = generate_matrix(n=2, m=1)
equilibria = solve_game(payoffs, player_labels=("Farmer A", "Farmer B"))

if not equilibria:
    print("No equilibria found.")
else:
    for i, eq in enumerate(equilibria, 1):
        print(f"\nEquilibrium {i}:")
        for player, probs in eq.items():
            print(f"{player}:")
            for strat, prob in zip([f"{i+1}" for i in range(len(probs))], probs):
                print(f"  Strategy {strat}: {prob:.2f}")
