import numpy as np
import pandas as pd

FLOAT_DECIMALS = 3


def build_gene_meta(bounds):
    gene_names = list(bounds.keys())
    low_vec = np.array([bounds[k]["low"] for k in gene_names], dtype=float)
    high_vec = np.array([bounds[k]["high"] for k in gene_names], dtype=float)
    is_int_mask = np.array([bounds[k]["type"] == "int" for k in gene_names], dtype=bool)
    return gene_names, low_vec, high_vec, is_int_mask


def quantize_population(pop, low_vec, high_vec, is_int_mask, float_decimals=FLOAT_DECIMALS):
    pop = np.clip(pop, low_vec, high_vec)

    if np.any(~is_int_mask):
        pop[:, ~is_int_mask] = np.round(pop[:, ~is_int_mask], decimals=float_decimals)

    if np.any(is_int_mask):
        pop[:, is_int_mask] = np.round(pop[:, is_int_mask])

    pop = np.clip(pop, low_vec, high_vec)

    if np.any(is_int_mask):
        pop[:, is_int_mask] = pop[:, is_int_mask].astype(int)

    return pop


def init_population_array(pop_size, low_vec, high_vec, is_int_mask, rng, float_decimals=FLOAT_DECIMALS):
    n_genes = len(low_vec)
    pop = rng.uniform(low=low_vec, high=high_vec, size=(pop_size, n_genes))

    if np.any(is_int_mask):
        for j in np.where(is_int_mask)[0]:
            pop[:, j] = rng.integers(
                int(low_vec[j]),
                int(high_vec[j]) + 1,
                size=pop_size
            )

    pop = quantize_population(pop, low_vec, high_vec, is_int_mask, float_decimals=float_decimals)
    return pop


def roulette_select_rankbased_indices(rng, fitness, n_select=2, eps=1e-12):
    Z = np.asarray(fitness, dtype=float)
    n = len(Z)

    order = np.argsort(Z)  # smaller is better
    weights = np.zeros(n, dtype=float)

    for rank, idx in enumerate(order):
        weights[idx] = n - rank

    weights += eps
    probs = weights / weights.sum()

    return rng.choice(n, size=n_select, replace=True, p=probs)


def two_point_crossover_array(parent1, parent2, rng, crossover_rate=0.9):
    child1 = parent1.copy()
    child2 = parent2.copy()

    if rng.random() > crossover_rate:
        return child1, child2

    n_gene = parent1.shape[0]
    if n_gene < 2:
        return child1, child2

    cut1, cut2 = sorted(rng.choice(np.arange(n_gene), size=2, replace=False))

    child1[cut1:cut2 + 1] = parent2[cut1:cut2 + 1]
    child2[cut1:cut2 + 1] = parent1[cut1:cut2 + 1]

    return child1, child2


def mutate_population_array(pop, low_vec, high_vec, is_int_mask, rng, mutation_rate=0.1, float_decimals=FLOAT_DECIMALS):
    out = pop.copy()
    pop_size, n_genes = out.shape

    mutate_mask = rng.random((pop_size, n_genes)) < mutation_rate

    for j in range(n_genes):
        rows = np.where(mutate_mask[:, j])[0]
        if len(rows) == 0:
            continue

        if is_int_mask[j]:
            out[rows, j] = rng.integers(
                int(low_vec[j]),
                int(high_vec[j]) + 1,
                size=len(rows)
            )
        else:
            out[rows, j] = rng.uniform(
                low=low_vec[j],
                high=high_vec[j],
                size=len(rows)
            )

    out = quantize_population(out, low_vec, high_vec, is_int_mask, float_decimals=float_decimals)
    return out


def population_array_to_df(pop, gene_names):
    df = pd.DataFrame(pop, columns=gene_names)

    if "turn" in df.columns:
        df["turn"] = df["turn"].round().astype(int)

    return df


def run_ga_fast(
    bounds,
    ga_cfg,
    evaluate_population_fn,
    rng=None,
    verbose=True,
    float_decimals=FLOAT_DECIMALS,
    **eval_kwargs
):
    if rng is None:
        rng = np.random.default_rng(ga_cfg.get("seed", 42))

    pop_size = int(ga_cfg["pop_size"])
    n_gen = int(ga_cfg["n_gen"])
    elite_size = int(ga_cfg.get("elite_size", 1))
    crossover_rate = float(ga_cfg.get("crossover_rate", 0.9))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.1))

    gene_names, low_vec, high_vec, is_int_mask = build_gene_meta(bounds)
    n_genes = len(gene_names)

    population = init_population_array(
        pop_size=pop_size,
        low_vec=low_vec,
        high_vec=high_vec,
        is_int_mask=is_int_mask,
        rng=rng,
        float_decimals=float_decimals,
    )

    history = []
    best_individual = None
    best_fitness = np.inf
    best_record = None

    for gen in range(n_gen):
        pop_df = population_array_to_df(population, gene_names)
        fitness, records, meta = evaluate_population_fn(pop_df, **eval_kwargs)

        fitness = np.asarray(fitness, dtype=float)

        if len(fitness) != len(population):
            raise ValueError("Length of fitness does not match population size")
        if len(records) != len(population):
            raise ValueError("Length of records does not match population size")

        order = np.argsort(fitness)
        gen_best_idx = int(order[0])
        gen_best_fit = float(fitness[gen_best_idx])

        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_individual = population[gen_best_idx].copy()
            best_record = records[gen_best_idx].copy()

        history.append({
            "generation": gen,
            "best_fitness": gen_best_fit,
            "mean_fitness": float(np.mean(fitness)),
            "best_individual": population[gen_best_idx].copy(),
            "best_record": records[gen_best_idx].copy(),
            "meta": meta.copy() if isinstance(meta, dict) else meta,
        })

        if verbose:
            msg = (
                f"[Gen {gen + 1}/{n_gen}] "
                f"best_fitness = {gen_best_fit:.6f}, "
                f"mean_fitness = {float(np.mean(fitness)):.6f}"
            )
            if isinstance(meta, dict):
                if "db_hit_count" in meta:
                    msg += f", db_hit = {meta['db_hit_count']}"
                if "selected_for_physics" in meta:
                    msg += f", physics_selected = {meta['selected_for_physics']}"
                if "n_physics" in meta:
                    msg += f", physics_eval = {meta['n_physics']}"
            print(msg)

        elites = population[order[:elite_size]].copy()
        next_population = np.empty((pop_size, n_genes), dtype=float)
        fill_ptr = 0

        if elite_size > 0:
            next_population[:elite_size] = elites
            fill_ptr = elite_size

        while fill_ptr < pop_size:
            idx1, idx2 = roulette_select_rankbased_indices(rng, fitness, n_select=2)

            parent1 = population[idx1]
            parent2 = population[idx2]

            child1, child2 = two_point_crossover_array(
                parent1=parent1,
                parent2=parent2,
                rng=rng,
                crossover_rate=crossover_rate,
            )

            children = np.vstack([child1, child2])
            children = mutate_population_array(
                children,
                low_vec=low_vec,
                high_vec=high_vec,
                is_int_mask=is_int_mask,
                rng=rng,
                mutation_rate=mutation_rate,
                float_decimals=float_decimals,
            )

            next_population[fill_ptr] = children[0]
            fill_ptr += 1

            if fill_ptr < pop_size:
                next_population[fill_ptr] = children[1]
                fill_ptr += 1

        population = next_population

    final_df = population_array_to_df(population, gene_names)
    final_fitness, final_records, final_meta = evaluate_population_fn(final_df, **eval_kwargs)
    final_fitness = np.asarray(final_fitness, dtype=float)

    final_order = np.argsort(final_fitness)
    final_best_idx = int(final_order[0])

    return {
        "best_individual": best_individual,
        "best_fitness": float(best_fitness),
        "best_record": best_record,
        "final_population": final_df,
        "final_fitness": final_fitness.tolist(),
        "final_records": final_records,
        "final_meta": final_meta.copy() if isinstance(final_meta, dict) else final_meta,
        "history": history,
        "final_best_individual": population[final_best_idx].copy(),
        "final_best_fitness": float(final_fitness[final_best_idx]),
        "final_best_record": final_records[final_best_idx].copy(),
    }