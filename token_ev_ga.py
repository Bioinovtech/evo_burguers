from __future__ import annotations

import argparse
import os
import random
from itertools import cycle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import StableDiffusionPipeline
from deap import base, creator, tools
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetic prompt search with SD")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--seed_path", type=str, help="Path to a file with one seed per line")
    parser.add_argument("--cuda", type=str, default="0", help="Comma‑separated GPU indices (e.g. '0,1') or empty for CPU")
    parser.add_argument("--predictor", type=int, default=0, choices=(0, 1), help="Aesthetic model: 0=Simulacra, 1=LAION")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_pipeline(model_id: str, device: str, dtype: torch.dtype = torch.float16) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline on the given device."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def generate_image_from_embeddings(
    token_vector: torch.Tensor,
    *,
    pipe: StableDiffusionPipeline,
    device: str,
    guidance_scale: float,
    num_inference_steps: int,
    latents: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    min_token_id: int,
    max_token_id: int,
) -> torch.Tensor:
    """Run a single forward diffusion pass and return a (H,W,C) float32 image in [0,1]."""
    # ------------------------------------------------------------------ tokens
    with torch.no_grad():
        tmp_vec = token_vector.clone().to(torch.long).cpu().numpy().flatten()
        tmp_vec = np.insert(tmp_vec, 0, start_token_id)
        # right‑pad / truncate
        pad = pipe.tokenizer.model_max_length - len(tmp_vec)
        if pad < 0:
            tmp_vec = tmp_vec[: pipe.tokenizer.model_max_length]
            pad = 0
        tmp_vec = np.append(tmp_vec, [end_token_id] * pad)
        tmp_vec = torch.tensor(tmp_vec, device=device)
        tmp_vec.clamp_(min_token_id, max_token_id)
        tmp_vec = tmp_vec.view(1, -1)

        text_embeddings = pipe.text_encoder(tmp_vec)[0]
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        # -------------------------------------------------------------- denoising
        latents_t = latents.clone()
        for t in pipe.scheduler.timesteps:
            latent_model_input = torch.cat([latents_t] * 2) if guidance_scale > 1.0 else latents_t
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents_t = pipe.scheduler.step(noise_pred, t, latents_t)["prev_sample"]

        image = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor)["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.squeeze(0).permute(1, 2, 0)  # (H,W,C)


# -----------------------------------------------------------------------------
# Main GA logic
# -----------------------------------------------------------------------------

def run_for_seed(seed: int, cfg: dict, devices: List[str], predictor_choice: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    # -------------------------- load SD pipelines & predictors per device
    pipes: List[StableDiffusionPipeline] = []
    predictors = []
    for dev in devices:
        pipe = build_pipeline(cfg["model_id"], dev)
        pipe.scheduler.set_timesteps(cfg["num_inference_steps"])
        pipes.append(pipe)
        if predictor_choice == 1:
            from src.laion_rank_image import LAIONAesthetic

            predictors.append(LAIONAesthetic(dev))
        else:
            from src.simulacra_rank_image import SimulacraAesthetic

            predictors.append(SimulacraAesthetic(dev))

    # -------------------------------------------------------------- tokens info
    tokenizer = pipes[0].tokenizer
    start_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.eos_token_id
    min_token_id = 0
    max_token_id = tokenizer.vocab_size - 1

    # -------------------------------------------------------------- DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, min_token_id, max_token_id)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, cfg["vector_size"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=min_token_id, up=max_token_id, indpb=cfg["ind_mutation_prob"]
    )
    toolbox.register("select", tools.selTournament, tournsize=cfg["tournament_size"])

    population = toolbox.population(n=cfg["pop_size"])

    # -------------------------------------------------------------- run GA
    results_dir = Path("results") / f"{cfg['experience_name']}_pred{predictor_choice}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Tuple[float, float, float]] = []  # max, avg, std per gen

    # resource cycle for round‑robin scheduling
    device_cycle = cycle(range(len(devices)))

    # Thread pool stays alive for the full run
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for gen in range(cfg["num_generations"]):
            # ---------------- selection / var
            elites = tools.selBest(population, cfg["elitism"])
            offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg["crossover_prob"]:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for mut in offspring:
                if random.random() < cfg["mutation_prob"]:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            offspring.extend(elites)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            # ------------- schedule fitness evaluation across GPUs (threads)
            futures = []
            for ind in invalid_ind:
                dev_idx = next(device_cycle)
                futures.append(
                    pool.submit(
                        _evaluate_individual,
                        ind,
                        pipes[dev_idx],
                        predictors[dev_idx],
                        devices[dev_idx],
                        cfg,
                        start_token_id,
                        end_token_id,
                        min_token_id,
                        max_token_id,
                        seed,
                    )
                )
            for ind, fut in zip(invalid_ind, futures):
                ind.fitness.values = (fut.result(),)

            population[:] = offspring

            # ------------- logging & artefacts
            fits = [ind.fitness.values[0] for ind in population]
            max_fit, avg_fit, std_fit = max(fits), float(np.mean(fits)), float(np.std(fits))
            metrics.append((max_fit, avg_fit, std_fit))
            print(f"[Seed {seed}] Gen {gen+1}/{cfg['num_generations']}  max={max_fit:.3f}  avg={avg_fit:.3f}")

            best_ind = tools.selBest(population, 1)[0]
            _save_best_image(best_ind, pipes[0], devices[0], cfg, results_dir, gen, start_token_id, end_token_id, min_token_id, max_token_id)

    # ----------------------------------------------------------- save metrics
    df = pd.DataFrame(metrics, columns=["best", "average", "std"])
    df.insert(0, "generation", np.arange(1, cfg["num_generations"] + 1))
    df.to_csv(results_dir / "fitness_results.csv", index=False)
    _plot_results(df, results_dir)

    best_of_run = tools.selBest(population, 1)[0]
    print(f"Seed {seed} finished  → best fitness {best_of_run.fitness.values[0]:.3f}")


# -----------------------------------------------------------------------------
# Fitness evaluation helper (runs inside thread)
# -----------------------------------------------------------------------------

def _evaluate_individual(
    individual: np.ndarray,
    pipe: StableDiffusionPipeline,
    predictor,
    device: str,
    cfg: dict,
    start_token_id: int,
    end_token_id: int,
    min_token_id: int,
    max_token_id: int,
    base_seed: int,
) -> float:
    torch.manual_seed(base_seed + int(torch.randint(0, 10000, ()).item()))
    num_channels = pipe.unet.in_channels

    latents = torch.randn(
        (1, num_channels, cfg["height"] // 8, cfg["width"] // 8), device=device, dtype=pipe.dtype
    )

    uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]

    token_tensor = torch.tensor(individual, device=device)
    image = generate_image_from_embeddings(
        token_tensor,
        pipe=pipe,
        device=device,
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        latents=latents,
        uncond_embeddings=uncond_embeddings,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        min_token_id=min_token_id,
        max_token_id=max_token_id,
    )

    # predictor expects CHW float32
    score = predictor.predict_from_tensor(image.permute(2, 0, 1).to(torch.float32)).item()
    return score


# -----------------------------------------------------------------------------
# Artefact helpers
# -----------------------------------------------------------------------------

def _save_best_image(
    ind: np.ndarray,
    pipe: StableDiffusionPipeline,
    device: str,
    cfg: dict,
    out_dir: Path,
    gen_idx: int,
    start_token_id: int,
    end_token_id: int,
    min_token_id: int,
    max_token_id: int,
) -> None:
    num_channels = pipe.unet.in_channels
    latents = torch.randn((1, num_channels, cfg["height"] // 8, cfg["width"] // 8), device=device, dtype=pipe.dtype)
    uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]

    img = generate_image_from_embeddings(
        torch.tensor(ind, device=device),
        pipe=pipe,
        device=device,
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        latents=latents,
        uncond_embeddings=uncond_embeddings,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        min_token_id=min_token_id,
        max_token_id=max_token_id,
    )

    img_np = (img.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(out_dir / f"best_gen{gen_idx+1}.png")


def _plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    gens = df["generation"]
    plt.fill_between(gens, df["average"] - df["std"], df["average"] + df["std"], alpha=0.3, label="Avg ± SD")
    plt.plot(gens, df["average"], "--", label="Average")
    plt.plot(gens, df["best"], label="Best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir / "fitness_evolution.png")
    plt.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    # --------------------------------------------------------------------- seed list
    seed_list = [args.seed]
    if args.seed_path and Path(args.seed_path).exists():
        with open(args.seed_path) as f:
            seed_list = [int(line.strip()) for line in f if line.strip()]

    # --------------------------------------------------------------------- config
    with open("config.yml") as f:
        cfg = yaml.safe_load(f)["algorithm"]

    # sanity: vector size must leave room for BOS/EOS
    assert cfg["vector_size"] <= cfg.get("prompt_max_tokens", 75), "VECTOR_SIZE too large"

    # --------------------------------------------------------------------- devices
    if torch.cuda.is_available() and args.cuda:
        device_strs = [f"cuda:{d}" for d in args.cuda.split(",") if d]
    else:
        device_strs = ["cpu"]
    print("Using devices:", ", ".join(device_strs))

    for s in seed_list:
        run_for_seed(s, cfg, device_strs, args.predictor)


if __name__ == "__main__":
    main()
