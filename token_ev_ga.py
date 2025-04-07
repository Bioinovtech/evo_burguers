import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline
from deap import base, creator, tools
import random
from PIL import Image
import src.simulacra_rank_image as simulacra_rank_image
import src.laion_rank_image as laion_rank_image
import copy
import argparse
import sys
import os
import matplotlib.pyplot as plt
import yaml

parser = argparse.ArgumentParser(description='Receives argument seed (int).')

parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--seed_path', type=str, help='Path to seed list file')
parser.add_argument('--cuda', type=int, help='Cuda GPU to use')
parser.add_argument('--predictor', type=int, help='Aesthetic predictor to use\n0 - SAM\n1 - LAION')

args = parser.parse_args()

if args.seed_path is not None:
    SEED_PATH = args.seed_path
elif args.seed is not None:
    print("Seed path not provided, executing single run")
    SEED = args.seed
    SEED_PATH = None
else:
    print("Seed path not provided, executing single run")
    print("Seed not provided, default is 42")
    SEED = 42
    SEED_PATH = None

if args.cuda is not None:
    cuda_n = str(args.cuda)
else:
    print("Cuda device not provided, default is 0")
    cuda_n = str(0)

if args.predictor is not None:
    predictor = args.predictor
else:
    print("Aesthetic predictor not provided, default is 0 (SAM)")
    predictor = 0

# Load parameters from the YAML file
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Extract algorithm parameters
CROSSOVER_PROB = config["algorithm"]["crossover_prob"]
MUTATION_PROB = config["algorithm"]["mutation_prob"]
IND_MUTATION_PROB = config["algorithm"]["ind_mutation_prob"]
NUM_GENERATIONS = config["algorithm"]["num_generations"]
POP_SIZE = config["algorithm"]["pop_size"]
TOURNMENT_SIZE = config["algorithm"]["tournment_size"]
ELITISM = config["algorithm"]["elitism"]
LAMBDA = config["algorithm"]["lambda"]
num_inference_steps = config["algorithm"]["num_inference_steps"]
guidance_scale = config["algorithm"]["guidance_scale"]
VECTOR_SIZE = config["algorithm"]["vector_size"]
height = config["algorithm"]["height"]
width = config["algorithm"]["width"]
model_id = config["algorithm"]["model_id"]
experience_name = config["algorithm"]["experience_name"]

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

# Check if a GPU is available and if not, use the CPU
device = "cuda:"+cuda_n  if torch.cuda.is_available() else "cpu"

# Load the components of the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

if predictor == 1:
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
else:
    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)

MIN_VALUE, MAX_VALUE = 0, pipe.tokenizer.vocab_size-3
START_OF_TEXT, END_OF_TEXT = pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

latents = []

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Define the aesthetic evaluation function
def aesthetic_evaluation(image):
    image_input = image.permute(2, 0, 1).to(torch.float32)  
    if predictor == 0:
        aesthetic_score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 1 or predictor == 2:
        pil_image = Image.fromarray((image))
        aesthetic_score = aesthetic_model.predict_from_tensor(image_input)
    else:
        # outras metricas aqui
        return 0
    return aesthetic_score.item()

# Function to generate an image from text embeddings
def generate_image_from_embeddings(token_vector, num_inference_steps=25):
    tmp_token_vector = np.insert(token_vector.cpu().detach().numpy().flatten(), 0, START_OF_TEXT)

    padding_size =  pipe.tokenizer.model_max_length - len(tmp_token_vector.flatten())

    tmp_token_vector = np.append(tmp_token_vector, [END_OF_TEXT] * padding_size)
    tmp_token_vector = torch.tensor(tmp_token_vector, dtype=torch.int64).to(device)
    tmp_token_vector = torch.clamp(tmp_token_vector, MIN_VALUE, MAX_VALUE).view(1, len(tmp_token_vector)).to(device)

    text_embeddings = pipe.text_encoder(tmp_token_vector)[0]

    # Concatenate the unconditional and text embeddings
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

    tmp_latents = copy.deepcopy(latents)

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([tmp_latents] * 2) if guidance_scale > 1.0 else tmp_latents
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        tmp_latents = pipe.scheduler.step(noise_pred, t, tmp_latents)["prev_sample"]

    with torch.no_grad():
        image = pipe.vae.decode(tmp_latents / pipe.vae.config.scaling_factor)["sample"]

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    image = image.squeeze(0).permute(1, 2, 0)  # Convert to [H, W, C]

    return image

# Tokenize and encode the initial prompt
prompt = ""
#token_vector = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length - 2, return_tensors="pt").input_ids

def evaluate(individual):
    text_embeddings = torch.tensor(individual, device=device).unsqueeze(0)
    image = generate_image_from_embeddings(text_embeddings)
    score = aesthetic_evaluation(image)
    return score,

# Registering exponential mutation for integers
def mutExponential(individual, lambd, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            # Draw a number from an exponential distribution and add it to the gene
            delta = 1 + random.expovariate(lambd)
            # Randomly decide if the delta should be positive or negative
            if random.random() < 0.5:
                delta = -delta
            individual[i] += int(round(delta))
            # Ensure the mutated value is within bounds
            if individual[i] < low:
                individual[i] = low
            elif individual[i] > up:
                individual[i] = up
    return individual,

def main(seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    results_folder = "results/"+experience_name+"_results_"+str(predictor)+"_"+str(seed)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    num_channels_latents = pipe.unet.in_channels

    global latents
    latents = torch.randn((1, num_channels_latents, height // 8, width // 8), device=device, generator=generator, requires_grad=False)

    # Genetic Algorithm setup
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, MIN_VALUE, MAX_VALUE)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, VECTOR_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=MIN_VALUE, up=MAX_VALUE, indpb=IND_MUTATION_PROB)
    #toolbox.register("mutate", mutExponential, lambd=LAMBDA, low=MIN_VALUE, up=MAX_VALUE, indpb=IND_MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNMENT_SIZE)
    toolbox.register("evaluate", evaluate)

    random.seed(seed)
    population = toolbox.population(n=POP_SIZE)

    max_fit_list = []
    avg_fit_list = []
    std_fit_list = []
    best_list = []
    prompt_list = []
    for gen in range(NUM_GENERATIONS):
        elites = tools.selBest(population, ELITISM)
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        offspring = offspring + elites
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        max_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        std_fit = np.std(fits)
        print(f"Gen {gen + 1}: Max fitness {max_fit}, Avg fitness {avg_fit}")

        # Generate and display the best image
        best_ind = tools.selBest(population, 1)[0]
        prompt = detokenize(best_ind)

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        std_fit_list.append(std_fit)
        best_list.append(best_ind)
        prompt_list.append(prompt)

        best_text_embeddings = torch.tensor(best_ind, device=device).unsqueeze(0)
        best_image = generate_image_from_embeddings(best_text_embeddings)
        best_image_np = best_image.detach().cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray((best_image_np))
        pil_image.save(results_folder+"/best_%d.png" % (gen+1))

    best_ind = tools.selBest(population, 1)[0]
    print("Seed %d, best individual is %s, with fitness: %s" % (seed, best_ind, best_ind.fitness.values))

    # Generate and display the best image
    best_text_embeddings = torch.tensor(best_ind, device=device).unsqueeze(0)
    best_image = generate_image_from_embeddings(best_text_embeddings)
    best_image_np = best_image.detach().cpu().numpy()
    best_image_np = (best_image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray((best_image_np))
    pil_image.save(results_folder+"/best_all.png")

    results = pd.DataFrame({"generation": list(range(1,NUM_GENERATIONS+1)), "best_fitness": max_fit_list, 
                  "average_fitness": avg_fit_list, "std_fitness": std_fit_list, 
                  "best_individual": best_list, "prompt": prompt_list})
    
    results.to_csv(results_folder+"/fitness_results.csv", index=False)

    save_plot_results(results, results_folder)

def detokenize(individual):
    tmp_solution = torch.tensor(individual, dtype=torch.int64)
    tmp_solution = torch.clamp(tmp_solution, 0, pipe.tokenizer.vocab_size - 1)
    decoded_string = pipe.tokenizer.decode(tmp_solution, skip_special_tokens=True, clean_up_tokenization_spaces = True)

    return decoded_string

def plot_mean_std(x_axis, m_vec, std_vec, description, title = None, y_label = None, x_label = None):
    lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
    upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]
    
    plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
    plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")  
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

def save_plot_results(results, results_folder):
    plt.figure()
    plot_mean_std(results['generation'], results['average_fitness'], results['std_fitness'], "Population")
    plt.plot(results['generation'], results['best_fitness'], 'r-', label="Best")
    plt.ylim(0, 10)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Simulacra Score)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder+"/fitness_evolution.png")

if __name__ == "__main__":
    for seed in seed_list:
        main(seed)
        print(f"Run with seed {seed} finished!")
