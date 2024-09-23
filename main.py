import openai
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configurar logs
logging.basicConfig(level=logging.INFO)

# Configurar a chave da API OpenAI
openai.api_key = '<your-token>'
# Carregar o modelo de embeddings e fact checker globalmente para evitar recarregamentos repetidos
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device='cuda')
fact_checker = pipeline("text-classification", model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device=0)

def check_factuality(response):
    try:
        if isinstance(response, list):
            response = ' '.join(response)

        result = fact_checker(response)

        label = result[0]['label'].lower()
        score = result[0]['score']

        if label == 'entailment':
            factuality_score = score
        elif label == 'neutral':
            factuality_score = score * 0.5
        else:
            factuality_score = 0.0

        return factuality_score
    except Exception as e:
        raise e


# Função para gerar respostas usando GPT-3.5 via API da OpenAI
def generate_gpt_3_5_responses(solicitation):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": solicitation}],
            max_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return remove_repetitions(generated_text)
    except Exception as e:
        logging.error(f"Erro ao chamar API do GPT-3.5: {e}")
        return ""

def remove_repetitions(text):
    sentences = [sentence.strip() for sentence in text.split('.')]

    sentences = [sentence for sentence in sentences if len(sentence) > 0]

    unique_sentences = list(dict.fromkeys(sentences))

    return '. '.join(unique_sentences) + '.'

# Função para calcular a similaridade semântica entre sentenças usando embeddings
def calculate_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()

# Função para avaliar coerência entre múltiplas respostas
def evaluate_coherence(responses):
    if len(responses) < 2:
        return 0

    coherence_score = 0
    for i in range(len(responses) - 1):
        coherence_score += calculate_similarity(responses[i], responses[i + 1])

    return coherence_score / (len(responses) - 1)

# Função de crossover baseado em similaridade
def crossover(parent1, parent2):
    child = []
    for sentence1 in parent1:
        similarities = [calculate_similarity(sentence1, sentence2) for sentence2 in parent2]
        max_sim_idx = similarities.index(max(similarities))
        child.append(sentence1 if similarities[max_sim_idx] > 0.5 else parent2[max_sim_idx])
    return child

# Função de mutação aleatória
def mutate(solution, mutation_rate=0.1):
    if random.random() < mutation_rate and len(solution) > 1:
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        logging.info(f"Mutação aplicada: Reordenar sentenças {idx1} e {idx2}")
    return solution

# Função de fitness - avalia a solução com base em factualidade e coerência
def fitness(solution):
    factuality_score = check_factuality(' '.join(solution))
    coherence_score = evaluate_coherence(solution)

    # Retornar o score final com pesos aplicados
    return 0.7 * factuality_score + 0.3 * coherence_score, factuality_score, coherence_score


# Algoritmo Genético com condição de parada ao atingir fitness máximo de 1
def genetic_algorithm(population, generations, mutation_rate=0.1):
    generation_data = []
    for generation in range(generations):
        # Avaliação da população com função de fitness
        scores = [(fitness(solution), solution) for solution in population]

        # Coletar factualidade e coerência para cada solução
        fitness_scores = [(f[0], f[1], f[2], solution) for (f, solution) in scores]

        # Selecionar melhores soluções (top 50%)
        population = [solution for _, _, _, solution in sorted(fitness_scores, reverse=True)[:len(population) // 2]]
        new_population = []

        # Verificar se alguma solução atingiu o fitness máximo de 0.9
        best_solution = max(fitness_scores, key=lambda x: x[0])
        if best_solution[0] == 0.9:
            logging.info(f"Geração {generation}: Melhor solução encontrada com fitness máximo de 1.0")
            return best_solution[3], best_solution[0], generation_data

        while len(new_population) < len(population):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population.extend(new_population)

        # Armazenar todos os dados de cada geração para análise posterior
        for score, factuality, coherence, solution in fitness_scores:
            generation_data.append({
                "generation": generation,
                "score": score,
                "factuality": factuality,
                "coherence": coherence,
                "solution": ' '.join(solution)
            })

        logging.info(f"Geração {generation}: Melhor pontuação: {best_solution[0]}")

    return best_solution[3], best_solution[0], generation_data

# Função para gerar as respostas em paralelo
def generate_responses_parallel(solicitation, pop_init):
    with ThreadPoolExecutor(max_workers=150) as executor:
        futures = [executor.submit(generate_gpt_3_5_responses, solicitation) for _ in range(pop_init)]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results

# Função para rodar o experimento e exportar os resultados para CSV
def run_experiment():
    solicitations = [
        "Escreva um texto explicando as causas da Revolução Francesa e seus impactos na sociedade europeia.",
        "Descreva os principais avanços tecnológicos do século XIX e como eles influenciaram a Revolução Industrial.",
        "Explique como a energia nuclear afeta o meio ambiente, tanto em termos de uso pacífico quanto de desastres nucleares.",
        "Escreva sobre as tradições culturais de uma civilização antiga e pouco conhecida."
    ]

    # DataFrames para armazenar os dados das populações iniciais e das gerações
    df_initial_population = pd.DataFrame(columns=['Solicitation', 'Initial_Population_Size', 'Initial_Response'])
    df_generations = pd.DataFrame(columns=['Solicitation', 'Initial_Population_Size', 'Generation', 'Score', 'Factuality', 'Coherence', 'Response'])

    # Testar com diferentes tamanhos de população inicial
    population_sizes = [10,100]
    generations = 100
    mutation_rate = 0.5

    for solicitation in solicitations:
        for pop_size in population_sizes:
            logging.info(f"Rodando experimento com {pop_size} indivíduos na população inicial.")

            # Gerar a população inicial
            population_responses = generate_responses_parallel(solicitation, pop_init=pop_size)
            population = [response.split('. ') for response in population_responses]

            # Salvar as respostas iniciais e o tamanho da população no DataFrame
            for initial_response in population:
                df_initial_population = pd.concat([df_initial_population, pd.DataFrame({
                    'Solicitation': [solicitation],
                    'Initial_Population_Size': [pop_size],
                    'Initial_Response': [' '.join(initial_response)]
                })], ignore_index=True)

            # Executar o algoritmo genético
            best_solution, best_score, generation_data = genetic_algorithm(population, generations=generations, mutation_rate=mutation_rate)

            # Salvar os dados das gerações
            for data in generation_data:
                df_generations = pd.concat([df_generations, pd.DataFrame({
                    'Solicitation': [solicitation],
                    'Initial_Population_Size': [pop_size],
                    'Generation': [data['generation']],
                    'Score': [data['score']],
                    'Factuality': [data['factuality']],
                    'Coherence': [data['coherence']],
                    'Response': [data['solution']]
                })], ignore_index=True)

    # Salvar os resultados em CSV
    df_initial_population.to_csv('initial_population.csv', index=False)
    df_generations.to_csv('generations_data.csv', index=False)
    logging.info("Resultados exportados para 'initial_population.csv' e 'generations_data.csv'.")


if __name__ == '__main__':
    inicio = datetime.now()
    run_experiment()
    fim = datetime.now()
    print(fim - inicio)
