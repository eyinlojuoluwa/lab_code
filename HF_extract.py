import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def extract_models_and_pipelines(pages=5):
    base_url = "https://huggingface.co/models?page="
    models_data = []

    for page_num in range(1, pages + 1):
        url = base_url + str(page_num)
        print(f"Fetching models from page {page_num}...")

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            for model in soup.find_all('h4',
                                       class_='text-md truncate font-mono text-black dark:group-hover/repo:text-yellow-500 group-hover/repo:text-indigo-600 text-smd'):
                model_name = model.text.strip()

                # Extract owner and model_name
                if '/' in model_name:
                    owner, model_name = model_name.split('/', 1)
                else:
                    print(f"Invalid model name format: {model_name}")
                    continue

                # Find the correct pipeline tag element
                pipeline_tag_element = model.find_next('div',
                                                       class_='mr-1 flex items-center overflow-hidden whitespace-nowrap text-sm leading-tight text-gray-400')
                if pipeline_tag_element:
                    pipeline_tag = pipeline_tag_element.text.strip().split('•')[0].strip()
                else:
                    pipeline_tag = 'Unknown'

                models_data.append({
                    'owner': owner.strip(),
                    'model_name': model_name.strip(),
                    'pipeline': pipeline_tag
                })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page_num}: {e}")

        time.sleep(2)  # Add a short delay to avoid overloading the server

    return models_data


def save_to_csv(data, filename='/home/local/SAIL/aajibode/paper_2/all_models.csv'):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Example usage:
if __name__ == "__main__":
    num_pages = 2
    models_data = extract_models_and_pipelines(pages=num_pages)
    save_to_csv(models_data)
