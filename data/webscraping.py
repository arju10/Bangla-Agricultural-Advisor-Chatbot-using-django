import requests
from bs4 import BeautifulSoup
import csv

# Path to the dataset
DATASET_PATH = "data/agriculture_data.csv"

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract questions and answers (customize based on website structure)
    questions = [q.text.strip() for q in soup.find_all("div", class_="question")]
    answers = [a.text.strip() for a in soup.find_all("div", class_="answer")]
    
    return list(zip(questions, answers))

def update_dataset(new_data):
    with open(DATASET_PATH, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Category", "Question (Bangla)", "Answer (Bangla)"])
        writer.writerows(new_data)

# Example usage
import requests
url = "https://www.dae.gov.bd"
try:
    response = requests.get(url, verify=False)  # Disable SSL verification
    print(response.text)
except requests.exceptions.RequestException as e:
    print("Error:", e)
scraped_data = scrape_website(url)
new_data = [["Crop Advice", q, a] for q, a in scraped_data]
update_dataset(new_data)
print("Data updated successfully!")