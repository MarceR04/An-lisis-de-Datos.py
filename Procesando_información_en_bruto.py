import requests

url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
response = requests.get(url)

with open("heart_failure_clinical_records_dataset.csv", "w") as f:
    f.write(response.text)


def download_csv(url: str, filename: str) -> None:
    response = requests.get(url)
    with open(filename, "w") as f:
        f.write(response.text)
