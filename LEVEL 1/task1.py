import requests 
from  bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://books.toscrape.com/catalogue/"

def scrape_page(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, 'html.parser')
    
    books = []
    for article in soup.find_all("article", class_="product_pod"):
        title = article.h3.a["title"]
        price = article.find("p", class_="price_color").text
        rating = article.find("p", class_="star-rating")["class"][1]

        books.append({"title": title, 
                      "price": price, 
                      "rating": rating})
    return books
 # Print the first book to verify the data

all_books = []
for page_num in range(1, 51):  # There are 50 pages to scrape
    url = BASE_URL + f"page-{page_num}.html"
    books = scrape_page(url)
    all_books.extend(books)
    print(f"Page {page_num}/50 — {len(books)} books scraped")

print(f"Total books scraped: {len(all_books)}")

df = pd.DataFrame(all_books)
df.to_csv("books.csv", index=False)
print("Data saved to books.csv")