"""
Web module: Search the web, scrape, and automate browser.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By

def search_web(query):
    # Example: DuckDuckGo search
    driver = webdriver.Firefox()
    driver.get(f'https://duckduckgo.com/?q={query}')
    results = driver.find_elements(By.CSS_SELECTOR, 'a.result__a')
    links = [r.get_attribute('href') for r in results]
    driver.quit()
    return links
