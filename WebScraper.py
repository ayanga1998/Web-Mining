#!/usr/bin/env python
# coding: utf-8

# In[7]:


import requests
from bs4 import BeautifulSoup  
import pandas as pd
    
def getReviews(page_url):
    '''
    Extract information such as reviewer, source, date, etc. from each review for a given movie on Rotten Tomatoes
    Input:
        page_url: URL link for a single page of top critic reviews on Rotten Tomatoes
    
    Output:
        df: a DataFrame containing all the stored reviews and relevant information
    '''
    
    rows = []
    reviews= None
    page = requests.get(page_url)
    
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select("div.panel-body div div div.row")
        
        for idx, div in enumerate(divs):
            
            date = None
            score = None
            source = None
            content = None
            reviewer = None
            
            a_critic = div.select("div div div a.unstyled")
            if a_critic != []:
                reviewer = a_critic[0].get_text()
            
            em_source = div.select("div div div a em.subtle")
            if em_source != []:
                source = em_source[0].get_text()
                
            d_content = div.select("div div div div.the_review")
            if d_content != []:
                content = d_content[0].get_text().strip()
                
            d_date = div.select("div div div div.review-date")
            if d_date != []:
                date = d_date[0].get_text().strip()
            
            d_score = div.select("div div div div.small")
            if d_score != []:
                score = d_score[2].get_text().strip()
                if 'Original Score' not in score:
                    score = None
                else:
                    score = clean_content(score)
                
            rows.append([reviewer, source, content, date, score])
            
        df = pd.DataFrame(rows, columns=['reviewer', 'source', 'content', 'date', 'score'])
    return df

def clean_content(content):
    '''
    Cleans the data by removing all unneccessary words, spacing, or dividers
    
    Input:
        content: scraped text data from HTML tags
    
    Output:
        contnet: cleaned data ready to be used
    ''' 
    
    common_puncs = ['\r','\n', ':','|', 'Full Review', 'Original Score']
    for i in common_puncs:
        content = content.replace(i,'')
        
    content = content.strip()
    return content


def get_list_of_pages(url):
    '''
    Extract URL links for each page containing top critic reviews for a selected movie
    
    Input: 
        url: URL link to the first page in the top critic reviews for the movie
        
    Output:
        page_list: a list of URL's for each page of top critic reviews
        
    '''
    check = True
    page_list = [url]
    url_base = 'https://www.rottentomatoes.com'
    while check:
        next_page = get_newpage_url_ext(url)
        url = url_base + next_page
    
        if 'page' in url:
            page_list.append(url)
        else:
            break
    return page_list

def get_newpage_url_ext(url):
    ''' 
    Utilize the next button to scrape the link for the next page of a website
    
    Output: 
        new_link: The link to the next page of a sequence of pages
    '''
    page = requests.get(url)
    
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select("div.content div a.btn" )
        new_link = divs[3]['href']

        return new_link


def get_all_reviews(page_url):
    '''
    Crawls all pages of top critic reviews for a movie in Rotten Tomatoes
    
    Input: 
        page_url: The link to the top critic review page for a movie
        
    Output:
        reviews: A pandas dataframe containing all of the top critic reviews
    '''
    
    url_base = 'https://www.rottentomatoes.com'
    pages = get_list_of_pages(url)
    reviews = pd.DataFrame()
    
    for page in pages:
        df = getReviews(page)
        reviews = reviews.append(df)
    reviews.reset_index(drop = True, inplace = True)
    return reviews


# In[10]:


if __name__ == "__main__":  
    
    #url = 'https://www.rottentomatoes.com/m/soul_2020/reviews?type=top_critics'
    url = 'https://www.rottentomatoes.com/m/coco_2017/reviews?type=top_critics'
    
    # Test Question 1
    reviews = getReviews(url)
    print('========== Quetion 1 ============')
    print(reviews.head(5))
    
    # Test Question 2
    print('\n')
    print('========== Question 2 (BONUS) =============')
    reviews = get_all_reviews(url)
    print(reviews)

