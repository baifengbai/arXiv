
import os
from os.path import join, realpath, dirname
import pandas as pd
import numpy as np
import requests
import shlex
import textwrap
import string
from datetime import date, timedelta
import nltk.data
from nltk.corpus import stopwords


def main():
    '''
    Query newly added articles to selected arXiv categories, rank them
    according to given keywords, and print out the ranked list.

    This method ranks articles from 0 to 4, extract the unique words in them,
    and stores the result in a csv file.

    Not very good, if a word is present in more than one rank it will tend
    to cancel itself out when obtaining the probability.
    '''

    mypath = realpath(join(os.getcwd(), dirname(__file__), 'input'))
    nltk.data.path.append(mypath)

    # Read accepted/rejected keywords and categories from file.
    mode, date_range, in_k, ou_k, categs = get_in_out()

    wordsRank = readWords()

    dates_no_wknds = ['']
    if mode == 'range':
        start_date, end_date, dates_no_wknds = dateRange(date_range)
        print("\nDownloading arXiv data for range {} / {}".format(
            start_date, end_date))
    elif mode == 'recent':
        print("\nDownloading recent arXiv data.")
    else:
        raise ValueError("Unknown mode '{}'".format(mode))

    print("Categories selected: {}".format(', '.join(categs)))
    # articles = []
    # for day_week in dates_no_wknds:
    #     # Get new data from all the selected categories.
    #     for cat_indx, categ in enumerate(categs):

    #         # Get data from each category.
    #         soup = get_arxiv_data(categ, day_week)

    #         # Store titles, links, authors and abstracts into list.
    #         articles = articles + get_articles(soup)

    import pickle
    # with open('filename.pickle', 'wb') as f:
    #     pickle.dump(articles, f)
    with open('filename.pickle', 'rb') as f:
        articles = pickle.load(f)

    # Clean title and abstract.
    clTitle, clAbs = cleanText(articles)

    # Obtain articles' probabilities according to keywords.
    K_prob = get_Kprob(clTitle, clAbs, wordsRank)
    # Sort articles.
    articles, K_prob = sort_rev(articles, K_prob)

    newRank = manualRank(articles, K_prob, clTitle, clAbs)

    updtRank(wordsRank, newRank)

    print("\nFinished.")


def get_in_out():
    '''
    Reads in/out keywords from file.
    '''
    in_k, ou_k, categs = [], [], []
    with open("keywords.dat", "r") as ff:
        for li in ff:
            if not li.startswith("#"):
                # Mode.
                if li[0:2] == 'MO':
                    # Store each keyword separately in list.
                    mode, start_date, end_date = shlex.split(li[3:])
                # Categories.
                if li[0:2] == 'CA':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        categs.append(i)
                # Accepted keywords.
                if li[0:2] == 'IN':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        in_k.append(i)
                # Rejected keywords.
                if li[0:2] == 'OU':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        ou_k.append(i)

    return mode, [start_date, end_date], in_k, ou_k, categs


def readWords():
    """
    Read ranked words from input file.
    """
    wordsRank = pd.read_csv("input/wordsRank.dat")
    return wordsRank


def dateRange(date_range):
    """
    Store individual dates for a range, skipping weekends.
    """
    start_date = list(map(int, date_range[0].split('-')))
    end_date = list(map(int, date_range[1].split('-')))

    ini_date, end_date = date(*start_date), date(*end_date)

    d, delta, weekend = ini_date, timedelta(days=1), [5, 6]
    dates_no_wknds = []
    while d <= end_date:
        if d.weekday() not in weekend:
            # Store as [year, month, day]
            dates_no_wknds.append(str(d).split('-'))
        d += delta

    return ini_date, end_date, dates_no_wknds


def get_arxiv_data(categ, day_week):
    '''
    Downloads data from arXiv.
    '''
    if day_week == '':
        url = "http://arxiv.org/list/" + categ + "/new"
    else:
        year, month, day = day_week
        url = "https://arxiv.org/catchup?smonth=" + month + "&group=grp_&s" +\
              "day=" + day + "&num=50&archive=astro-ph&method=with&syear=" +\
              year

    html = requests.get(url)
    soup = BS(html.content, 'lxml')

    return soup


def get_articles(soup):
    '''
    Splits articles into lists containing title, abstract, authors and link.
    Article info is located between <dt> and </dd> tags.
    '''
    # Get links.
    links = ['https://' + _.text.split()[0].replace(':', '.org/abs/') for _ in
             soup.find_all(class_="list-identifier")]
    # Get titles.
    titles = [_.text.replace('\n', '').replace('Title: ', '') for _ in
              soup.find_all(class_="list-title mathjax")]
    # Get authors.
    authors = [_.text.replace('\n', '').replace('Authors:', '')
               for _ in soup.find_all(class_="list-authors")]
    # Get abstract.
    abstracts = [_.text.replace('\n', ' ') for _
                 in soup.find_all('p', class_="mathjax")]

    articles = list(zip(*[authors, titles, abstracts, links]))

    return articles


def cleanText(articles):
    """
    """
    stpwrds = stopwords.words("english") +\
        ['find', 'data', 'observed', 'using', 'show', 'showed', 'well',
         'around', 'used', 'thus', 'within', 'investigate', 'also',
         'recently', 'however', 'even', 'institute', 'taken']

    clean_text = [[], []]
    for art in articles:
        title, abstr = art[1], art[2]
        for i, text in enumerate((title, abstr)):
            # Remove punctuation.
            translator = str.maketrans('', '', string.punctuation)
            # To lowercase.
            text = str(text).lower().translate(translator).split()
            # Remove stopwords and some common words.
            clean_text[i].append([w for w in text if w not in stpwrds])

    return clean_text


def get_Kprob(clTitle, clAbs, wordsRank):
    '''
    Obtains keyword base probabilities for each article, according to the
    in/out keywords.
    '''
    art_K_prob = []
    # Loop through each article stored.
    for i, title in enumerate(clTitle):

        K_prob = 0.
        for w in title:
            if w in wordsRank['0rank']:
                K_prob += 0. * w
            if w in wordsRank['1rank']:
                K_prob += 1. * w
            if w in wordsRank['2rank']:
                K_prob += 2. * w
            if w in wordsRank['3rank']:
                K_prob += 3. * w
            if w in wordsRank['4rank']:
                K_prob += 4. * w
        for w in clAbs[i]:
            if w in wordsRank['0rank']:
                K_prob += 0. * w
            if w in wordsRank['1rank']:
                K_prob += 1. * w
            if w in wordsRank['2rank']:
                K_prob += 2. * w
            if w in wordsRank['3rank']:
                K_prob += 3. * w
            if w in wordsRank['4rank']:
                K_prob += 4. * w

        art_K_prob.append(K_prob)

    return art_K_prob


def sort_rev(articles, K_prob):
    '''
    Sort articles according to rank so larger values will be located last.
    '''
    articles = [x for _, x in sorted(zip(K_prob, articles))]

    return articles, sorted(K_prob)


def manualRank(articles, K_prob, clTitle, clAbs):
    """
    """
    newRank = {'0rank': [], '1rank': [], '2rank': [], '3rank': [], '4rank': []}
    for i, art in enumerate(articles):
        # Title
        title = str(art[1])
        print('\n{}) (P={:.2f}) {}'.format(
            str(len(articles) - i), K_prob[i], textwrap.fill(title, 70)))
        # Authors + arXiv link
        authors = art[0] if len(art[0].split(',')) < 4 else\
            ','.join(art[0].split(',')[:3]) + ', et al.'
        print(textwrap.fill(authors, 77), '\n* ' + str(art[3]) + '\n')
        # Abstract
        print(textwrap.fill(str(art[2]), 80))

        # Rank
        while True:
            pn = input("Rank (0 to 4): ")
            # import random
            # pn = random.choice(['0', '1', '2', '3', '4'])
            if pn in ['0', '1', '2', '3', '4']:
                pn = pn + 'rank'
                newRank[pn] += clTitle[i] + clAbs[i]
                break
            elif pn in ['q', 'quit', 'quit()', 'exit']:
                return newRank

    return newRank


def updtRank(wordsRank, newRank):
    """
    Update the ranked words file.
    """

    col0 = np.array(list(set(newRank['0rank'] + list(wordsRank['0rank']))))
    col1 = np.array(list(set(newRank['1rank'] + list(wordsRank['1rank']))))
    col2 = np.array(list(set(newRank['2rank'] + list(wordsRank['2rank']))))
    col3 = np.array(list(set(newRank['3rank'] + list(wordsRank['3rank']))))
    col4 = np.array(list(set(newRank['4rank'] + list(wordsRank['4rank']))))

    df0 = pd.DataFrame({'0rank': col0})
    df1 = pd.DataFrame({'1rank': col1})
    df2 = pd.DataFrame({'2rank': col2})
    df3 = pd.DataFrame({'3rank': col3})
    df4 = pd.DataFrame({'4rank': col4})

    df = pd.concat([df0, df1, df2, df3, df4], axis=1)
    df.to_csv("input/wordsRank.dat", index=False)



if __name__ == "__main__":
    main()
