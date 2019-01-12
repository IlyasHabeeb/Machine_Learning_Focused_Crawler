import csv
import datetime
import heapq
import itertools
import logging
import math
import pickle
import sys
from collections import OrderedDict, deque
from heapq import heapify, heappush, heappop
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import requests
from bs4 import BeautifulSoup
from url_normalize import url_normalize

import nltk
from nltk.corpus import wordnet

# Defining Variables to be Used
LOG = logging.getLogger(__name__)
BASE_URL = 'http://google.com/search?q='
STOP_WORDS = ['webcache', 'watch']

QUERY_SIZE_DICT = {'wildfires': 98600000, 'california': 2370000000,
                   'brooklyn': 609000000, 'dodgers': 105000000,
                   'shahrukh': 31000000, 'khan': 774000000,
                   'pangolin': 5420000, 'armadillo': 32700000,
                   'world': 15890000000, 'cup': 3240000000,
                   'hurricane': 432000000, 'florence': 398000000,
                   'mac': 2170000000, 'miller': 1030000000,
                   'kate': 1280000000, 'spade': 404000000,
                   'anthony': 809000000, 'bourdain': 12000000,
                   'black': 25270000000, 'panther': 414000000,
                   'mega': 1150000000, 'million': 2910000000,
                   'results': 11350000000, 'stan': 1350000000,
                   'lee': 2340000000, 'demi': 568000000,
                   'lovato': 115000000, 'election': 910000000}

N_SIZE = 25270000000
VISITED_URLS = set()
VISITED_URLS_DICT = {}
COUNTER = itertools.count()

try:
    crawler_to_run = sys.argv[1]
except Exception as e:
    print("Run File as: python Crawler_ML.py withoutML")
    print("or")
    print("Run File as: python Crawler_ML.py withML")
    sys.exit()

if crawler_to_run == 'withoutML':
    RUN_ML = 'False'
elif crawler_to_run == 'withML':
    RUN_ML = 'True'
else:
    print("Did not understand. Only understands 'withoutML' and 'withML'.")
    sys.exit()

QUERY = input(
    "Please Enter the Query in small letters (Words Should be Spaced): ")
N = int(input("Please Enter the Number of Pages to Crawl: "))

QUERY_SIMILAR = {}
for q in QUERY.split(' '):
    synonyms = set()

    for syn in wordnet.synsets(q):
        for l in syn.lemmas():
            if l.name().lower() != q:
                synonyms.add(l.name().lower())

    QUERY_SIMILAR[q] = synonyms


# Open-sourced from https://gist.github.com/matteodellamico/4451520
class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'
    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.
    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


VISITED_URLS_PRIOR_DICT = priority_dict(VISITED_URLS_DICT)


def allowed_to_crawl(url, host_url, scheme):
    '''
    url: the full url, string
    host_url: the domain url, string (eg. wikipedia.org)
    scheme: the communication protocol, string (eg. https)
    '''

    # if host URL is google, assume we are allowed to crawl
    if host_url == 'google.com':
        return True

    # if it is not a link, return False
    if host_url == '' or scheme == '':
        return False

    try:
        # get the robots.txt
        rp = RobotFileParser()
        rp.set_url(scheme + "://" + host_url + "/robots.txt")
        rp.read()

        return rp.can_fetch("*", url)

    except:
        pass

    return True


def mime_type_okay(url):
    '''
    url: the full url, string
    '''
    # url last 3 letters check img ,pdf ,mp4,mp3
    exten = url[-3:]
    cond = (exten == 'img' or exten ==
            'pdf' or exten == 'mp4' or exten == 'mp3')

    if cond:
        return False

    mime_ignore_list = ["image/mng", "image/bmp", "image/gif", "image/jpg",
                        "image/jpeg", "image/png", "image/pst", "image/psp",
                        "image/fif", "image/tiff", "image/ai", "image/drw",
                        "image/x-dwg", "audio/mp3", "audio/wma", "audio/mpeg",
                        "audio/wav", "audio/midi", "audio/mpeg3", "audio/mp4",
                        "audio/x-realaudio", "video/3gp", "video/avi",
                        "video/mov", "video/mp4", "video/mpg", "video/mpeg",
                        "video/wmv", "text/css", "application/x-pointplus",
                        "application/pdf", "application/octet-stream",
                        "application/x-binary", "application/zip",
                        "application/pdf"]

    try:
        session = requests.Session()
        session_resp = session.head(url)

        contentType = session_resp.headers['content-type']
        end_len = contentType.find(';')

        if contentType.split(";")[0] in mime_ignore_list:
            return False
        else:
            return True

    except Exception as e:
        LOG.error("Encountered this error: " + str(e))
        return False


def get_canonical(url):

    # only want to find canonical links for wikipedia
    # if 'wikipedia' not in url:
    #     return url

    canonical_url = ""

    try:
        resp = requests.get(url)
        header_url = resp.headers.get('content-type')

    except Exception as e:
        LOG.error(url+" is not a link")
        return None

    # If link is not html, don't get it
    if header_url != 'html':
        return None

    if resp.status_code == 200 and 'html' in header_url:
        soup = BeautifulSoup(resp.text, features="lxml")
        canonical = soup.find("link", rel="canonical")

        if canonical:
            canonical_url = canonical['href']

        else:

            try:
                og_url = soup.find("meta", property="og:url")
                canonical_url = og_url['content']

            except Exception as e:
                LOG.error("Something is wrong. Returning the same URL")
                return url

    return canonical_url


def cosine_score(justSoup, size):

    def compute_cosine_measure(t_dict, D):
        '''
        t_dict: A dictionary with frequency of query in doc,
        D: number of words, an int
        '''

        def compute_first_term(q):
            return math.log(1 + (N_SIZE/QUERY_SIZE_DICT[q]))

        def compute_second_term(q):

            # if the frequency is 0, return 0
            if t_dict[q] == 0:
                return 0
            else:
                return 1 + math.log(t_dict[q])

        res = 0

        for q in QUERY.split(' '):

            first_term = compute_first_term(q)
            second_term = compute_second_term(q)

            try:
                doc_size = math.sqrt(abs(D))
                res += (first_term * second_term) / doc_size

            except Exception as e:
                LOG.error("Size of the document is None. Setting it to 0...")
                res += 0

        return res

    query_weight_each_doc = {}

    for q in QUERY.split(' '):
        total_count = 0
        synonym_total_count = 0

        if size is not None:
            query_count = justSoup.get_text().lower().count(q.lower())
            total_count += query_count

            for value in QUERY_SIMILAR[q]:
                synonym_total_count += justSoup.get_text().lower().count(value.lower())

            synonym_total_count *= 0.5
            total_count += synonym_total_count
            query_weight_each_doc[q] = total_count
        else:
            query_weight_each_doc[q] = total_count

    D = len(justSoup.get_text().split())

    return compute_cosine_measure(query_weight_each_doc, D)


def estimate_promise(cosine_parent, curr_hyper_text, curr_link, ml):
    '''
    cosine_parent: cosine score of the parent, a float
    curr_hyper_text: hyperlink text of the child, a string
    curr_link: link of the child, a string
    '''

    total_text = len(curr_hyper_text.split())
    total_link_length = len(curr_link.split('/'))
    res = 0

    for q in QUERY.split(' '):
        q_in_hyper_text = curr_hyper_text.lower().count(q.lower())
        q_in_curr_link = curr_link.lower().count(q.lower())
        q_syn_hyper_count = 0
        q_syn_curr_count = 0

        for value in QUERY_SIMILAR[q]:
            q_syn_hyper_count += curr_hyper_text.lower().count(value.lower())
            q_syn_curr_count += curr_link.lower().count(value.lower())

        q_syn_hyper_count *= 0.5
        q_syn_curr_count *= 0.5

        idf_for_all = math.log2(N_SIZE/QUERY_SIZE_DICT[q])

        if total_text == 0:
            q_divide_hyper_text = 0  # tf
            q_syn_divide_hyper_text = 0
        else:
            q_divide_hyper_text = q_in_hyper_text/total_text  # tf
            q_syn_divide_hyper_text = q_syn_hyper_count/total_text
        # q_divide_hyper_text *= idf_for_all

        if total_link_length == 0:
            q_divide_curr_link = 0
            q_syn_divide_curr_link = 0
        else:
            q_divide_curr_link = q_in_curr_link/total_link_length
            q_syn_divide_curr_link = q_syn_curr_count/total_link_length

        res += (q_divide_hyper_text + q_divide_curr_link +
                q_syn_divide_curr_link + q_syn_divide_curr_link)

    int_promise = res * cosine_parent

    if ml == 'True':

        try:
            loaded_model = pickle.load(
                open('best_svr_model.sav', 'rb'))
            arr = np.array([cosine_parent, q_in_hyper_text, q_syn_hyper_count, total_text,
                            q_in_curr_link, q_syn_curr_count, total_link_length]).reshape(1, -1)
            # arr = np.array([int_promise/cosine_parent,
            #                 cosine_parent]).reshape(1, -1)
            promise = float(loaded_model.predict(arr))

        except Exception as e:
            LOG.error("Encountered Error: " + str(e))
            LOG.error("int_promise: "+str(int_promise) +
                      " cosine_parent: "+str(cosine_parent))
            promise = int_promise

    else:
        promise = int_promise

    return promise, [q_in_hyper_text, q_syn_hyper_count, total_text, q_in_curr_link, q_syn_curr_count, total_link_length]


def compute_harvest_score(cosine_list):

    threshold = np.nanmedian(cosine_list[:8])

    print("Threshold", threshold)

    relv_pages = 0

    for i in range(0, len(cosine_list)):

        if cosine_list[i] >= threshold:
            relv_pages += 1

    print("# of Relevant Pages:", relv_pages)
    print("Total Pages Crawled:", len(cosine_list))

    harv_score = relv_pages / len(cosine_list)

    return harv_score


def get_element_from_url(url):
    '''
    url: a string which contains the full URL,
    returns justSoup, the text stored in the URL
    '''
    host_url = urlparse(url).netloc
    scheme = urlparse(url).scheme

    cond = (allowed_to_crawl(url, host_url, scheme) and mime_type_okay(url))

    if cond:

        try:
            res = requests.get(url)

        except Exception as e:
            LOG.error(url+" is not a link")
            return None, None

        # extracting the text within the URL #.encode('utf-8').strip()
        justSoup = BeautifulSoup(res.text, features="lxml")

        return res, justSoup

    else:
        return None, None


def get_current_url_info(res, justSoup):
    '''
    res: request of the URL,
    justSoup: BeautifulSoup constructor of the URL
    returns list of current_url_info
    '''

    curr_url_info_list = []

    current_time = str(datetime.datetime.now())
    curr_url_info_list.append(current_time)

    if res is None or justSoup is None:
        curr_url_info_list.append(float('nan'))
        curr_url_info_list.append(float('nan'))
        curr_url_info_list.append(float('nan'))

    else:
        size = res.headers.get('Content-Length')
        curr_url_info_list.append(size)

        status_code = res.status_code
        curr_url_info_list.append(status_code)

        c_score = cosine_score(justSoup, size)
        curr_url_info_list.append(c_score)

    return curr_url_info_list


def get_parent_child_info(url, starting_regex, google, depth, top_ten):
    '''
    input:\n
    url - Complete URL, should be a string \n
    starting_regex - Reg expression from where the link should start,a string\n
    google - To indicate whether to fetch from Google, a boolean

    returns:\n
    all_urls - the URLS on the Google Search Result, a list \n
    urls_info - Related info to the URLs, a list
    '''

    def get_google_results(justSoup, starting_regex):
        '''
        '''
        urls_list = []
        urls_info = []

        # select only the Google Search Links
        only_res = justSoup.select('.r a')

        # Iterating over the Google Search Links
        for link in only_res:

            # If the link starts with the specified reg. expression
            if link.get('href').startswith(starting_regex):

                # extract link that starts from http and ends before &sa
                start_log = link.get('href').find('http')
                end_log = link.get('href').find('&sa')

                norm_link = url_normalize(
                    link.get('href')[start_log:end_log])

                # append the link to the only_urls list
                urls_list.append(norm_link)

                # Also append the hyperlink text and the depth
                hyperlink_text = link.text.encode('utf-8').strip()
                # None -> estimated promise
                urls_info.append((hyperlink_text, depth + 1, None))

        return urls_list, urls_info

    def get_other_results(justSoup, depth, cos_par, top_ten, url_par):

        def return_top_10(inner_heap, ranging_len):

            top_urls = []
            top_info = []

            for i in range(0, ranging_len):

                # -est_promise, next(COUNTER), norm_link, csv_all_info, child_depth, cos_par, [url_par]

                top_prom, top_counter, top_link, top_csv_info, top_depth, top_cos_par, top_parents = heapq.heappop(
                    inner_heap)

                if top_link in VISITED_URLS_PRIOR_DICT.keys():
                    VISITED_URLS_PRIOR_DICT[top_link][0] = top_prom
                else:
                    top_urls.append(top_link)
                    top_info.append((top_csv_info, top_depth, top_prom))
                    heap_dict = [top_prom, top_counter, top_csv_info,
                                 top_depth, top_cos_par, top_parents]
                    VISITED_URLS_PRIOR_DICT[top_link] = heap_dict

                # top_link = get_canonical(top_link)

                # if top_link in top_urls or top_link is None:
                #     continue

                # top_urls.append(top_link)
                # top_info.append((decoded_text, top_depth, top_prom))

            return top_urls, top_info

        urls_list = []
        urls_info = []
        heap_list = []
        inner_heap = []

        for link in justSoup.find_all('a'):

            # adding new content
            child_depth = depth

            try:
                norm_link = url_normalize(urljoin(url_par, link.get('href')))
            except Exception as e:
                LOG.error(
                    "Some Encoding Error because URL link is too long: "+str(e))
                LOG.error("Don't worry! Dealing with it!")
                # norm_link = link.get('href')
                continue

            if urlparse(norm_link).netloc == urlparse(url_par).netloc:
                child_depth += 1

                if child_depth > 5:
                    continue

            # Prototype code
            new_cosine = cos_par

            if norm_link in VISITED_URLS_PRIOR_DICT.keys():

                if url_par in VISITED_URLS_PRIOR_DICT[norm_link][-1]:
                    continue

                else:

                    VISITED_URLS_PRIOR_DICT[norm_link][-1].append(url_par)

                    old_cosine = VISITED_URLS_PRIOR_DICT[norm_link][-2]
                    length_cosine = len(VISITED_URLS_PRIOR_DICT[norm_link][-1])

                    new_cosine = old_cosine + \
                        ((cos_par - old_cosine)/length_cosine)
                    VISITED_URLS_PRIOR_DICT[norm_link][-2] = new_cosine

            encoded_text = link.text.encode('utf-8').strip()
            decoded_text = encoded_text.decode('utf-8').strip()

            if cos_par == new_cosine:
                est_promise, csv_all_info = estimate_promise(
                    cos_par, decoded_text, norm_link, ml=RUN_ML)
            else:
                est_promise, csv_all_info = estimate_promise(
                    new_cosine, decoded_text, norm_link, ml=RUN_ML)

            inner_tuple = (-est_promise, next(COUNTER), norm_link,
                           csv_all_info, child_depth, cos_par, [url_par])
            heapq.heappush(inner_heap, inner_tuple)
            urls_list.append(norm_link)
            urls_info.append((decoded_text, child_depth, est_promise))
            heap_list.append([-est_promise, next(COUNTER),
                              csv_all_info, child_depth, cos_par, [url_par]])
            # Prototype code
            # if norm_link in VISITED_URLS_PRIOR_DICT.keys():
            #     VISITED_URLS_PRIOR_DICT[norm_link][0] = -est_promise
            # else:
            #     urls_list.append(norm_link)
            #     urls_info.append((decoded_text, child_depth, est_promise))
            #     heap_list = [-est_promise,
            #                  next(COUNTER), csv_all_info, child_depth, cos_par, [url_par]]
            #     VISITED_URLS_PRIOR_DICT[norm_link] = heap_list

        if top_ten:

            try:
                min_ranging_len = min(len(urls_list), 10)

            except Exception as e:
                min_ranging_len = 0

            return return_top_10(inner_heap, min_ranging_len)

        return urls_list, urls_info

    res, justSoup = get_element_from_url(url)

    current_link_info = get_current_url_info(res, justSoup)
    print("Current Link Info:", current_link_info)

    if res is None or justSoup is None:
        return None, None, current_link_info

    # if the current URL is google
    if google is True:
        child_links, child_info = get_google_results(justSoup, starting_regex)

    else:
        cos_par = current_link_info[-1]
        print("Cos Par:", cos_par)
        child_links, child_info = get_other_results(
            justSoup, depth, cos_par, top_ten, url)

    return child_links, child_info, current_link_info


def get_google_search_urls(query):
    '''
    input:\n
    query - User-Defined query, should be a string

    returns:\n
    all_urls - the URLS on the Google Search Result, a list \n
    urls_info - Related info to the URLs, a list
    '''

    url = url_normalize(BASE_URL+query)

    all_urls, urls_info, par_urls_info = get_parent_child_info(
        url, starting_regex='/url?q=', google=True, depth=0, top_ten=False)

    for sw in STOP_WORDS:

        front, back = 0, len(all_urls) - 1

        while front < back:

            if sw in all_urls[front]:
                del all_urls[front]
                del urls_info[front]
                back -= 1

            else:
                front += 1

    # New Code
    for url, url_info in zip(all_urls, urls_info):

        heap_list = [float('-inf'), next(COUNTER),
                     url_info[0], url_info[1], par_urls_info[2], []]

        VISITED_URLS_PRIOR_DICT[url] = heap_list

    return all_urls, urls_info


def run_focused():

    start_time = datetime.datetime.now()

    counter = itertools.count()

    # seed_info will have (hyperlink text, depth, cosine score)
    get_google_search_urls(QUERY)

    cosine_list = []

    while len(VISITED_URLS_PRIOR_DICT) > 0 and len(VISITED_URLS) < N:

        # Prototype code
        pop_list = VISITED_URLS_PRIOR_DICT[VISITED_URLS_PRIOR_DICT.smallest()]
        p_prom, p_tiebreaker, p_hyperlink, p_depth, p_p_c_score, list_of_ps = pop_list
        print("Before: ", len(VISITED_URLS_PRIOR_DICT))
        p_url = VISITED_URLS_PRIOR_DICT.pop_smallest()
        print("After: ", len(VISITED_URLS_PRIOR_DICT))

        if p_url in VISITED_URLS:
            print("Link already Crawled! Skipping...")
            print("----")
            continue

        print("Current URL:", p_url)
        print("Current URL's Promise & Counter & depth:",
              p_prom, p_tiebreaker, p_depth)
        print("Current URL's Hyperlink related info:", p_hyperlink)

        child_pages, child_info, parent_info = get_parent_child_info(
            p_url, '', False, p_depth, top_ten=True)

        prev_len = len(VISITED_URLS)

        VISITED_URLS.add(p_url)

        new_len = len(VISITED_URLS)

        if new_len > prev_len:
            cosine_list.append(parent_info[-1])

        with open(crawler_to_run+'_'+QUERY+'.txt', 'a+') as log_file:
            log_file.write(
                '------------------------------------------------\n')
            log_file.write('# No: '+str(len(VISITED_URLS))+'\n')
            log_file.write('URL: '+str(p_url)+'\n')
            log_file.write('Time Crawled: '+str(parent_info[0])+'\n')
            log_file.write('Size of the Page: '+str(parent_info[1])+'\n')
            log_file.write('Status Code: '+str(parent_info[2])+'\n')
            log_file.write('Avg Cosine Score of Parents: ' +
                           str(p_p_c_score)+'\n')
            log_file.write('Estimated Promise: '+str(p_prom)+'\n')
            log_file.write('Cosine Relevance Score: '+str(parent_info[3])+'\n')

        # CSV Info
        if p_prom != float('-inf'):
            with open(crawler_to_run+'_wse_training.csv', 'a+') as file:
                file.write(str(p_url))  # writing URL name
                file.write(',')
                # writing average cosine of parents
                file.write(str(p_p_c_score))
                file.write(',')
                file.write(str(p_hyperlink[0]))
                file.write(',')
                file.write(str(p_hyperlink[1]))
                file.write(',')
                file.write(str(p_hyperlink[2]))
                file.write(',')
                file.write(str(p_hyperlink[3]))
                file.write(',')
                file.write(str(p_hyperlink[4]))
                file.write(',')
                file.write(str(p_hyperlink[5]))
                file.write(',')
                # for i in p_hyperlink:  # writing frequency related info
                #     file.write(str(i))
                #     file.write(',')
                file.write(str(parent_info[-1]))  # writing actual cosine
                file.write('\n')

        print("# Links Visited:", len(VISITED_URLS))
        print("----")

        if child_pages is None:
            continue

    end_time = datetime.datetime.now()
    time_elapsed = end_time - start_time
    print("Crawling Finished!")
    harv_score = compute_harvest_score(cosine_list)
    print("Harvest Score:", harv_score)

    print("Debugging starts!!")
    print("Total links in this dict:", len(VISITED_URLS_PRIOR_DICT))

    with open(crawler_to_run+'_'+QUERY+'.txt', 'a+') as log_file:
        log_file.write('\n\n')
        log_file.write('#### Statistics ####'+'\n\n')
        log_file.write('Crawl Start Time: '+str(start_time)+'\n')
        log_file.write('Crawl End Time: '+str(end_time)+'\n')
        log_file.write('Time it took to Crawl: '+str(time_elapsed)+'\n')
        log_file.write('Harvest Score: '+str(harv_score)+'\n')


run_focused()
