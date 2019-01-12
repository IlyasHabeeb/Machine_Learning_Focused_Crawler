# Machine_Learning_Focused_Crawler
A focused web crawler that uses Machine Learning to fetch better relevant results.

The list of files are as follows:

<b>1. Crawler_ML.py:</b> This is the python crawler. It runs as follows:

<b><i>python Crawler_ML.py withoutML</b></i> - To run Focused Crawler without Machine Learning <br />
<b><i>python Crawler_ML.py withML</b></i> - To run Focused Crawler with Machine Learning

After executing the above command, the program asks for the following input:

<b><i>Please Enter the Query in small letters (Words Should be Spaced):</b></i> election results <br />
<b><i>Please Enter the Number of Pages to Crawl:</b></i> 1000

Currently, the crawler supports queries with only the following words:

'wildfires', 'california', 'brooklyn', 'dodgers', 'shahrukh', 'khan', 'pangolin', 'armadillo', 'world', 'cup','hurricane', 'florence', 'mac', 'miller', 'kate', 'spade', 'anthony', 'bourdain', 'black', 'panther', 'mega', 'million', 'results', 'stan', 'lee', 'demi','lovato', 'election'

<b>2. withoutML_election results.txt</b> - This is the log file query 'election results' for the Focused Crawler without ML for large topic query

<b>3. withML_election results.txt</b> - This is the log file query 'election results' for the Focused Crawler with ML for large topic query

<b>4. withoutML_brooklyn dodgers.txt</b> - This is the log file query 'brooklyn dodgers' for the Focused Crawler without ML for rare topic query

<b>5. withML_brooklyn dodgers.txt</b> - This is the log file query 'brooklyn dodgers' for the Focused Crawler with ML for rare topic query

Note: 2, 3, 4, 5 outputs the following:

i) Name of the URL

ii) Time the URL was Crawled

iii) Size of the Page

iv) Status Code

v) HyperLink Text Info: (text, depth of the link)

vi) Estimated Promise (only for focused crawler)

vii) Cosine Relevance Score

viii) Statistics of the Entire Crawl that includes:

	a) Crawl Start Time
	
	b) Crawl End Time
	
	c) Time it took to Crawl: hh:mm:ss
	
	d) Harvest Score 

<b>6. Project Report</b> - A pdf file that describes the Project in detail.
