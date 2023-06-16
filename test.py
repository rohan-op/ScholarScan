# from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
# llm = OpenAI(temperature=0.9)
llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.9, "max_length":1000, "min_length":300})

# Go through the following mentioned information about a research paper and provide a 100-150 word summary of it :

text = '''
Title: Sentiment analysis: A combined approach
Author: Rudy Prabowo, Mike Thelwall

Sentiment analysis is an important current research area. This paper combines rule-based
classification, supervised learning and machine learning into a newcombinedmethod. This
method is tested on movie reviews, product reviews and MySpace comments. The results
show that a hybrid classification can improve the classification effectiveness in terms of
micro- and macro-averaged F1. F1 is a measure that takes both the precision and recall of
a classifiers effectiveness into account. In addition, we propose a semi-automatic, complementary
approach in which each classifier can contribute to other classifiers to achieve a
good level of effectiveness.

The sentiment found within comments, feedback or critiques provide useful indicators formany different purposes. These sentiments can be categorised either into two categories: positive and negative; or into an n-point scale, e.g., very good, good, satisfactory, bad, very bad. In this respect, a sentiment analysis task can be interpreted as a classification task where each category represents a sentiment. Sentiment analysis provides companies with a means to estimate the extent of product acceptance and to determine strategies to improve product quality. It also facilitates policy makers or politicians to analyse public sentiments with respect to policies, public services or political issues.

This paper presents the empirical results of a comparative study that evaluates the effectiveness of different classifiers, and shows that the use of multiple classifiers in a hybrid manner can improve the effectiveness of sentiment analysis. The procedure is that if one classifier fails to classify a document, the classifier will pass the document onto the next classifier, until the document is classified or no other classifier exists. Section 2 reviews a number of automatic classification techniques used in conjunction with machine learning. Section 3 lists existing work in the area of sentiment analysis. Section 4 explains the different approaches used in our comparative study. Section 5 describes the experimental method used to carry out the comparative study, and reports the results. Section 6 presents the conclusions.

The use of multiple classifiers in a hybrid manner can result in better effectiveness in terms ofmicro- and macro-averaged F1 than any individual classifier. By using a Sentiment Analysis Tool (SAT), we can apply a semi-automatic, complementary approach, i.e., each classifier contributes to other classifiers to achieve a good level of effectiveness. Moreover, a high level of reduction in terms of the number of induced rules can result in a low level of effectiveness in terms of micro- and macroaveraged F1. The induction algorithm can generate a set of induced antecedents that are too sparse for a deeper analysis.
Therefore, in a real-world scenario, it is desirable to have two rule sets: the original set and the induced rule set.
'''

print(llm(text))
