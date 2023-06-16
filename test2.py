# from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
# llm = OpenAI(temperature=0.9)
llm = HuggingFaceHub(repo_id="gpt2-xl", model_kwargs={"temperature": 0.9, "max_length":500, "min_length":500})

# Go through the following mentioned information about a research paper and provide a 100-150 word summary of it :

text = '''
Write Introduction of an academic review paper based on the information about different research papers mentioned below:

Research Paper1:
The sentiment found within comments, feedback or critiques provide useful indicators formany different purposes. These sentiments can be categorised either into two categories: positive and negative; or into an n-point scale, e.g., very good, good, satisfactory, bad, very bad. Sentiment analysis provides companies with a means to estimate the extent of product acceptance and to determine strategies to improve product quality. It also facilitates policy makers or politicians to analyse public sentiments with respect to policies, public services or political issues. The use of multiple classifiers in a hybrid manner can result in better effectiveness in terms ofmicro- and macro-averaged F1 than any individual classifier. By using a Sentiment Analysis Tool (SAT), we can apply a semi-automatic, complementary approach, i.e., each classifier contributes to other classifiers to achieve a good level of effectiveness. The induction algorithm can generate a set of induced antecedents that are too sparse for a deeper analysis. A high level of reduction in terms. of the number of induced rules can result. in a low levels of effectiveness in Terms of micro- and Macroaveraged. F1. The method is tested on movie reviews, product reviews and MySpace comments. The results show that a hybrid classification can improve the classification effectiveness in. terms of precision and recall of the classifiers. It is desirable to have two rule sets: the original set and the induced rule set. This paper presents the empirical results of a comparative study that evaluates the effectiveness of different classifiers, and shows that the use of several classifiers can improve sentiment analysis.
'''

print(llm(text))
