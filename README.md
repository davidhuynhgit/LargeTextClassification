# Large Scale Data Mining - News Documents Classification  📰🔍
The provided dataset `News.arff` is a text dataset consisting of 14,018 news documents. These news documents are categorised into four classes: computer, politics, science, and sports. 
In this task, the analyst is required first to classify the news documents using Weka  to determine 
some parameters in the filter, then develop a Java program to build a classifier to classify the news in this dataset.   
 
## 1.  Attribute selection in [Weka](https://ml.cms.waikato.ac.nz/weka/) 🐤
   
In this part, I used a filter in Weka to select attributes from the documents. I chose 100 attributes and used J48 classifier to do the classification.

For the parameters in the filter, I used their default values as the foundational settings, then adjusted several values to achieve better results. 
I was required to tune 4 or 5 parameters that I thought they were important for determining the 
attributes, then:
- List out the parameters in the filter to be tuned and the chosen values for these parameters. 
- Briefly describe the working process in Weka to determine the values for the parameters in the filter with evidence
  
## 2. Java program 💻
For this part, I was required to develop a Java program to classify the documents in the news dataset. There were a set of tasks to be completed:   
1)  Perform the classification task using 4 classification algorithms, IBk, SMO, J48, and the method HoeffdingTree in Weka, and use a filter with the parameter settings determined in question 1 of this task.   
2)  The program should display the correctly classified instances results, accuracy, and the time taken by each algorithm.   
3)  Which classifier performs the best in terms of time efficiency? Describe why this algorithm is faster than others.  

## 3. Report 📊
For this task, we used the FilteredClassifier in Weka to classify the text data in the News.arff file. The internal classifier was set as J48. The internal filter was set as StringToWordVector. We then only tuned the parameter within the filter while keeping the rest of FilteredClassifier unchanged. Here’s a step-by-step explanation of the attribute selection process and parameter tuning:

## 3.1. Parameters Tuned in the Filter
I selected the following five parameters to tune for optimizing the classification:

---
| Setting | Accuracy (%) | Time (s) | Observations |
|----------|----------|----------|----------|
| 1. Default <br> Stemmer = LovinsStemmer <br>stopwordsHandler = Rainbow <br> WordsToKeep =  100   | 76.8369  | 32.15  | Baseline configuration with moderate accuracy.|
| 2. Stemmer = LovinsStemmer <br> stopwordsHandler = Rainbow <br> WordsToKeep =  100 <br> Tokenizer = AlphabeticTokenizer  | 78.2851   | 33.84  | Improved accuracy due to better tokenization.   |
| 3. Stemmer = LovinsStemmer <br> stopwordsHandler = Rainbow <br> WordsToKeep =  100 <br> Tokenizer = AlphabeticTokenizer <br> normalizeDocLenght = Normalize all data <br> OutputWordCounts = true   | 77.4932   | 36.24   | Slightly lower accuracy but balanced performance.  |
| 4. Stemmer = LovinsStemmer <br> stopwordsHandler = Rainbow <br> WordsToKeep =  100 <br> Tokenizer = AlphabeticTokenizer <br> normalizeDocLenght = Normalize all data <br> OutputWordCounts = true <br> DoNotOperateOnPerClassBasis = true   | 63.9464  | 14.18   |  Fastest but with significantly reduced accuracy.   |
| 5. Stemmer = LovinsStemmer <br> stopwordsHandler = Rainbow <br> WordsToKeep =  100 <br> Tokenizer = AlphabeticTokenizer <br> normalizeDocLenght = Normalize all data <br> OutputWordCounts = true <br> IDF = true <br> TF = true <br> DoNotOperateOnPerClassBasis = false   | 77.5646   | 38.52   | TF-IDF transformations improved accuracy, longer processing time.  |
---

**Key Observations:**
- **Setting 2** had the highest accuracy, showing the benefits of using AlphabeticTokenizer.
- **Setting 4** was the quickest, but with much lower accuracy, indicating a trade-off between time and classification performance.
- **Setting 3** and **Setting 5** offered balanced performance between accuracy and time, making them suitable for cases where both factors are important.

## 3.2. Working Process in Weka
1. Load the Dataset: I first loaded the News.arff dataset into Weka.
2. Configure the Filter:
Then I go to the classify tab and choose the FilteredClassifier from meta. I click on the filter to change it to StringToWordVector, then click on the box to adjust the settings as above. 
3. Classification Using J48:
Chose the J48 classifier to classify the documents based on the filtered attributes.
Performed a 10-fold cross-validation to evaluate the performance.
4. Results and Evaluation:
Compared the classification accuracy, error rates, and model-building time across different filter configurations.


## 3.3 Java programming
In this section, I used the same parameter settings from question 1 final result to perform the classification task with 4 approaches. The performance metrics are listed in the following table. The time taken for each algorithm varies by each run but their ranks remain unchanged. 

| Algorithm       | Correctly Classified Instances | Accuracy    | Time taken (seconds) (First run) |
|---------------|--------------------------------|------------|---------------------------------|
| IBk           | 10193                          | 72.7137 %  | 88.990155899                      |
| SMO           | 11693                          | 83.4142 %  | 1127.4793354                      |
| J48           | 10974                          | 78.2851 %  | 345.014511299                      |
| HoeffdingTree | 10307                          | 73.5269 %  | 59.5451182                       |

The program has been implemented using FilteredClassifier as a foundation model to apply other classification algorithms. The correctness of our implementation can be clarified by comparing the J48’s results with the outputs from Weka in question 1. The figure below demonstrates one of the output generated by my Java program.

## 3.4 Results and justification

Although its accuracy is not the best, HoeffdingTree performs the best in terms of
time efficiency. With the 10-fold cross-validation and batch processing, the time
taken for each batch will determine the total time efficiency of that algorithm.

It is obvious that the Hoeffding Tree is very effective in processing data streams, but
the estimation with Hoeffding bound allows it to make decisions with a smaller
sample of the data, rather than requiring the entire batch before making each split,
resulting in a quicker tree model. For each batch, the Hoeffding tree is built
incrementally and the classification for some class can be done on the go without
revisiting the whole dataset.

The other algorithms, which are IBk, SMO, J48, need to access all data points,
consequently leading to multiple visits of instances. The bigger the dataset is, the
higher requirements for time and storage are needed to proceed. For some
classifiers, such as SMO or J48, batch processing may retain data or values from
previous batches to make decisions in finding optimal solutions, which explains the
cost of time efficiency.
