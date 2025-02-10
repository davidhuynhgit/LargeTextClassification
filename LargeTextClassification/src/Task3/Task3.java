package Task3;

import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Task3 { 
	
	public void doFilteredClassification(Instances data, AbstractClassifier classifier){
		try {  
			long start = System.nanoTime();
			// Sets the class index of the dataset 
			data.setClassIndex(1);
			// Create a StringToWordVector filter 
			StringToWordVector swFilter = new StringToWordVector();
			//Specify range of attributes to act on. We want to work on the entire list of words
			swFilter.setAttributeIndices("first-last"); 
			
			// Configure the filter
//			swFilter.setIDFTransform(true);
//			swFilter.setTFTransform(true);			
//			swFilter.setNormalizeDocLength(new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER));
//			swFilter.setOutputWordCounts(true);
			swFilter.setStemmer(new LovinsStemmer());
			swFilter.setStopwordsHandler(new Rainbow());
			swFilter.setTokenizer(new AlphabeticTokenizer());
			swFilter.setWordsToKeep(100);
			
			// Create a FilteredClassifier object
			FilteredClassifier filter_classifier = new FilteredClassifier();
			// Set the filter to the filtered classifier
			filter_classifier.setFilter(swFilter);
					
			// Add the classifier into the filtered classifier
			filter_classifier.setClassifier(classifier);
			filter_classifier.buildClassifier(data);

			// Evaluation
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(filter_classifier, data, 10, new Random(1));
			
			long end = System.nanoTime();
			double elapsed = end - start;
			elapsed = elapsed/ 1000000000;
		    System.out.println("Time taken by " + classifier.getClass().getSimpleName() + ": " + elapsed + " seconds.");
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println("==========================================================================");						   
			}
	         catch(Exception e){
	       	  System.out.println("Error in ...");
	       	  System.out.println(e.getMessage());
	       	  e.printStackTrace();
        }
	}
	
	public static void main(String[] args) throws Exception {
		//Load *. arff dataset 		
		DataSource source = new DataSource("C:\\IFN645\\Capstone\\dataset\\News.arff");
		Instances data = source.getDataSet();
		
		//Initilise task solver
		Task3 task3 =new Task3();
	
		//Perform the classification task using 4 classification algorithms
		task3.doFilteredClassification(data, new IBk());
		task3.doFilteredClassification(data, new SMO());
		task3.doFilteredClassification(data, new J48());
		task3.doFilteredClassification(data, new HoeffdingTree());
	}
}
