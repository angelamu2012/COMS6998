# **COMS6998 Project:** Enhancing Emotion Summarization through the Introduction of Nuanced Emotions and Synthetic Datasets

# Datasets

The cleaned and preprocessed datasets used for fine-tuning and testing are found in the datasets folder. They are specified as command line arguments when running detection_summarization.py.  

**train_anonymized-WITH_POSTS.json**: Original CovidET training dataset, baseline dataset

**test_anonymized-WITH_POSTS.json**: Original CovidET test dataset 

**train_anonymized-WITH_POSTS_additional_emotions.json**: Original CovidET training dataset that includes manual annotations of the new emotions: confusion and resilience. Manual annotations include a labelled emotion and human-written abstractive summaries. 

**test_anonymized-WITH_POSTS_additional_emotions_SUPPLEMENTED.json**: Original CovidET test dataset that includes manual annotations of the new emotions: confusion and resilience. Manual annotations include a labelled emotion and human-written abstractive summaries. 

**BT_combined_w_orig_final.json**: Backtranslations of CovidET, combined with the CovidET training dataset

**gpt_combined_with_original_additional_emotions.json**: ChatGPT-generated data, combined with the CovidET training dataset

**scraped_combined_with_original.json**: Reddit-scraped data, combined with the CovidET training dataset

**all_additional_emotions_train.json**: Backtranslations of CovidET, ChatGPT-generated data, and Reddit-scraped data, all combined with the CovidET training dataset


# Jupyter Notebooks

## Data Gathering, Processing, Visualization 
Information regarding data downloading, generation, data preprocessing is described alongside the corresponding notebook. 

**Back_Translation.ipynb**: Performs Spanish backtranslation on the Reddit Posts from CovidET using MarianMT. Used to generate the backtranslation dataset. Results can be found in the datasets folder. 

**Reddit_Scrape.ipynb**: Scrapes Reddit Posts and cleans data. Results can be found in the datasets folder. Scraping methods include using PRAW and Pullpush. Preprocessing includes replacing certain entities with the corresponding tag, as well as replacing urls, empty space, etc. 

**CSV_Data_To_JSON_Format.ipynb**: Converts gathered data into the required JSON format to be consistent with the CovidET dataset format. 

**Data_Stats_and_Heatmap.ipynb**: Gathers statistics like dataset size, length, and generates emotion co-occurence heatmap. 

**lda.ipynb**: Performs Latent Dirichlet allocation to identify topics within abstractive summaries for each dataset. 


## Metrics

**metrics.ipynb**: Finds BERTScore, SummaCConv score, BLEU, self-BLEU on generated summaries or on Reddit Posts. The calculation of these metrics is done separately and outside of detection_summarization due to memory limits and so that it can be run on a CPU instead of GPU. 

# Model Training and Evaluation
**Baseline and Experiments** 
The baseline used for comparison is the original CovidET dataset, found in the datasets folder. These files are specified as command line arguments to detection_summarization.py. To run the other experiments, specify the relevant datasets. 

**Model** 
The model used for emotion summarization is a joint model proposed by [CovidET](https://github.com/honglizhan/CovidET/tree/main)
The script is named detection_summarization.py. To run the script, the emotion, training dataset, test dataset, learning rate, and a suffix string for the output file must be specified. 

Modifications include calculating ROUGE 1 and 2 scores instead of just ROUGE-L and refactoring relevant methods (full_eval(), ev_once(), evaluate_summaries()) to do so, writing intermediate outputs (detection results, generated summaries) to files, removing BERTScore calculations and calculating them separately due to memory limits, adding comments to code

```
$ TOKENIZERS_PARALLELISM=false python detection_summarization.py \
	--emotion <emotion> \
	--training_path <...> \
	--validation_path <...> \
	--test_path <...> \
	--results_detection_summarization <filename> \
	--learning_rate <...>
```

**Evaluation**
Running detection_summarization.py creates 3 text files consisting of the emotion detection prediction results (detection_output...txt) and generated summaries. 2 files are generated for generated summaries, one which is formatted such that it's easy to compare the generated summary with the target (summaries_output...txt), annotated summary, and one which is used in metrics.ipynb to compute the BERTScore (summaries_output_for_bert...txt). It also creates a JSON file consisting of F-scores and ROUGE scores. A placeholder value is generated for the BERTScore in the JSON file. The corresponding generated summaries text file is fed into metrics.ipynb to calculate the BERTScore and replace the placeholder value. 

