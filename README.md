# **COMS6998 Project**

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

***Data Gathering, Processing, Visualization***

**Back_Translation.ipynb**: Performs backtranslation on the Reddit Posts from CovidET. Used to generate the backtranslation dataset. Results can be found in the datasets folder 

**Reddit_Scrape.ipynb**: Scrapes Reddit Posts and cleans data. Results can be found in the datasets folder 

**CSV_Data_To_JSON_Format.ipynb**: Converts gathered data into the required JSON format to be consistent with the CovidET dataset format. 

**Data_Stats_and_Heatmap.ipynb**: Gathers statistics like dataset size, length, and generates emotion co-occurence heatmap. 

**lda.ipynb**: Performs Latent Dirichlet allocation to identify topics within abstractive summaries for each dataset. 


***Metrics***

**metrics.ipynb**: Finds BERTScore, SummaCConv score, BLEU, self-BLEU on generated summaries or on Reddit Posts. The calculation of these metrics is done separately and outside of detection_summarization due to memory limits and so that it can be run on a CPU instead of GPU. 

# Model Evaluation
Running detection_summarization.py creates text files consisting of the emotion detection results and generated summaries. 2 files are generated for generated summaries, one which is formatted such that it's easy to compare the generated summary with the target, annotated summary, and one which is used to compute the BERTScore. It also creates JSON files consisting of F-scores and ROUGE scores. A placeholder value is generated for the BERTScore in the JSON file. The corresponding generated summaries text file is fed into metrics.ipynb to calculate the BERTScore and replace the placeholder value. 
