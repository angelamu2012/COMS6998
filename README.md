# COMS6998 Project

**Datasets** 

The cleaned and preprocessed datasets used for fine-tuning and testing are found in the datasets folder. They are specified as command line arguments when running detection_summarization.py. 

**Jupyter Notebooks**

*Data Gathering, Processing, Visualization* 

Back_Translation.ipynb: Performs backtranslation on the Reddit Posts from CovidET. Used to generate the backtranslation dataset. Results can be found in the datasets folder 

Reddit_Scrape.ipynb: Scrapes Reddit Posts and cleans data. Results can be found in the datasets folder 
CSV_Data_To_JSON_Format.ipynb: Converts gathered data into the required JSON format to be consistent with the CovidET dataset format. 

Data_Stats_and_Heatmap.ipynb: Gathers statistics like dataset size, length, and generates emotion co-occurence heatmap. 

lda.ipynb: Performs Latent Dirichlet allocation to identify topics within abstractive summaries for each dataset. 

*Metrics*

metrics.ipynb: Finds BERTScore, SummaCConv score, BLEU, self-BLEU on generated summaries or on Reddit Posts. 
