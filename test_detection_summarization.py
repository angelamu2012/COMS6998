#START: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
from bert_score import score as bert_scr
import pandas as pd
from absl import app
from absl import flags

from datasets import load_metric
from sklearn.metrics import f1_score

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Optional

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput, Seq2SeqModelOutput
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("emotion", "confusion", "Emotion type")
flags.DEFINE_string("training_path",
                    "datasets/all_additional_emotions_train.json",
                    "Path to training json")
flags.DEFINE_string("validation_path",
                    "datasets/val_anonymized-WITH_POSTS_additional_emotions.json",
                    "Path to validation json")
flags.DEFINE_string("test_path",
                    "datasets/test_anonymized-WITH_POSTS_additional_emotions_SUPPLEMENTED.json",
                    "Path to test json")
flags.DEFINE_string("results_detection_summarization",
                    "detection_summarization_prime", "")
flags.DEFINE_integer("gradient_accumulation_steps", 8, "")
flags.DEFINE_float("learning_rate", 0.00003, "")

#END: COPIED FROM https://github.com/honglizhan/CovidET/tree/main

flags.DEFINE_string("model", "echarlaix/bart-base-cnn-r2-19.4-d35-hybrid",
                    "Model name from HuggingFace")
flags.DEFINE_integer("batch_size", 4, "")

#START: COPIED FROM https://github.com/honglizhan/CovidET/tree/main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rouge_metric = load_metric("rouge")


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(p=0.1) #apply regularization to prevent overfitting
        self.out_proj = nn.Linear(256, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int,
                       decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class DetectionSummarizationModel(BartForConditionalGeneration):
  # inherits from BartForConditionalGeneration, used for detection and summarization

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg = self.model.config
        self.classification_head = BartClassificationHead(
            cfg.d_model,
            1, #binary classification
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # 0 -> Emotion Detection | 1 -> Summarization | 2 -> Emotion Detection + Summarization
        loss_calculator_type=1,
        positive=True,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        if encoder_outputs is None:
            # Make sure we make a single forward pass through the encoder
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.model.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if summarization or detection & summarization
        if loss_calculator_type != 0:
            if labels is not None: #if ground truth labels are provided
                if decoder_input_ids is None and decoder_inputs_embeds is None:
                    #ensure input sequence for the decoder is appropriately aligned
                    decoder_input_ids = shift_tokens_right(
                        labels, self.model.config.pad_token_id,
                        self.model.config.decoder_start_token_id)

            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions)

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            lm_logits = self.lm_head(outputs[0])
            # apply bias term to logits
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                #flatten/reshape logits and labels, calculate loss b/t labels and logits
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.model.config.vocab_size),
                    labels.view(-1))

            if not return_dict:
                output = (lm_logits, ) + outputs[1:]
                return ((masked_lm_loss, ) +
                        output) if masked_lm_loss is not None else output
            # output to be returned
            output_lm_head = Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        #if detection or Summarization & detection 
        if loss_calculator_type != 1:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id,
                self.config.decoder_start_token_id)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions)

            hidden_states = outputs[0]
            #create binary end of sentence mask, each elt is true if corresponding elt in input_ids = EOS token ID 
            eos_mask = input_ids.eq(self.config.eos_token_id)
            sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
            logits = self.classification_head(sentence_representation)

            if positive == True:
                #create tensor of 1s
                classification_labels = torch.ones(logits.shape).to(device)
            else:
                #create tensors of 0s
                classification_labels = torch.zeros(logits.shape).to(device)
            classification_loss_fn = BCEWithLogitsLoss()
            #compute loss b/t logits and labels
            classification_loss = classification_loss_fn(
                logits, classification_labels)
          
            #output to be returned
            output_classification_head = Seq2SeqSequenceClassifierOutput(
                loss=classification_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        if loss_calculator_type == 0: #detection only 
            return output_classification_head
        elif loss_calculator_type == 1: #summarization only 
            return output_lm_head
        else:   #detection and summarization
            return (output_lm_head, output_classification_head)


def load_split_to_dataset(filepath, emotion):
    with open(filepath) as f:
        dataset = json.load(f)

    positive_posts = []
    negative_posts = []
    summarizations = []

    for k in dataset:
        emo = 0
        summ = []
        for annotation in dataset[k]['Annotations']:
            for hit in dataset[k]['Annotations'][annotation]:
                if hit['Emotion'] == emotion:
                    emo = 1
                    summ.append(hit['Abstractive'])

        if emo == 0:
            negative_posts.append(dataset[k]['Reddit Post'])
        else:
            for i in range(len(summ)):
                positive_posts.append(dataset[k]['Reddit Post']) #add post 2x if has 2 summaries
                summarizations.append(summ[i])

    return positive_posts, negative_posts, summarizations


def load_test_split_to_dataset(filepath, emotion):

    with open(filepath) as f:
        dataset = json.load(f)

    positive_posts = []
    negative_posts = []
    summarizations = []

    for k in dataset:
        emo = 0
        summ = []
        for annotation in dataset[k]['Annotations']:
            for hit in dataset[k]['Annotations'][annotation]:
                if hit['Emotion'] == emotion:
                    emo = 1
                    summ.append(hit['Abstractive'])
        if emo == 0:
            negative_posts.append(dataset[k]['Reddit Post'])
        else:
            positive_posts.append(dataset[k]['Reddit Post'])
            summarizations.append(summ)

    return positive_posts, negative_posts, summarizations


def convert_positive_examples_to_features(example_batch):
    #tokenize posts to retrieve input encodings
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                truncation=True,
                                padding='max_length')
    #tokenize summaries to retrive target encodings
    target_encodings = tokenizer(text_target=example_batch['Summaries'],
                                 max_length=128,
                                 truncation=True)

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }


def convert_positive_test_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                truncation=True,
                                padding='max_length')

    for i in range(len(example_batch['Summary2'])):
        if example_batch['Summary2'][i] == None:
            example_batch['Summary2'][i] = ''

    target_summaries1 = tokenizer(text_target=example_batch['Summary1'],
                                  max_length=128,
                                  truncation=True,
                                  padding='max_length')
    target_summaries2 = tokenizer(text_target=example_batch['Summary2'],
                                  max_length=128,
                                  truncation=True,
                                  padding='max_length')

    d = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "summary1": target_summaries1["input_ids"],
        "summary2": target_summaries2["input_ids"]
    }
    return d


def convert_negative_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                padding='max_length',
                                truncation=True)

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"]
    }


def infer(dataloader, filler, model):
    full_predictions = []
    true_labels = []

    # don't track gradients bc not needed for prediction
    with torch.no_grad():
        #iterate over dataloader
        for elem in tqdm(dataloader):
            cuda_tensors = {
                key: elem[key].to(device)
                for key in elem if key in ['attention_mask', 'input_ids']
            }
            classification_head = model(
                cuda_tensors["input_ids"],
                loss_calculator_type=0, #loss calculation is for detection 
                attention_mask=cuda_tensors['attention_mask'])
            
            #if classification_head.logits > 0, result is 1, else 0
            results = torch.squeeze(torch.where(classification_head.logits > 0,
                                                1, 0),
                                    dim=1).cpu().numpy()
            
            #append prediction results to full_predictions
            full_predictions = full_predictions + results.tolist()
            #since infer called on either all pos or all neg posts, true labels are the filler value
            true_labels = true_labels + [filler] * len(results)
            assert (len(full_predictions) == len(true_labels))
    #END: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
    with open('detection_output' + FLAGS.results_detection_summarization + '_' + FLAGS.emotion + '.txt','a') as file:
        for i, item in enumerate(full_predictions):
            file.write(str(i) + ", our: "+ str(item)+"\n")
            file.write(str(i) + ", target: " + str(true_labels[i]) + "\n")
    #START: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
    return full_predictions, true_labels

def evaluate_detection(model, test_dataloader_positives,
                       test_dataloader_negatives):
    full_predictions = []
    true_labels = []

    #set model in evaluation mode, so inference can be performed such that the model's behavior is consistent w/ how it was trained
    #eval mode deactives training-specific features like dropout, freezes learnable params, etc
    model.eval()

    print("calling infer")
    #retrieve predictions and true labels 
    prediction_positives, true_labels_positives = infer(
        test_dataloader_positives, 1, model)
    prediction_negatives, true_labels_negatives = infer(
        test_dataloader_negatives, 0, model)
    
    #concactenate pos and neg data
    full_predictions = prediction_positives + prediction_negatives
    true_labels = true_labels_positives + true_labels_negatives

    #return model to training mode
    model.train()

    print("computing f1 score in evaluate_detection")
    #compute f1 score using true labels and predictions
    return f1_score(true_labels, full_predictions, average=None)


def evaluate_summaries(dataset,
                       metric,
                       model,
                       tokenizer,
                       batch_size=8,
                       device=device,
                       column_text="Posts"):
    
    #dataset is df_positive, create a list of posts of batch-sized chunks
    article_batches = list(chunks(dataset[column_text], batch_size))
    #if post has 2 summaries, 2nd summary belongs to s2/'Summary2', otherwise s2 is '' empty str 
    target_batches_s1 = list(chunks(dataset['Summary1'], batch_size))
    target_batches_s2 = list(chunks(dataset['Summary2'], batch_size))
    zped = list(zip(target_batches_s1, target_batches_s2))
    fin_zip = []
    for elem in zped:
        inner_zip = list(zip(elem[0], elem[1]))
        for i in range(len(inner_zip)):
            if inner_zip[i][1] == '': #if no 2nd summary
                inner_zip[i] = [inner_zip[i][0]]
            else:
                inner_zip[i] = list(inner_zip[i])
        fin_zip.append(inner_zip)

    our_summaries = []
    target_summaries = []
    q = False

    # for post, summary
    for article_batch, target_batch in tqdm(zip(article_batches, fin_zip),
                                            total=len(article_batches)):
        #tokenize post, return dictionary of input_ids, attention_mask, etc
        #return tensors are type pytorch tensors
        inputs = tokenizer(article_batch,
                           truncation=True,
                           padding="max_length",
                           return_tensors="pt")
        #generate summary from encoded input
        #encourage shorter summaries using 0.8 for penalty
        #consider 8 different sequences during beam search and select seq w/ highest probability
        #generate at most 128 tokens for summary  
        #attention mask ensures model focuses on relevant parts of the input
        summaries = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            length_penalty=0.8,
            num_beams=8,
            max_length=128)
        
        #decode input_ids and convert to text 
        decoded_summaries = [
            tokenizer.decode(s,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for s in summaries
        ]
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

        our_summaries += decoded_summaries
        target_summaries += target_batch
        
        #add batches of predictions and references
        metric.add_batch(predictions=decoded_summaries,
                         references=target_batch)
    #perform BERTScore calculation separately 
    '''
    print("calculating bert score in evalute_summaries")
    _, _, F1 = bert_scr(our_summaries,
                        target_summaries,
                        model_type='microsoft/deberta-xlarge-mnli',
                        lang='en',
                        verbose=True)
    '''
    print("writing summaries to file in evaluate_summaries")

    new_score = metric.compute()

    #END: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
    with open('summaries_output' + FLAGS.results_detection_summarization + '_' + FLAGS.emotion + '.txt','a') as file:
        for i, item in enumerate(our_summaries):
            file.write(str(i) + ", our: "+ item+"\n")
            for add_item in target_summaries[i]:
                file.write(str(i) + ", target: " + add_item + "\n")

    with open('summaries_output_for_bert' + FLAGS.results_detection_summarization + '_' + FLAGS.emotion + '.txt','a') as file:
        json.dump(our_summaries, file) 
        file.write("\n")
        json.dump(target_summaries, file)
        file.write("\n")

    print("done writing summaries, returning from evaluate_summaries")
    return new_score['rougeL'].mid.fmeasure, new_score['rouge1'].mid.fmeasure, new_score['rouge2'].mid.fmeasure, -1 #F1.mean() #return rouge-L and Bert scores, for each post's f1 score (bertscore), take the average 

#START: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i + batch_size]


def prepare_dataset(positive_posts, negative_posts, summarizations):
    #convert list of posts/summaries into df
    df_positive = pd.DataFrame(list(zip(positive_posts, summarizations)),
                               columns=['Posts', 'Summaries'])
    df_negative = pd.DataFrame(negative_posts, columns=['Posts'])
    #convert into dataset
    df_positive = Dataset.from_pandas(df_positive)
    df_negative = Dataset.from_pandas(df_negative)
    #use datasets map method to apply a preprocessing fxn to dataset, returns input_ids, attention_mask, labels
    df_positive = df_positive.map(convert_positive_examples_to_features,
                                  batched=True)

    df_positive.set_format(type="torch",
                           columns=["input_ids", "labels", "attention_mask"])
    
    #use datasets map method to apply a preprocessing fxn to dataset, returns input_ids, attention_mask
    df_negative = df_negative.map(convert_negative_examples_to_features,
                                  batched=True)

    df_negative.set_format(type="torch",
                           columns=["input_ids", "attention_mask"])

    return df_positive, df_negative


def prepare_test_dataset(positive_posts, negative_posts, summarizations):
    s1, s2 = [], []
    #retrieve summaries
    for elem in summarizations:
        if len(elem) == 2:
            s1.append(elem[0])
            s2.append(elem[1])
        else:
            s1.append(elem[0])
            s2.append(None) #only 1 annotator labelled target emotion 
    
    #convert list of posts/summaries into df        
    df_positive = pd.DataFrame(list(zip(positive_posts, s1, s2)),
                               columns=['Posts', 'Summary1', 'Summary2'])
    df_negative = pd.DataFrame(negative_posts, columns=['Posts'])
    df_positive = Dataset.from_pandas(df_positive)
    
    df_negative = Dataset.from_pandas(df_negative)
    
    #use datasets map method to apply a preprocessing fxn to dataset, returns input_ids, attention_mask, target summary input_ids
    df_positive = df_positive.map(convert_positive_test_examples_to_features,
                                  batched=True)

    df_positive.set_format(
        type="torch",
        columns=["input_ids", "summary1", "summary2", "attention_mask"])
    #use datasets map method to apply a preprocessing fxn to dataset, returns input_ids, attention_mask
    df_negative = df_negative.map(convert_negative_examples_to_features,
                                  batched=False)

    df_negative.set_format(type="torch",
                           columns=["input_ids", "attention_mask"])
    return df_positive, df_negative


def return_dataloaders(model, df_positive, df_negative):
    #combines individual examples into batches 
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                   model=model,
                                                   padding='longest')
    #dataloader allows for iterating over batches of data
    train_positives = torch.utils.data.DataLoader(
        df_positive,
        batch_size=FLAGS.batch_size,
        collate_fn=seq2seq_data_collator,
        num_workers=2,
        shuffle=True)
    train_negatives = torch.utils.data.DataLoader(
        df_negative,
        batch_size=FLAGS.batch_size,
        collate_fn=seq2seq_data_collator,
        num_workers=2,
        shuffle=True)
    return train_positives, train_negatives


def return_test_dataloaders(df_positive, df_negative):
    train_positives = torch.utils.data.DataLoader(df_positive,
                                                  batch_size=FLAGS.batch_size,
                                                  num_workers=2,
                                                  shuffle=False)
    train_negatives = torch.utils.data.DataLoader(df_negative,
                                                  batch_size=FLAGS.batch_size,
                                                  num_workers=2,
                                                  shuffle=False)
    return train_positives, train_negatives


def ev_once(model, dataloader_positives, dataloader_negatives, df_positive):
    model.eval()
    #find f1 score for pos and neg datsets
    print("run evaluate_detection")
    f1 = evaluate_detection(model, dataloader_positives, dataloader_negatives)

    print("run evaluate_summaries")
    #evaluate summaries for positive datset
    rL, r1, r2, bert_score = evaluate_summaries(df_positive,
                                        rouge_metric,
                                        model,
                                        tokenizer,
                                        batch_size=8,
                                        device=device)
    print("returning from evaluate_summaries, calling model.train()")
    model.train()

    return f1, rL, r1, r2, bert_score


def full_eval(model, test_dataloader_positives, test_dataloader_negatives,
              df_test_positive, dev_dataloader_positives,
              dev_dataloader_negatives, df_validation_positive, results,
              epoch):

    print("starting to write epoch to file - full_eval")
    #positive df used for summarization, pos and neg dataloaders used for emotion detection
    with open('summaries_output' + FLAGS.results_detection_summarization + '_' + FLAGS.emotion + '.txt','a') as file:
        file.write("\n" + str(epoch) + '_test'+"\n")
    
    print("running ev_once on test dataset")       
    f1, rL, r1, r2, bert_score = ev_once(model, test_dataloader_positives,
                                 test_dataloader_negatives, df_test_positive)
    print("returning from ev_once on test")
    results[str(epoch) + '_test'] = (rL, r1, r2, float(bert_score), str(f1)) #float(bert_score.numpy())


    with open('summaries_output' + FLAGS.results_detection_summarization + '_' + FLAGS.emotion + '.txt','a') as file:
        file.write("\n" + str(epoch) + '_validation'+"\n")

    print("running ev_once on validate dataset")
    f1, rL, r1, r2, bert_score = ev_once(model, dev_dataloader_positives,
                                 dev_dataloader_negatives,
                                 df_validation_positive)
    print("returning from ev_once on validate")
    results[str(epoch) + '_validation'] = (rL, r1, r2, float(bert_score),
                                           str(f1)) #float(bert_score.numpy())
    with open(
            FLAGS.results_detection_summarization + '_' + FLAGS.emotion +
            '.json', 'w') as f:
        json.dump(results, f)
    print("returning from full_eval")
    return f1, rL, r1, r2, bert_score


def main(argv):

    global tokenizer

    print("loading datasets")

    #load datasets and split reddit posts into pos/neg, obtain summaries 
    train_positive_posts, train_negative_posts, train_summarizations = load_split_to_dataset(
        FLAGS.training_path, FLAGS.emotion)
    dev_positive_posts, dev_negative_posts, dev_summarizations = load_test_split_to_dataset(
        FLAGS.validation_path, FLAGS.emotion)
    test_positive_posts, test_negative_posts, test_summarizations = load_test_split_to_dataset(
        FLAGS.test_path, FLAGS.emotion)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)

    print("calling prepare_dataset")

    #convert dataset to features -- tokenize, get attention masks, input_ids 
    df_train_positive, df_train_negative = prepare_dataset(
        train_positive_posts, train_negative_posts, train_summarizations)
    df_validation_positive, df_validation_negative = prepare_test_dataset(
        dev_positive_posts, dev_negative_posts, dev_summarizations)
    df_test_positive, df_test_negative = prepare_test_dataset(
        test_positive_posts, test_negative_posts, test_summarizations)

    #load BART model
    print("starting to load model")
    model = DetectionSummarizationModel.from_pretrained(FLAGS.model)
    model.to(device)
    print("finished loading model")


    print("calling return_dataloaders")

    #divide data into batches and makes it iterable, allows data to be loaded in parallel w/ multiple workers, etc
    train_dataloader_positives, train_dataloader_negatives = return_dataloaders(
        model, df_train_positive, df_train_negative)
    dev_dataloader_positives, dev_dataloader_negatives = return_test_dataloaders(
        df_validation_positive, df_validation_negative)
    test_dataloader_positives, test_dataloader_negatives = return_test_dataloaders(
        df_test_positive, df_test_negative)

    #use adam optimizer to adjust weights
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    print("starting w/ epoch 0")

    results = {}
    #epoch 0 evaluation
    full_eval(model, test_dataloader_positives, test_dataloader_negatives,
              df_test_positive, dev_dataloader_positives,
              dev_dataloader_negatives, df_validation_positive, results, 0)
    #epochs 1-10
    for epoch in range(10):
        print("epoch: ", epoch+1)
        ctr = 1
        for data_positive, data_negative in tqdm(
                zip(train_dataloader_positives, train_dataloader_negatives)):
            cuda_tensors_negatives = {
                key: data_negative[key].to(device)
                for key in data_negative
            }
            cuda_tensors_positives = {
                key: data_positive[key].to(device)
                for key in data_positive
            }
            print("classification head")
            lm_head, classification_head = model(
                cuda_tensors_positives['input_ids'],
                loss_calculator_type=2, #detection and summarization 
                positive=True,
                attention_mask=cuda_tensors_positives['attention_mask'],
                labels=cuda_tensors_positives['labels']) #labels are summary input_ids
            print("classification negatives")
            classification_negatives = model(
                cuda_tensors_negatives['input_ids'],
                loss_calculator_type=0, #detection only 
                positive=False,
                attention_mask=cuda_tensors_negatives['attention_mask'],
                labels=None) #summary input_ids not needed for negative posts
            
            print("computing loss")
            loss = 0.25 * classification_head.loss + 0.5 * lm_head.loss
            loss += 0.25 * classification_negatives.loss
            # computes gradients of model parameters w/ respect to loss, initiate backpropagation
            loss.backward()

            print("calling step and zero_grad")
            if ctr % FLAGS.gradient_accumulation_steps == 0:
                #update model parameters
                optimizer.step()
                #clear accumulated gradients for next iteration
                optimizer.zero_grad()
            ctr += 1
        optimizer.step()
        optimizer.zero_grad()
        print("calling full_eval in loop")
        full_eval(model, test_dataloader_positives, test_dataloader_negatives,
                  df_test_positive, dev_dataloader_positives,
                  dev_dataloader_negatives, df_validation_positive, results,
                  epoch + 1)


if __name__ == "__main__":
    app.run(main)

#END: COPIED FROM https://github.com/honglizhan/CovidET/tree/main
