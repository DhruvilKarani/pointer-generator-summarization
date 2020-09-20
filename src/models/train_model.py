import torch
import torchtext
import sys
sys.path.append("../")
import features
from features.build_features import SummarizationDataset
import torch.nn as nn
from model import Encoder, VanillaDecoder, Embedding, AttentionDecoder
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score

def idx2string(idxes, itos):
    return " ".join([itos[i] for i in idxes])

class SummarizationNetwork(pl.LightningModule):
  def __init__(self, model, text_vocab_size, summary_vocab_size):
    super().__init__()
    self.text_vocab_size = text_vocab_size
    self.summary_vocab_size = summary_vocab_size
    self.model = model
    self.softmax = nn.Softmax(1)

  def forward(self, x, y):
    return self.model(x, y)

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    return optimizer

  def validation_step(self, batch, batch_idx):
    x, y = batch.text[0], batch.summary[0]
    output = self.model(x, y)
    y=y.squeeze(0)
    loss = F.cross_entropy(output, y)
    y_pred = torch.argmax(self.softmax(output),1)
    result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
    # import pdb; pdb.set_trace()
    y_pred_string = idx2string(y_pred, summ_data.summary_field.vocab.itos)
    y_string = idx2string(y, summ_data.summary_field.vocab.itos)
    meteor = torch.tensor(meteor_score.meteor_score([y_string], y_pred_string))
    result.log('val_loss', loss, prog_bar=True, on_step=True)
    result.log('val_acc', meteor, prog_bar=True, on_step=True)
    return result

  def training_step(self, batch, batch_idx):
    x, y = batch.text[0], batch.summary[0]
    output = self.model(x, y)
    loss = F.cross_entropy(output, y.squeeze(0))
    y_pred = torch.argmax(self.softmax(output),1)
    result = pl.EvalResult(checkpoint_on=loss)
    acc = FM.accuracy(y_pred, y)
    result = pl.TrainResult(minimize=loss)
    result.log('loss', loss)
    result.log('train_acc', acc)
    return result


if __name__ == '__main__':

    BATCH_SIZE = 1

    CSV_PATH = "../../data/processed/data.csv"
    summ_data = SummarizationDataset(CSV_PATH, "text", "summary")
    train_dataset, test_dataset = summ_data.get_datasets()

    summ_data.text_field.build_vocab(train_dataset)
    summ_data.summary_field.build_vocab(train_dataset)


    PAD = summ_data.pad
    SOS = summ_data.sos
    EOS = summ_data.eos
    UNK = "<UNK>"

    EMBEDDING_DIM = 200
    HIDDEN_DIM = 1024
    ENCODER_BIDIRECTIONAL = False 
    NUM_LAYERS = 1
    DECODER_BIDIRECTIONAL = False
    ATTENTION_DIM = 200

    TEXT_VOCAB_SIZE = len(summ_data.text_field.vocab)
    SUMMARY_VOCAB_SIZE = len(summ_data.summary_field.vocab)


    encoder_embedding = Embedding(TEXT_VOCAB_SIZE, EMBEDDING_DIM)
    x = torch.LongTensor([[1,2,3]])
    encoder = Encoder(encoder_embedding, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, ENCODER_BIDIRECTIONAL)
    outputs, ht = encoder(x)
    decoder_embedding = Embedding(SUMMARY_VOCAB_SIZE, EMBEDDING_DIM)
    decoder = VanillaDecoder(decoder_embedding, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DECODER_BIDIRECTIONAL)
    decoder_output = decoder(x, ht)

    attn_decoder = AttentionDecoder(encoder, decoder, ATTENTION_DIM, HIDDEN_DIM, SUMMARY_VOCAB_SIZE)
    iterator = torchtext.data.Iterator(
                            train_dataset, 
                            sort_key = lambda x: len(x.text[0][0]), 
                            batch_size=BATCH_SIZE
                        )

    train_iterator, val_iterator = iterator.splits(
                            (train_dataset, test_dataset), 
                            batch_sizes = (BATCH_SIZE, BATCH_SIZE),
                            sort_key = lambda x: len(x.text[0][0])
                        ) 


    model = SummarizationNetwork(attn_decoder, TEXT_VOCAB_SIZE, SUMMARY_VOCAB_SIZE)

    early_stopping = EarlyStopping('val_loss', patience=3, mode='min')
    trainer = pl.Trainer(gpus=1, max_epochs=10, early_stop_callback=early_stopping)
    trainer.fit(model, train_iterator, val_iterator)

