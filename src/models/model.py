import torch
import torch.nn as nn
import torch.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embed(x)


class Encoder(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, bidirectional):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.embedding = embedding

    def forward(self, x):
        x = self.embedding(x)
        outputs, (ht,_) = self.lstm(x)
        return outputs, ht

class VanillaDecoder(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, bidirectional):
        super(VanillaDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.embedding = embedding
        # self.linear = nn.Linear((int(bidirectional)+1)*hidden_dim, embedding.vocab_size)

    def forward(self, x, encoder_input):
        x = self.embedding(x)
        outputs, (ht,_) = self.lstm(x, (encoder_input,torch.zeros_like(encoder_input)))
        # outputs = self.linear(outputs)
        return outputs, ht


class AttentionDecoder(nn.Module):
    def __init__(self, encoder, decoder, attn_dim, hidden_dim, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn_dim = attn_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.Wh = nn.Linear( hidden_dim, attn_dim, bias=False)
        self.Ws = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.b = nn.Linear(1, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0)
        self.linear1 = nn.Linear(self.hidden_dim*2, self.vocab_size*2)
        self.linear2 = nn.Linear(self.vocab_size*2, self.vocab_size)

    def encoder_hidden(self, x):
        encoder_hidden_states = []
        for i in x[0]:
            encoder_input = i.unsqueeze(0).unsqueeze(0)
            _, encoder_ht = self.encoder(encoder_input)
            encoder_hidden_states.append(encoder_ht)
        return encoder_hidden_states

    def forward(self, x, y):
        encoder_hidden_states = self.encoder_hidden(x)
        encoder_hidden_last = encoder_hidden_states[-1]
        decoder_output, _ = self.decoder(y, encoder_hidden_last)
        final_outputs = []

        for j, idx in enumerate(y[0]):
            attention_weights = []
            decoder_step_output = decoder_output[0, j, :].unsqueeze(0)
            for i, ht in enumerate(encoder_hidden_states):
                ht = ht.squeeze(1).view(1, -1)
                # import pdb; pdb.set_trace()
                ei = self.Wh(ht) + self.Ws(decoder_step_output) + self.b(torch.ones(1, 1).cuda())
                ei = self.tanh(ei)
                ei = self.v(ei).squeeze(0)
                attention_weights.append(ei)
        
            attention_weights = torch.Tensor(attention_weights)
            attention_weights = self.softmax(attention_weights)
            context_vector = [ht*ai for ai, ht in zip(attention_weights, encoder_hidden_states)]
            context_vector = sum(context_vector)
            context_vector = context_vector.view(1,-1)
            concat_context = torch.cat((context_vector, decoder_step_output), 1)
            # import pdb; pdb.set_trace()
            output = self.linear1(concat_context)
            output = self.linear2(output)
            final_outputs.append(output)
        return torch.cat(final_outputs, 0)



if __name__ == '__main__':
    embedding = Embedding(10,10)
    x = torch.LongTensor([[1,2,3]])
    encoder = Encoder(embedding, 10, 10, 2, True)
    outputs, ht = encoder(x)
    decoder = VanillaDecoder(embedding, 10, 10, 2, True)
    decoder_output = decoder(x, ht)

    attn = AttentionDecoder(encoder, decoder, 20, 10, 100)

    ao = attn(x, x)

    print(ao)
