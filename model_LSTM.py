# __description__: LSTM model with max over-time pooling

from torch.nn import functional
import torch.nn as nn
import torch
from torch.autograd import Variable



class LSTM( nn.Module ):

    def __init__( self, ntoken, ninp, nhid, nlayers, mlp_nhid, nclass, emb_matrix, cuda ):
        super( LSTM , self).__init__()
        """
        Args:
            ntoken:
            ninp:
            nhid:
            nlayers:
            da:
            r:
            mlp_nhid:
            nclass:
            emb_matrix:
            cuda:
        """


        # Embedding Layer
        self.encoder = nn.Embedding( ntoken, ninp )
        self.nlayers = nlayers
        self.nhid = nhid

        # RNN type
        self.rnn = nn.LSTM( ninp, nhid, nlayers, bias=False, batch_first=True, bidirectional=True )
#         # max pool layer
#         self.max_pool = nn.MaxPool1d(2*nhid, stride=1)

        # Final MLP Layers
        self.MLP = nn.Linear( nhid * 2, mlp_nhid, bias=True )
        self.decoder = nn.Linear( mlp_nhid, nclass )

        self.init_wordembedding( emb_matrix )
        self.init_weights()

        if cuda:
            self.cuda()

    def init_weights( self ):
        initrange = 0.1

        self.MLP.weight.data.uniform_( -initrange, initrange )
        self.MLP.bias.data.fill_( 0 )

        self.decoder.weight.data.uniform_( -initrange, initrange )
        self.decoder.bias.data.fill_( 0 )

    def init_wordembedding( self, embedding_matrix ):
        self.encoder.weight.data = embedding_matrix

    def forward(self, input, hidden, len_li ):
        emb = self.encoder( input )

        rnn_input = torch.nn.utils.rnn.pack_padded_sequence( emb, list( len_li.data ), batch_first=True )
        output, (hidden, cell) = self.rnn( rnn_input , hidden)

        depacked_output, lens = torch.nn.utils.rnn.pad_packed_sequence( output, batch_first=True )
        if self.cuda:
            BM = Variable( torch.zeros( input.size( 0 ) , self.nhid * 2 ).cuda() )
        else:
            BM = Variable( torch.zeros( input.size( 0 ) , self.nhid * 2 ) )


        # MLP block for Classifier Feature
        for i in range( input.size( 0 ) ):
          H = depacked_output[ i , :lens[ i ], : ]
          H = H.unsqueeze(0) # (1, seq_len, 2*nhid)
          H_ = functional.max_pool2d(H, kernel_size=(H.size(1), 1), stride=1)
          H_ = H_.squeeze(0)
          BM[i, :] = H_

        MLPhidden = self.MLP( BM )
        decoded = self.decoder( functional.relu( MLPhidden ) )

        return decoded

    def init_hidden( self, bsz ):
        weight = next( self.parameters() ).data

        return ( Variable( weight.new( self.nlayers * 2 , bsz, self.nhid ).zero_() ),
                Variable( weight.new( self.nlayers * 2, bsz, self.nhid ).zero_() ) )
