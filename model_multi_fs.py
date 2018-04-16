# # Define the RNN Model
from torch.nn import functional
import torch.nn as nn
import torch
from torch.autograd import Variable


class SelfAttentive( nn.Module ):

    def __init__( self, ntoken, ninp, nhid, nlayers, da, r, mlp_nhid, nclass_et, nclass_prt, nclass_popt, emb_matrix, cuda ):
        super( SelfAttentive , self).__init__()
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

        # RNN type
        self.rnn = nn.LSTM( ninp, nhid, nlayers, bias=False, batch_first=True, bidirectional=True )

        # Self Attention Layers
        self.S1 = nn.Linear( nhid * 2, da, bias=False )
        self.S2 = nn.Linear( da, r, bias=False )

        # Task-specific MLP Layers
        # event type
        self.MLP_et = nn.Linear( r * nhid * 2, mlp_nhid )
        self.decoder_et = nn.Linear( mlp_nhid, nclass_et )

        # protest type
        self.MLP_prt = nn.Linear( r * nhid * 2, mlp_nhid )
        self.decoder_prt = nn.Linear( mlp_nhid, nclass_prt )

        # population type
        self.MLP_popt = nn.Linear( r * nhid * 2, mlp_nhid )
        self.decoder_popt = nn.Linear( mlp_nhid, nclass_popt )

        self.init_wordembedding( emb_matrix )
        self.init_weights()

        self.r = r
        self.nhid = nhid
        self.nlayers = nlayers

        if cuda:
            self.cuda()

    def init_weights( self ):
        initrange = 0.1
        self.S1.weight.data.uniform_( -initrange, initrange )
        self.S2.weight.data.uniform_( -initrange, initrange )

        self.MLP_et.weight.data.uniform_( -initrange, initrange )
        self.MLP_et.bias.data.fill_( 0 )

        self.MLP_prt.weight.data.uniform_( -initrange, initrange )
        self.MLP_prt.bias.data.fill_( 0 )

        self.MLP_popt.weight.data.uniform_( -initrange, initrange )
        self.MLP_popt.bias.data.fill_( 0 )

        self.decoder_et.weight.data.uniform_( -initrange, initrange )
        self.decoder_et.bias.data.fill_( 0 )

        self.decoder_prt.weight.data.uniform_( -initrange, initrange )
        self.decoder_prt.bias.data.fill_( 0 )

        self.decoder_popt.weight.data.uniform_( -initrange, initrange )
        self.decoder_popt.bias.data.fill_( 0 )

    def init_wordembedding( self, embedding_matrix ):
        self.encoder.weight.data = embedding_matrix

    def forward(self, input, hidden, len_li ):
        emb = self.encoder( input )

        rnn_input = torch.nn.utils.rnn.pack_padded_sequence( emb, list( len_li.data ), batch_first=True )
        output, hidden = self.rnn( rnn_input , hidden )

        depacked_output, lens = torch.nn.utils.rnn.pad_packed_sequence( output, batch_first=True )

        if self.cuda:
            BM = Variable( torch.zeros( input.size( 0 ) , self.r * self.nhid * 2 ).cuda() )
            penal = Variable( torch.zeros( 1 ).cuda() )
            I = Variable( torch.eye( self.r ).cuda() )
        else:
            BM = Variable( torch.zeros( input.size( 0 ) , self.r * self.nhid * 2 ) )
            penal = Variable( torch.zeros( 1 ) )
            I = Variable( torch.eye( self.r ) )
        weights = {}


        # Attention Block
        # for i in range(sentences):  # sentences can be detected by an <s> token in the tokens list
                                    # compute sentence embedding for each sentence individually in this loop
                                    # combine them to get a document embedding

        for i in range( input.size( 0 ) ):

            H = depacked_output[ i , :lens[ i ], : ]
            s1 = self.S1( H )
            s2 = self.S2( functional.tanh( s1 ) )

            # Attention Weights and Embedding
            A = functional.softmax( s2.t() )
            M = torch.mm( A, H )
            BM[ i, : ] = M.view( -1 )    # embedding for one article

            # Penalization term
            AAT = torch.mm( A, A.t() )
            P = torch.norm( AAT - I, 2 )
            penal += P * P
            weights[ i ] = A

        # Penalization Term
        penal /= input.size( 0 )

        # MLP block for Classifier Feature
        MLPhidden_et = self.MLP_et( BM )    #event type
        decoded_et = self.decoder_et( functional.relu( MLPhidden_et ) )

        MLPhidden_prt = self.MLP_prt(BM)  #protest_type
        decoded_prt = self.decoder_prt(functional.relu( MLPhidden_prt ))

        MLPhidden_popt = self.MLP_popt(BM)  #population_type
        decoded_popt = self.decoder_popt(functional.relu( MLPhidden_popt ))

        return decoded_et, decoded_prt, decoded_popt, hidden, penal, weights

    def init_hidden( self, bsz ):
        weight = next( self.parameters() ).data

        return ( Variable( weight.new( self.nlayers * 2 , bsz, self.nhid ).zero_() ),
                Variable( weight.new( self.nlayers * 2, bsz, self.nhid ).zero_() ) )
