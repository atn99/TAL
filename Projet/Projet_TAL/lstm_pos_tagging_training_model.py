import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Utiliser Fast Text pour le carractere embedding

# %% [markdown]
# #1 Prepare data

# %%
#donne les index de seq et les mets dans un tensor
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[seq]]
    return torch.tensor(idxs, dtype=torch.long)

#affiche
def affiche(losses_eval, losses_train):
    plt.plot(np.arange(0,len(losses_eval)),losses_eval)
    plt.title('Evolution loss evaluation set')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(np.arange(0,len(losses_train)),losses_train)
    plt.title('Evolution loss train set')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# %%
data_file = open('pos_reference.txt.lima', 'r')
data = []
word_to_ix = {}
tag_to_ix = {}

for line in data_file : 
    # Pour ne pas prendre les lignes vides
    if not line.isspace():

        # On retire les retours chariots à la fin des lignes + separation du mot et de sa forme morpho-syntaxique dans un tuple
        wordAndToken = line.rstrip('\n').split('\t')

        # Si le mot n'est pas dans le dico on l'ajoute avec son index qui est la taille actuelle du dico
        if wordAndToken[0] not in word_to_ix:
            word_to_ix[wordAndToken[0]] = len(word_to_ix)

        # Pareil pour les tokens
        if wordAndToken[1] not in tag_to_ix:
            tag_to_ix[wordAndToken[1]] = len(tag_to_ix)

        data.append(wordAndToken)

# %%
print(tag_to_ix)

# %%
#print(word_to_ix)

# %%
for i in range(10):
    print(data[i])

# %% [markdown]
# Séparation des trains set, test set et eval set

# %%
nb_line = len(data)
line_80 = round((nb_line*80)/100)
line_10 = round((nb_line*10)/100)  

trainSet = data[:line_80]
testSet = data[line_80+1:line_80+line_10]
evalSet = data[line_80+line_10+1:]

print(trainSet[0])
print(np.shape(trainSet))
print(evalSet[0])
print(np.shape(evalSet))

# %%
def train(model, nb_epoch):

    losses_eval = []
    losses_train = []
    current_loss = 0
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(nb_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        print("Epoch : " + str(epoch) +'/'+ str(nb_epoch) + '\n')
        
        #On passe le model en phase d'entrainement
        model.train()
        for word, tag in trainSet:
            #sentence contient les mots
            #tags contient la reponse

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()
            
            
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            word_in_train = prepare_sequence(word, word_to_ix)
            target_train = prepare_sequence(tag, tag_to_ix)
            
            # Step 3. Run our forward pass.
            tag_scores_train = model(word_in_train)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss_train = loss_function(tag_scores_train, target_train)
            current_loss += loss_train.item()

            loss_train.backward()
            optimizer.step()
        
        mean_train_loss = current_loss/len(evalSet)
        losses_train.append(mean_train_loss)
        print('Train Loss after this epoch : ' + str(mean_train_loss) )
        current_loss = 0.0
        
        optimizer.zero_grad()

        #On passe le model en phase d'evaluation
        model.eval() 
        current_loss = 0.0
        i = 0

        with torch.no_grad():
            for word, tag in evalSet:

                word_in_eval = prepare_sequence(word, word_to_ix)
                target_eval = prepare_sequence(tag, tag_to_ix)

                tag_scores_eval = model(word_in_eval)
                
                loss_eval = loss_function(tag_scores_eval, target_eval)
                current_loss += loss_eval.item()  
                
            mean_eval_loss = current_loss/len(evalSet)
            losses_eval.append(mean_eval_loss)
            print('Evaluation Loss after this epoch : ' + str(mean_eval_loss) )
            current_loss = 0.0

    return losses_train, losses_eval



# %%
################# Create the model #################
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# %%
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

################# Train the model #################
model1 = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))


# %%
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(trainSet[0][0], word_to_ix)
    print(inputs)
    tag_scores = model1(inputs)
    print()
    print("=> Scores before training of the tags affected to each word")
    print(tag_scores)

# %%
loss_train, loss_eval = train(model1, 2)

# %%
affiche(loss_eval, loss_train)

# %%
# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(trainSet[4][0], word_to_ix)
    print()
    print('=> The sentence to analyze (first sentence of the Training data):')
    print(trainSet[4])

    print()
    print("=> Training data: each word is assigned to a unique index:")
    #print(word_to_ix)
    tag_scores = model1(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!

    print()
    print("=> Scores after training of the tags affected to each word of the sentence to analyze:")
    print(tag_scores)

# %%
#Dans le tensor ci-dessus, le score le plus haut correspond au resultat le plus probable.

#Penser a ajouter le eval, rappel, f1_score, precision

# %%
######## Création du model avec GLove

class LSTMTaggerGLove(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, weight):
        super(LSTMTaggerGLove, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.load_state_dict({'weight': weight}) #de taille vocab_size, donc marche pas

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#Chargement des donnees de GloVe
emmbed_dict = {}
embedding_dim = 50
path = '../../../glove.6B.50d.txt' #mettre le path des mots de Glove
with open(path,'r', encoding = "UTF-8") as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    emmbed_dict[word]=vector
      

print(emmbed_dict['will'])


#Création et remplissage de la matrice des poids
'''
Pour chaque mot du vocabulaire de l’ensemble de données, nous vérifions s’il est sur le vocabulaire de GloVe. 
S’il le fait, nous chargeons son vecteur de mots pré-entraîné. 
Sinon, nous initialisons un vecteur aléatoire.
'''

matrix_len = len(word_to_ix)
print(matrix_len)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(word_to_ix):
    try: 
        weights_matrix[i] = emmbed_dict[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

#Conversion en tensor pour l'embedding
weight = torch.FloatTensor(weights_matrix)

print(words_found)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 50
HIDDEN_DIM = 6

################# Train the model #################
model2 = LSTMTaggerGLove(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), weight)


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(trainSet[0][0], word_to_ix)
    print(inputs)
    tag_scores = model2(inputs)
    print()
    print("=> Scores before training of the tags affected to each word")
    print(tag_scores)


loss_train, loss_eval = train(model2, 10)


affiche(loss_train, loss_eval)


ono
