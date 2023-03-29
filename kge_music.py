import ampligraph
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from ampligraph.utils import save_model, restore_model
from sklearn.model_selection import train_test_split
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.discovery import query_topn
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation import mrr_score, hits_at_n_score
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer

# print('Ampligraph version: {}'.format(ampligraph.__version__))
df = pd.read_pickle('all_triple.pkl')
data = df.to_numpy()
X_train, X_test = train_test_split_no_unseen(data, test_size=1000)
# model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
#             loss_params={'margin':5})

# Initialize a ComplEx neural embedding
model = ScoringBasedEmbeddingModel(k=20,
                                   eta=20,
                                   scoring_type='TransE')

# Optimizer, loss and regularizer definition
optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = get_loss('multiclass_nll')
regularizer = get_regularizer('LP', {'p': 3, 'lambda': 1e-5})

# Compilation of the model
model.compile(optimizer=optim, loss=loss, entity_relation_regularizer=regularizer)

# Fit the model
model.fit(data,
          batch_size=int(X_train.shape[0] / 50),
          epochs=4000,  # Number of training epochs
          verbose=True  # Enable stdout messages
          )

# Specify triples to filter out of corruptions since true positives
filter_triples = {'test': np.concatenate((X_train, X_test))}

save_model(model, model_name_path = 'model_latest_20.pkl')
# model = restore_model(model_name_path="model_latest.pkl")

#get the entities and their embeddings
all_entities = np.array(list(set(df.values[:, 0]).union(df.values[:, 2])))
embeddings = dict(zip(all_entities, model.get_embeddings(all_entities, embedding_type='e')))
df_embed = pd.DataFrame(embeddings.items(), columns=['entity', 'embeddings'])
print(df_embed.head(5))
df_embed.to_csv('embed_latest_20.csv')
pickle.dump(df_embed, open('embed_latest_20.pkl', 'wb'))

#scoring and ranking the top 10 triples via link prediction to use for recommendation
triples, scores = query_topn(model, top_n=10,
                                 head='b80344d063b5ccb3212f76538f3d9e43d87dca9e',
                                 relation='listens_to',
                                 tail=None,
                                 ents_to_consider=None,
                                 rels_to_consider=None)
print("________________________TransE________________________")
for triple, score in zip(triples, scores):
    print('Score: {} \t {} '.format(score, triple))

# # # MODEL EVALUATION # # #
# Evaluation of the model
ranks = model.evaluate(X_test,
                       use_filter=filter_triples,
                       verbose=True)
mrr = mrr_score(ranks)

print("MRR: %.2f" % (mrr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))


print('done')


'''


'''

"""
print(tf.__version__)
print(tf.config.list_physical_devices())"""
