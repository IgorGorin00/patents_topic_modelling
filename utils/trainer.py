from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer, util


def fit_model(corpus, n_neighbors: int=10, min_cluster_size: int=40, min_samples: int=30):
    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0)

    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            prediction_data=True, gen_min_span_tree=True)

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(embedding_model=sentence_model, ctfidf_model=ctfidf_model,
                        umap_model=umap_model, hdbscan_model=hdbscan_model)

    topic_model.fit(corpus)

    return topic_model
