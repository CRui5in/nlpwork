import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def find_similar_words(word_embeddings, target_word, top_n=10):
    if target_word not in word_embeddings:
        return []

    target_vector = word_embeddings[target_word].reshape(1, -1)

    similarities = {}
    for word, vector in word_embeddings.items():
        if word == target_word:
            continue

        similarity = cosine_similarity(
            target_vector,
            vector.reshape(1, -1)
        )[0][0]

        similarities[word] = similarity

    sorted_words = sorted(
        similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_words[:top_n]

def load_word_embeddings(load_dir='cn_embeddings'):
    """
    从本地加载词向量

    参数:
    - load_dir: 加载目录

    返回:
    - word_embeddings: 词向量字典
    """
    vocab_path = os.path.join(load_dir, 'vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    vectors_path = os.path.join(load_dir, 'vectors.npy')
    vectors = np.load(vectors_path)

    word_embeddings = dict(zip(vocab, vectors))

    print(f"从 {load_dir} 加载词向量，共 {len(word_embeddings)} 个词")
    return word_embeddings

def visualize_word_embeddings(word_embeddings, top_n=500):
    words = list(word_embeddings.keys())
    vectors = list(word_embeddings.values())

    vectors = np.array(vectors)

    top_n = min(top_n, len(words))

    import random
    sample_indices = random.sample(range(len(words)), top_n)

    selected_words = [words[i] for i in sample_indices]
    selected_vectors = vectors[sample_indices]

    perplexity = max(5, min(30, (top_n - 1) // 3))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity
    )

    reduced_vectors = tsne.fit_transform(selected_vectors)

    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    for i, word in enumerate(selected_words):
        plt.annotate(
            word,
            (reduced_vectors[i, 0], reduced_vectors[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontproperties='Microsoft YaHei'
        )

    plt.title('词向量visualize', fontproperties='Microsoft YaHei')
    plt.xlabel('t-SNE D1', fontproperties='Microsoft YaHei')
    plt.ylabel('t-SNE D2', fontproperties='Microsoft YaHei')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    loaded_embeddings = load_word_embeddings(load_dir='cn_embeddings')

    target_words = ['粮食', '工程', '建设', '水文', '通信']
    for target_word in target_words:
        print(f"\n与'{target_word}'相似的词:")
        similar_words = find_similar_words(loaded_embeddings, target_word)
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")

    visualize_word_embeddings(loaded_embeddings)