import torch
import numpy as np
import os
import json
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class BertWordEmbedding:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_word_embedding(self, corpus):
        word_embeddings = {}

        for line in corpus:
            words = line.split()

            for word in words:
                if len(word) == 0 or word in [',', '。', '、']:
                    continue

                if word in word_embeddings:
                    continue

                try:
                    inputs = self.tokenizer(word, return_tensors='pt', truncation=True, max_length=512).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        hidden_states = outputs.last_hidden_state

                    word_vector = hidden_states[0].mean(dim=0).cpu().numpy()
                    word_embeddings[word] = word_vector

                except Exception as e:
                    print(f"处理词 '{word}' 时出错: {e}")

        return word_embeddings

    def save_word_embeddings(self, word_embeddings, save_dir):
        """
        保存词向量到本地

        参数:
        - word_embeddings: 词向量字典
        - save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        vectors_path = os.path.join(save_dir, 'vectors.npy')
        np.save(vectors_path, list(word_embeddings.values()))

        vocab_path = os.path.join(save_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(list(word_embeddings.keys()), f, ensure_ascii=False)

        print(f"词向量已保存到 {save_dir}")

    def load_word_embeddings(self, load_dir):
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

    def find_similar_words(self, word_embeddings, target_word, top_n=10):
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

    def visualize_word_embeddings(self, word_embeddings, top_n=500):
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


def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences


def main():
    corpus_path = 'data/en.txt'

    corpus = load_corpus(corpus_path)

    bert_embedder = BertWordEmbedding()

    word_embeddings = bert_embedder.get_word_embedding(corpus)

    bert_embedder.save_word_embeddings(word_embeddings, save_dir='en_embeddings')

    loaded_embeddings = bert_embedder.load_word_embeddings(load_dir="en_embeddings")

    target_words = ['food', 'engineer', 'build']
    for target_word in target_words:
        print(f"\n与'{target_word}'相似的词:")
        similar_words = bert_embedder.find_similar_words(loaded_embeddings, target_word)
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")

    bert_embedder.visualize_word_embeddings(loaded_embeddings)


if __name__ == '__main__':
    main()