class Configuration():
    def __init__(self):
        self.GloveEmbeddingSize = 100
        self.word_emb_size = 100
        self.CNNEmbeddingSize = 100
        self.char_emb_size = 8
        self.MaxSentenceLength = 400
        self.MaxQuestionLength = 20
        self.MaxNumberOfSentences = 8
        self.BatchSize = 50
        self.train_src_file = "../data/train_lines"
        self.dev_src_file = "../data/dev_lines"
        self.glove_path = "../data/glove/"
        self.EPOCHS = 1
        self.hidden_size = 100
        self.numOfHighwayLayers = 2
        self.numOfLSTMLayers = 1
        self.outputDropout = 0.2
        self.is_train = True
        self.emb_mat = None
        self.init_learningRate = 0.5
