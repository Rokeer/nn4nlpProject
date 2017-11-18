class Configuration():
    def __init__(self):
        self.GloveEmbeddingSize = 100
        self.word_emb_size = 100
        self.CNNEmbeddingSize = 100
        self.char_emb_size = 8
        self.char_vocab_size = 300
        self.cnn_dropout_keep_prob = 0.8
        self.use_char_emb = True
        self.padding = 0

        self.MaxSentenceLength = 400
        self.MaxQuestionLength = 20
        self.MaxNumberOfSentences = 1
        self.max_word_size = 16
        self.BatchSize = 5
        self.train_src_file = "../data/train_lines"
        self.dev_src_file = "../data/dev_lines"
        self.glove_path = "../data/glove/"
        self.EPOCHS = 50
        self.hidden_size = 200
        self.numOfHighwayLayers = 2
        self.numOfLSTMLayers = 1
        self.outputDropout = 0.2
        self.is_train = True
        self.emb_mat = None
        self.init_learningRate = 0.5
        self.DevBatchSize = 5
        self.searchSize = 10
