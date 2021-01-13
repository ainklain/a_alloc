
import re
import os


class Configs:
    def __init__(self, name):
        self.comment = """
        non_cash only
        """

        self.seed = 1000
        self.name = name
        self.pre_lr = 5e-3
        self.lr = 5e-2
        self.batch_size = 512
        self.num_epochs = 1000
        self.base_i0 = 2000
        self.mc_samples = 1000
        self.sampling_freq = 20
        self.k_days = 20
        self.label_days = 20
        self.strategy_days = 250

        # adaptive / earlystopping
        self.adaptive_flag = True
        self.adaptive_count = 30
        self.adaptive_lrx = 5  # learning rate * 배수
        self.es_max_count = 200

        self.retrain_days = 240
        self.test_days = 5000  # test days
        self.init_train_len = 500
        self.train_data_len = 2000
        self.normalizing_window = 500  # norm windows for macro data_conf
        self.use_accum_data = True  # [sampler] 데이터 누적할지 말지
        self.adv_train = True
        self.n_pretrain = 5
        self.max_entropy = True

        self.loss_threshold = None  # -1

        self.datatype = 'app'
        # self.datatype = 'inv'

        self.cost_rate = 0.003
        self.plot_freq = 10
        self.eval_freq = 1  # 20
        self.save_freq = 20
        self.model_init_everytime = False
        self.use_guide_wgt_as_prev_x = False  # models / forward_with_loss

        # self.hidden_dim = [72, 48, 32]
        self.hidden_dim = [128, 64, 64]
        self.alloc_hidden_dim = [128, 64]
        self.dropout_r = 0.3

        self.random_guide_weight = 0.1
        self.random_flip = 0.1  # flip sign
        self.random_label = 0.1  # random sample from dist.

        self.clip = 1.

        # logger
        self.log_level = 'DEBUG'

        ## attention
        self.d_model = 128
        self.n_heads = 8
        self.d_k = self.d_v = self.d_model // self.n_heads
        self.d_ff = 128

        self.loss_list = ['y_pf', 'mdd_pf', 'logy', 'wgt_guide', 'cost', 'entropy']
        self.adaptive_loss_wgt = {'y_pf': 0., 'mdd_pf': 0., 'logy': 0., 'wgt_guide': 0.5, 'cost': 10., 'entropy': 0.0}
        self.loss_wgt = {'y_pf': 1, 'mdd_pf': 1., 'logy': 1., 'wgt_guide': 0.01, 'cost': 1., 'entropy': 0.001}

        self.init()
        self.set_path()

    def init(self):
        self.init_weight()

    def init_weight(self):
        if self.datatype == 'app':
            self.cash_idx = 3
            self.base_weight = [0.25, 0.1, 0.05, 0.6]
        else:
            self.cash_idx = 0
            self.base_weight = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
            # self.base_weight = None

    def set_path(self):
        self.outpath = './out/{}/'.format(self.name)
        os.makedirs(self.outpath, exist_ok=True)

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str

    def load(self, model_path):
        from ast import literal_eval
        # load and parse 'c.txt'
        p = re.compile('[a-zA-Z0-9_]+[:]')
        with open(os.path.join(model_path, 'c.txt'), 'r') as f:
            print('{} loaded'.format(os.path.join(model_path, 'c.txt')))
            texts = ''.join(f.readlines())

        tags = p.findall(texts)

        for tag in tags[::-1]:
            *texts, val = texts.split(tag)
            texts = tag.join(texts)

            a = tag.split(':')[0]
            if a in ['outpath', 'name']:
                continue

            val = val.strip()
            if val in ['True', 'False']:
                self.__setattr__(a, bool(val))
            elif re.match("\[.+\]", val) is not None or re.match("\{.+\}", val) is not None:
                self.__setattr__(a, literal_eval(val))
            elif re.fullmatch("[0-9.]+", val):
                if "." in val:
                    self.__setattr__(a, float(val))
                else:
                    self.__setattr__(a, int(val))
            else:
                self.__setattr__(a, val)

