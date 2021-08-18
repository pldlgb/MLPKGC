class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        # 这里可能有问题 relation有重复啊
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
    
    def dis_entities(self):
        train_entities = self.get_entities(self.train_data)
        valid_entities = self.get_entities(self.valid_data)
        test_entities  = self.get_entities(self.test_data)

        dis_val_entities, dis_test_entities = [], []
        for e in valid_entities:
            if e not in train_entities:
                dis_val_entities.append(e)
        
        for e in test_entities:
            if e not in train_entities:
                dis_test_entities.append(e)
        
        print("Valid  :  {}".format(len(dis_val_entities)))
        print(dis_val_entities)
        print("Test   :  {}".format(len(dis_test_entities)))
        print(dis_test_entities)
        print("-"*20)
