class LabelCountEncoder(object):
    def __init__(self):
        self.count_dict = {}
    
    def fit(self, column):
        # This gives you a dictionary with level as the key and counts as the value
        count = column.value_counts().to_dict()
        # We want to rank the key by its value and use the rank as the new value
        for rank, key in enumerate(sorted(count.items(), key=lambda x: x[1])):
            self.count_dict[key[0]] = rank+1
         
    
    def transform(self, column):
        # If a category only appears in the test set, we will assign the value to zero.
        missing = 0
        return column.apply(lambda x: self.count_dict.get(x, 0))
        
        
    def fit_transform(self, column):
        self.fit(column)
        return self.transform(column)