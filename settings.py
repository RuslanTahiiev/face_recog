import os


class Settings:

    def __init__(self):
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        self.data_path = os.path.join(os.path.dirname(__file__), 'data\\')

    def data_path(self):
        return self.data_path

    def images(self):
        my_list = os.listdir(self.data_path)
        return my_list
