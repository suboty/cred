class Statistics:
    def __init__(self):
        self.stats = {}

    def add(
            self,
            name: str,
            value: float,
            tip: str = 'silh'
    ):
        if self.stats.get(tip) is None:
            self.stats[tip] = {}

        self.stats[tip][name] = value

    def get(
            self,
            tip: str = 'silh'
    ):
        return self.stats[tip]


stats = Statistics()
