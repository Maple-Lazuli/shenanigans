import matplotlib.pyplot as plt
import numpy as np
import hashlib
from datetime import datetime
plt.style.use('ggplot')


class Metric:
    def __init__(self, title, horizontal_label, vertical_label):
        self.title = title
        self.x = horizontal_label
        self.y = vertical_label
        self.metrics = dict()

    def add(self, key, value):
        """
        Adds a value to the metrics dictionary with the assumption

        Parameters
        ----------
        key
        value

        Returns
        -------

        """
        if key not in self.metrics.keys():
            self.metrics[key] = [value]
        else:
            self.metrics[key].append(value)

    def create_plot(self, report_location):
        fig, ax = plt.subplots(figsize=(10, 5))

        for key in self.metrics.keys():
            ax.plot(self.metrics[key], label=key)

        ax.set_title(self.title)
        ax.set_xlabel(self.x, fontsize=14)
        ax.set_ylabel(self.y, fontsize=14)
        ax.legend(loc='best', fontsize=12)
        # Create the save file name
        # use the current time as the seed
        now = datetime.now()
        # use the date hash as the file name
        date_hash = hashlib.md5(str(now).encode())
        image_name = date_hash.hexdigest()

        full_name = f"{report_location}images/{image_name}.png"
        plt.savefig(full_name)

        return f"images/{image_name}.png"

